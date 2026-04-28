# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data classes for MooncakeStoreConnector."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, cast

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


@dataclass
class KeyMetadata:
    """Metadata for constructing pool keys."""

    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int


@dataclass(order=True)
class PoolKey:
    """Key for addressing KV cache blocks in the distributed store."""

    key_metadata: KeyMetadata
    chunk_hash: str

    def __hash__(self):
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.chunk_hash,
            )
        )

    def to_string(self) -> str:
        return (
            f"{self.key_metadata.model_name}"
            f"@tp_rank:{self.key_metadata.tp_rank}"
            f"@pcp{self.key_metadata.pcp_rank}"
            f"@dcp{self.key_metadata.dcp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}"
            f"@{self.chunk_hash}"
        )


@dataclass
class GroupLayout:
    """Per-KV-cache-group memory layout.

    base_addrs and block_lens are flattened across the group's layers and
    K/V segments; entry i is one (layer, K|V) pair.
    """

    base_addrs: list[int]
    block_lens: list[int]


class ChunkedTokenDatabase:
    """Maps token positions to store keys and GPU memory addresses."""

    def __init__(self, metadata: KeyMetadata, block_size: int):
        self.metadata = metadata
        self.block_size = block_size
        # Legacy flat fields, kept for backwards-compatible callers (set via
        # set_kv_caches_base_addr / set_block_len). Populated automatically
        # by set_groups for new callers.
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        # New per-group fields. layer_to_group has the same length as
        # kv_caches_base_addr / block_len; entry i tells which group the
        # registered segment i belongs to.
        self.groups: list[GroupLayout] = []
        self.layer_to_group: list[int] = []

    def _make_key_by_hash(self, chunk_hash: str) -> PoolKey:
        return PoolKey(self.metadata, chunk_hash)

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        """Legacy setter — single-group flat layout."""
        self.kv_caches_base_addr = kv_caches_base_addr
        if not self.groups or len(self.groups) != 1:
            self.groups = [
                GroupLayout(base_addrs=kv_caches_base_addr, block_lens=self.block_len)
            ]
            self.layer_to_group = [0] * len(kv_caches_base_addr)

    def set_block_len(self, block_len: list[int]):
        """Legacy setter — single-group flat layout."""
        self.block_len = block_len
        if self.groups and len(self.groups) == 1:
            self.groups[0].block_lens = block_len
            self.layer_to_group = [0] * len(self.kv_caches_base_addr)

    def set_groups(
        self,
        groups: list[GroupLayout],
        layer_to_group: list[int],
    ) -> None:
        """Group-aware setter — flattens groups into kv_caches_base_addr/block_len."""
        self.groups = groups
        self.layer_to_group = layer_to_group
        flat_addrs: list[int] = []
        flat_lens: list[int] = []
        for layout in groups:
            flat_addrs.extend(layout.base_addrs)
            flat_lens.extend(layout.block_lens)
        self.kv_caches_base_addr = flat_addrs
        self.block_len = flat_lens

    @staticmethod
    def _normalize_per_group(
        block_ids: list[list[int]] | list[int],
    ) -> list[list[int]]:
        """Coerce a flat list[int] to a single-group list[list[int]]."""
        if block_ids and not isinstance(block_ids[0], list):
            return [cast(list[int], block_ids)]
        return cast(list[list[int]], block_ids)

    def prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[list[int]] | list[int],
    ) -> tuple[list[int], list[int], int]:
        """Compute memory addresses and sizes for a token range.

        block_ids is per-group (list[list[int]]). For single-group callers
        passing list[int], we wrap it.
        """
        per_group = self._normalize_per_group(block_ids)

        chunk_id = start // self.block_size
        # FA group is the longest; offset trims older chunks for SWA groups.
        total_chunks = max(len(g) for g in per_group) if per_group else 0
        per_group_local: list[int] = []
        for group in per_group:
            offset = total_chunks - len(group)
            local_i = chunk_id - offset
            per_group_local.append(local_i if 0 <= local_i < len(group) else -1)

        addr_list: list[int] = []
        size_list: list[int] = []
        last_block_id = 0
        for seg_idx, base_addr in enumerate(self.kv_caches_base_addr):
            g = self.layer_to_group[seg_idx] if self.layer_to_group else 0
            local_i = per_group_local[g]
            if local_i < 0:
                # Defensive: caller (worker) should have filtered via
                # is_chunk_savable. Emit zeros so list shape is preserved.
                addr_list.append(0)
                size_list.append(0)
                continue
            block_id = per_group[g][local_i]
            block_len = self.block_len[seg_idx]
            addr = base_addr + block_id * block_len
            size = int(block_len / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
            last_block_id = block_id
        return addr_list, size_list, last_block_id

    def is_chunk_savable(
        self,
        start: int,
        block_ids: list[list[int]] | list[int],
    ) -> bool:
        """A chunk is savable iff it is in window for every group."""
        per_group = self._normalize_per_group(block_ids)
        if not per_group:
            return False
        chunk_id = start // self.block_size
        total_chunks = max(len(g) for g in per_group)
        for group in per_group:
            offset = total_chunks - len(group)
            local_i = chunk_id - offset
            if not (0 <= local_i < len(group)):
                return False
        return True

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        mask_num: int = 0,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process tokens and yield (start_idx, end_idx, pool_key) tuples.

        Args:
            token_len: Total number of tokens.
            block_hashes: Block hashes for each block.
            mask_num: Number of tokens to skip from the beginning.
        """
        if not block_hashes:
            return
        if not isinstance(block_hashes[0], str):
            block_hashes = [
                h.hex()  # type: ignore[union-attr]
                for h in block_hashes
            ]
        for chunk_id, hash_val in enumerate(block_hashes):
            start_idx = chunk_id * self.block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + self.block_size, token_len)
            if start_idx < mask_num:
                continue
            else:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)


@dataclass
class LoadSpec:
    """Specification for loading KV cache from external store."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    can_load: bool
    token_len: int = 0


@dataclass
class RequestTracker:
    """Tracks per-request state across scheduler ticks."""

    req_id: str
    token_len: int
    # One list of block ids per KV cache group. Single-group models use [[...]]
    allocated_block_ids: list[list[int]]
    num_saved_tokens: int = 0
    token_ids: list[int] | None = None

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[list[int]] | list[int],
    ) -> None:
        """Append new blocks to each group.

        Accepts:
          - tuple/list of per-group block lists (HMA shape)
          - flat list[int] for legacy single-group callers
        """
        if len(new_block_ids) == 0:
            return

        if isinstance(new_block_ids, tuple):
            per_group = list(new_block_ids)
        elif isinstance(new_block_ids, list) and isinstance(new_block_ids[0], list):
            per_group = cast(list[list[int]], new_block_ids)
        elif isinstance(new_block_ids, list):
            self.allocated_block_ids[0].extend(cast(list[int], new_block_ids))
            return
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        assert len(per_group) == len(self.allocated_block_ids), (
            f"KV group count mismatch on update: got {len(per_group)} "
            f"groups, tracker has {len(self.allocated_block_ids)}"
        )
        for g, group_blocks in enumerate(per_group):
            self.allocated_block_ids[g].extend(group_blocks)


@dataclass
class ReqMeta:
    """Per-request metadata for store put/get operations."""

    req_id: str
    token_len_chunk: int
    block_ids: list[list[int]]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    is_last_chunk: bool | None = None
    current_event: torch.cuda.Event | None = None

    token_ids: list[int] | None = None
    original_block_size: int | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: int | None = None,
    ) -> Optional["ReqMeta"]:
        """Create ReqMeta from a RequestTracker."""
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, block_size) * block_size
            if discard_partial_chunks
            else 0
        )
        num_tokens_to_save = (
            (input_token_len // block_size * block_size)
            if discard_partial_chunks
            else input_token_len
        )

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            load_spec = None

        logger.debug(
            "request:%s, meta save spec:%s, meta load spec:%s",
            tracker.req_id,
            not skip_save,
            load_spec,
        )
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
        )


class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker."""

    def __init__(
        self,
        unfinished_request_ids: set[str],
        preempted_req_ids: set[str],
    ):
        self.requests: list[ReqMeta] = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        self.requests.append(req_meta)
