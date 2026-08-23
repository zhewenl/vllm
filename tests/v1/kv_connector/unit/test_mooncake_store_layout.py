# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts."""

import ctypes
import random

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BHLNCStoreLayout,
    BLHNCStoreLayout,
    BLNHCStoreLayout,
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    LHBNCStoreLayout,
    MambaStoreLayout,
    RankLocalStoreLayout,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_layout import KVCacheLayout

BLOCK_SIZE = 128


def _make_gdn_store_layout(
    *, local_tp_size: int, tp_rank: int
) -> tuple[MambaStoreLayout, torch.Tensor]:
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    local_factor = 4 // local_tp_size
    spec = MambaSpec(
        block_size=16,
        shapes=((6 * local_factor, 3), (local_factor, 2, 2)),
        dtypes=(torch.uint8, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    tensor = torch.empty((2, 1, 1, spec.state_content_size_bytes), dtype=torch.uint8)
    metadata = KeyMetadata(
        "test-model",
        tp_rank,
        0,
        0,
        0,
        group_id=1,
        store_namespace="@store_tp:4@store_pp:1@store_format:tp_shared_hybrid_lbhnc",
    )
    layout = MambaStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=16,
        local_tp_size=local_tp_size,
        store_tp_size=4,
        tp_rank=tp_rank,
        spec=spec,
    )
    layout.register_kv_caches([tensor], 2)
    return layout, tensor


def _descriptors_for_block(
    layout: MambaStoreLayout, shard_id: int
) -> tuple[list[int], list[int]]:
    addrs, sizes, _ = layout.prepare_values([(0, layout.block_size)], [0], [shard_id])
    return addrs[0], sizes[0]


def _read_segments(addrs: list[int], sizes: list[int]) -> bytes:
    return b"".join(
        ctypes.string_at(addr, size) for addr, size in zip(addrs, sizes, strict=True)
    )


def test_tp_shared_layout_loads_partial_prefix_from_physical_block():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        16,
        16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        num_kv_heads=8,
    )
    tensor = torch.empty((2, 4, 16, 4), dtype=torch.float16)
    layout.register_kv_caches([tensor], 2)
    shard_ids = layout.local_shard_ids

    addrs, sizes, block_ids = layout.prepare_values(
        [(0, 8)] * len(shard_ids), [1], shard_ids
    )

    assert len(addrs) == len(sizes) == len(shard_ids)
    assert block_ids == [1] * len(shard_ids)


@pytest.mark.parametrize(("producer_tp", "consumer_tp"), [(4, 2), (2, 4)])
def test_gdn_store_shards_round_trip_in_both_tp_directions(
    monkeypatch, producer_tp: int, consumer_tp: int
):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    block_hash = BlockHash(b"h")
    stored: dict[int, bytes] = {}
    keys: dict[int, str] = {}
    for tp_rank in range(producer_tp):
        layout, cache = _make_gdn_store_layout(
            local_tp_size=producer_tp, tp_rank=tp_rank
        )
        cache.copy_(torch.arange(cache.numel(), dtype=torch.uint8).view_as(cache))
        cache.add_(tp_rank * 32)
        for shard_id in layout.local_shard_ids:
            addrs, sizes = _descriptors_for_block(layout, shard_id)
            stored[shard_id] = _read_segments(addrs, sizes)
            keys[shard_id] = layout.key_for(shard_id, block_hash)

    assert set(stored) == set(range(4))
    for tp_rank in range(consumer_tp):
        layout, cache = _make_gdn_store_layout(
            local_tp_size=consumer_tp, tp_rank=tp_rank
        )
        cache.zero_()
        for shard_id in layout.local_shard_ids:
            assert layout.key_for(shard_id, block_hash) == keys[shard_id]
            addrs, sizes = _descriptors_for_block(layout, shard_id)
            offset = 0
            for addr, size in zip(addrs, sizes, strict=True):
                ctypes.memmove(addr, stored[shard_id][offset : offset + size], size)
                offset += size
            assert _read_segments(addrs, sizes) == stored[shard_id]


def test_gdn_store_shard_segments_preserve_projection_boundaries(monkeypatch):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    layout, cache = _make_gdn_store_layout(local_tp_size=2, tp_rank=0)

    shard0_addrs, sizes = _descriptors_for_block(layout, 0)
    shard1_addrs, shard1_sizes = _descriptors_for_block(layout, 1)

    assert sizes == shard1_sizes == [6, 6, 6, 4]
    assert [addr - cache.data_ptr() for addr in shard0_addrs] == [0, 12, 24, 36]
    assert [addr - cache.data_ptr() for addr in shard1_addrs] == [6, 18, 30, 40]


def _rank_local_layout(num_regions: int, num_block_lens: int) -> RankLocalStoreLayout:
    metadata = KeyMetadata("test-model", 1, 0, 0, 0)
    layout = RankLocalStoreLayout(metadata, BLOCK_SIZE, BLOCK_SIZE)
    layout.set_kv_caches_base_addr(
        [0x7F00_0000_0000 + i * (1 << 30) for i in range(num_regions)]
    )
    layout.set_block_len([30_208 + 512 * i for i in range(num_block_lens)])
    return layout


def _rank_local_reference(
    layout: RankLocalStoreLayout, start: int, end: int, block_ids: list[int]
) -> tuple[list[int], list[int], int]:
    block_id = block_ids[start // layout.block_size]
    length = len(layout.block_len)
    addrs = [
        base_addr + block_id * layout.block_len[index % length]
        for index, base_addr in enumerate(layout.kv_caches_base_addr)
    ]
    sizes = [
        layout.block_len[index % length] * cdiv(end - start, layout.block_size)
        for index in range(len(layout.kv_caches_base_addr))
    ]
    return addrs, sizes, block_id


@pytest.mark.parametrize("num_regions,num_block_lens", [(96, 96), (96, 2), (1, 1)])
def test_rank_local_descriptors_match_reference(num_regions: int, num_block_lens: int):
    layout = _rank_local_layout(num_regions, num_block_lens)
    rng = random.Random(0)
    block_ids = [rng.randrange(0, 1 << 20) for _ in range(300)]
    chunks = []
    block = 0
    while block < len(block_ids) - 4:
        span = rng.choice([1, 1, 1, 2, 4])
        chunks.append((block * BLOCK_SIZE, (block + span) * BLOCK_SIZE))
        block += span + rng.choice([0, 1])

    addrs, sizes, selected_blocks = layout.prepare_values(
        chunks, block_ids, [0] * len(chunks)
    )

    for chunk, chunk_addrs, chunk_sizes, block_id in zip(
        chunks, addrs, sizes, selected_blocks, strict=True
    ):
        assert (chunk_addrs, chunk_sizes, block_id) == _rank_local_reference(
            layout, *chunk, block_ids
        )
        assert all(type(addr) is int for addr in chunk_addrs)
        assert type(block_id) is int


def test_rank_local_descriptors_handle_empty_and_invalid_chunks():
    layout = _rank_local_layout(4, 4)
    assert layout.prepare_values([], [1, 2, 3], []) == ([], [], [])
    with pytest.raises(AssertionError):
        layout.prepare_values([(0, BLOCK_SIZE + 1)], [0, 1], [0])


def test_rank_local_database_api_rejects_tp_shared_layout():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        num_kv_heads=8,
    )
    database = ChunkedTokenDatabase(metadata, 16, store_layout=layout)

    with pytest.raises(RuntimeError, match="rank-local"):
        database.prepare_value_for_block(0)


_SHARED_LAYOUTS = [
    (KVCacheLayout.LBHNC, LBHNCStoreLayout),
    (KVCacheLayout.LBNHC, LBNHCStoreLayout),
    (KVCacheLayout.BLHNC, BLHNCStoreLayout),
    (KVCacheLayout.BLNHC, BLNHCStoreLayout),
    (KVCacheLayout.LHBNC, LHBNCStoreLayout),
    (KVCacheLayout.BHLNC, BHLNCStoreLayout),
]


def _physical_layer_views(
    layout: KVCacheLayout,
    num_layers: int,
    num_blocks: int,
    num_heads: int,
    block_size: int,
    content_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    logical_shape = (num_layers, num_blocks, num_heads, block_size, content_size)
    axis_order = layout.value
    physical = torch.empty(
        tuple(logical_shape[axis] for axis in axis_order), dtype=torch.float16
    )
    logical = physical.permute(tuple(axis_order.index(axis) for axis in range(5)))
    return physical, list(logical.unbind(0))


@pytest.mark.parametrize(("cache_layout", "layout_cls"), _SHARED_LAYOUTS)
def test_tp_shared_layout_round_trip_across_tp_sizes(
    cache_layout: KVCacheLayout, layout_cls
):
    block_size = 16
    num_layers = 2
    stored: dict[int, bytes] = {}
    producer_layers = []

    for tp_rank in range(4):
        physical, layers = _physical_layer_views(
            cache_layout, num_layers, 1, 2, block_size, 4
        )
        for layer_index, layer in enumerate(layers):
            layer.copy_(
                torch.arange(layer.numel(), dtype=torch.float16).view_as(layer)
                + tp_rank * 1000
                + layer_index * 100
            )
        producer_layers.append(layers)
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=4,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        layout.register_kv_caches(layers, 1)
        addrs, sizes, _ = layout.prepare_values([(0, block_size)], [0], [tp_rank])
        stored[tp_rank] = b"".join(
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs[0], sizes[0], strict=True)
        )

    for tp_rank in range(2):
        physical, layers = _physical_layer_views(
            cache_layout, num_layers, 1, 4, block_size, 4
        )
        physical.zero_()
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=2,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        layout.register_kv_caches(layers, 1)
        shard_ids = layout.local_shard_ids
        addrs, sizes, _ = layout.prepare_values(
            [(0, block_size)] * len(shard_ids), [0], shard_ids
        )
        for shard_id, shard_addrs, shard_sizes in zip(
            shard_ids, addrs, sizes, strict=True
        ):
            offset = 0
            for addr, size in zip(shard_addrs, shard_sizes, strict=True):
                ctypes.memmove(addr, stored[shard_id][offset : offset + size], size)
                offset += size

        for layer_index, layer in enumerate(layers):
            expected = torch.cat(
                [
                    rank_layers[layer_index]
                    for rank_layers in producer_layers[tp_rank * 2 : tp_rank * 2 + 2]
                ],
                dim=1,
            )
            torch.testing.assert_close(layer, expected)
