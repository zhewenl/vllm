# MooncakeStoreConnector HMA Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hybrid Memory Architecture (HMA) support to `MooncakeStoreConnector` so models with mixed FullAttention + SlidingWindow KV cache groups (Gemma-3, Llama-4) save and load correctly via the Mooncake distributed store, while keeping single-group models (DeepSeek/MLA, dense FA) fully working.

**Architecture:** Mirror PR #128's per-group propagation pattern — block IDs become `list[list[int]]` end-to-end, scheduler-side SWA clipping via `get_sw_clipped_blocks`, worker buckets layers by `kv_cache_config.kv_cache_groups`, `prepare_value` resolves each layer's block ID through a `layer_to_group` table. Single-group is the `N=1` case of the same path.

**Tech Stack:** Python, vLLM v1 KV connector framework, MooncakeDistributedStore client, pytest.

**Working repo:** `/home/zhewen/repos/vllm-mooncake`, branch `feat/mooncake-store-hma` (already created off `feat/mooncake-store-int`).

**Spec:** `docs/superpowers/specs/2026-04-28-mooncake-store-hma-design.md` (commit `8aeeb45d4`).

---

## Pre-flight

- [ ] **Confirm working tree is clean and on the right branch**

```bash
cd /home/zhewen/repos/vllm-mooncake && git status && git branch --show-current
```

Expected: clean, on `feat/mooncake-store-hma`. If untracked files exist (e.g. `Mooncake/`, `scripts/mooncake/Untitled`) leave them alone — they're not part of this work.

- [ ] **Identify the test runner**

```bash
cd /home/zhewen/repos/vllm-mooncake && ls .venv/bin/python 2>/dev/null && echo "using .venv" || echo "no .venv — use system python or pip install -e ."
```

All `pytest` invocations below assume `.venv/bin/python -m pytest ...`. Substitute as needed.

---

## Task 1: Plumb `kv_cache_config` + add HMA detection helpers in scheduler

**Goal:** Scheduler exposes `_is_hma_required`, `blocks_per_sw`, and `get_sw_clipped_blocks`. No behavior change yet — the helpers are tested but not used by the build path.

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py`
- Create: `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`

- [ ] **Step 1.1: Write the failing tests for `blocks_per_sw`, `_is_hma_required`, `get_sw_clipped_blocks`**

Create `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeStoreConnector HMA support."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_scheduler import (  # noqa: E501
    MooncakeStoreScheduler,
)

from .utils import create_vllm_config, make_kv_cache_config


def _make_scheduler(swa_enabled: bool, sw_size: int = 2048,
                    block_size: int = 16, disable_hma: bool = False):
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hma
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=swa_enabled, sw_size=sw_size,
    )
    return MooncakeStoreScheduler(vllm_config, kv_cache_config=kv_cache_config)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,expected_blocks_per_sw",
    [
        (True, [0, 2048 // 16 + 1]),  # FA=0, SWA=128+1=129
        (False, [0]),
    ],
)
def test_blocks_per_sw(swa_enabled, expected_blocks_per_sw):
    scheduler = _make_scheduler(swa_enabled=swa_enabled)
    assert scheduler.blocks_per_sw == expected_blocks_per_sw


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,disable_hma,expected_is_hma",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
    ],
)
def test_is_hma_required(swa_enabled, disable_hma, expected_is_hma):
    scheduler = _make_scheduler(swa_enabled=swa_enabled, disable_hma=disable_hma)
    assert scheduler._is_hma_required is expected_is_hma


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_clips_swa_keeps_fa():
    # sw_size=128 tokens / block_size=16 = 8 blocks + 1 = 9 blocks_per_sw
    scheduler = _make_scheduler(swa_enabled=True, sw_size=128)
    assert scheduler.blocks_per_sw == [0, 9]

    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))
    clipped = scheduler.get_sw_clipped_blocks((fa_blocks, sw_blocks))

    assert clipped[0] == fa_blocks            # FA untouched
    assert clipped[1] == sw_blocks[-9:]       # SWA clipped to last 9
    assert len(clipped[1]) == 9


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_noop_when_not_hma():
    scheduler = _make_scheduler(swa_enabled=False)
    assert scheduler._is_hma_required is False
    block_ids = ([1, 2, 3],)
    assert scheduler.get_sw_clipped_blocks(block_ids) == [[1, 2, 3]]
```

- [ ] **Step 1.2: Run the new tests — verify they fail**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py -v
```

Expected: FAIL — `MooncakeStoreScheduler.__init__()` got an unexpected keyword argument `kv_cache_config`, or `AttributeError: 'MooncakeStoreScheduler' object has no attribute 'blocks_per_sw'`.

- [ ] **Step 1.3: Modify `MooncakeStoreScheduler.__init__` to accept `kv_cache_config` and compute the helpers**

In `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py`:

Add imports near the top with the other vllm imports:

```python
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    SlidingWindowSpec,
)
```

Change the `__init__` signature and body (currently lines 38-63) to:

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig | None = None,
):
    self.kv_role = vllm_config.kv_transfer_config.kv_role
    self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
        "load_async", False
    )
    self.client = LookupKeyClient(vllm_config)

    self.load_specs: dict[str, LoadSpec] = {}
    self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    self.original_block_size = vllm_config.cache_config.block_size
    self._block_size = vllm_config.cache_config.block_size
    if self.pcp_size > 1:
        self._block_size *= self.pcp_size
    if self.dcp_size > 1:
        self._block_size *= self.dcp_size

    self._request_trackers: dict[str, RequestTracker] = {}
    self._preempted_req_ids: set[str] = set()
    self._discard_partial_chunks = (
        vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
    )
    self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
    self._unfinished_request_ids: set[str] = set()

    # HMA detection. blocks_per_sw[g] is non-zero only for SlidingWindowSpec
    # groups; FullAttentionSpec groups stay 0 and are never clipped.
    self._is_hma_required = False
    self.blocks_per_sw: list[int] = [0]
    if kv_cache_config is not None:
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self._block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

def get_sw_clipped_blocks(
    self,
    block_ids: tuple[list[int], ...] | list[list[int]],
) -> list[list[int]]:
    """Clip per-group block IDs to the SWA window. No-op for non-HMA."""
    if len(block_ids) == 0 or not self._is_hma_required:
        return list(block_ids)
    return [
        blocks[-self.blocks_per_sw[i]:] if self.blocks_per_sw[i] > 0 else blocks
        for i, blocks in enumerate(block_ids)
    ]
```

- [ ] **Step 1.4: Pass `kv_cache_config` from connector to scheduler**

In `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py` change line 105 (`MooncakeStoreScheduler(vllm_config)`) to:

```python
self.connector_scheduler = MooncakeStoreScheduler(
    vllm_config, kv_cache_config=kv_cache_config,
)
```

- [ ] **Step 1.5: Run the new tests — verify they pass**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py -v
```

Expected: 4 passed (parametrized cases unrolled to more).

- [ ] **Step 1.6: Run the existing store tests — make sure nothing regressed**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_connector.py tests/v1/kv_connector/unit/test_mooncake_store_worker.py -v
```

Expected: all green. The unused-arg path means existing call sites still work.

- [ ] **Step 1.7: Commit**

```bash
cd /home/zhewen/repos/vllm-mooncake && git add \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py
git commit -m "$(cat <<'EOF'
feat(mooncake-store): add HMA detection helpers to scheduler

Plumb kv_cache_config into MooncakeStoreScheduler and compute
_is_hma_required, blocks_per_sw, and get_sw_clipped_blocks. Helpers
are unused by the build path yet — wired in subsequent commits.

Mirrors the HMA detection design from vllm-svf PR #128 for the P2P
connector.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Convert `RequestTracker` / `ReqMeta` / scheduler block-id flow to `list[list[int]]`

**Goal:** Block IDs are propagated as per-group lists end-to-end on the scheduler side. Single-group case becomes `[[...]]`. SWA clipping is now applied.

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py`
- Modify: `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`

- [ ] **Step 2.1: Write the failing tests for per-group propagation in scheduler**

Append to `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`:

```python
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    RequestTracker,
)


@pytest.mark.cpu_test
def test_request_tracker_update_extends_each_group():
    """update() must extend the matching group, not flatten."""
    tracker = RequestTracker(
        req_id="r1", token_len=0, allocated_block_ids=[[1, 2], [10, 11]],
    )
    # New per-group blocks
    tracker.update(([3, 4], [12, 13]))
    assert tracker.allocated_block_ids == [[1, 2, 3, 4], [10, 11, 12, 13]]


@pytest.mark.cpu_test
def test_request_tracker_update_accepts_flat_list_as_single_group():
    """Legacy flat list extends group 0 (single-group compat)."""
    tracker = RequestTracker(
        req_id="r1", token_len=0, allocated_block_ids=[[1, 2]],
    )
    tracker.update([3, 4])
    assert tracker.allocated_block_ids == [[1, 2, 3, 4]]


@pytest.mark.cpu_test
def test_scheduler_request_finished_clips_swa_group():
    """request_finished must clip SWA group on the way to delay-free state."""
    scheduler = _make_scheduler(swa_enabled=True, sw_size=128, block_size=16)
    # blocks_per_sw = [0, 9]
    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))

    request = MagicMock()
    request.request_id = "r-finished"
    request.kv_transfer_params = {}
    # Pretend the request was producing blocks (tracker exists with saved tokens)
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
        RequestTracker,
    )
    scheduler._request_trackers["r-finished"] = RequestTracker(
        req_id="r-finished", token_len=20 * 16, allocated_block_ids=[fa_blocks, sw_blocks],
        num_saved_tokens=20 * 16,
    )

    delay, _ = scheduler.request_finished(request, (fa_blocks, sw_blocks))
    assert delay is True
```

- [ ] **Step 2.2: Run, verify fails**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py -v -k "tracker_update or request_finished_clips"
```

Expected: FAIL — `RequestTracker.update` doesn't yet handle per-group input; `request_finished` typing-mismatch on `list[int]`.

- [ ] **Step 2.3: Update `RequestTracker` and `ReqMeta` in `mooncake_store_data.py`**

Replace `RequestTracker` (currently lines 140-162) with:

```python
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

        if isinstance(new_block_ids, tuple) or (
            isinstance(new_block_ids, list)
            and len(new_block_ids) > 0
            and isinstance(new_block_ids[0], list)
        ):
            # Per-group shape
            assert len(new_block_ids) == len(self.allocated_block_ids), (
                f"KV group count mismatch on update: got {len(new_block_ids)} "
                f"groups, tracker has {len(self.allocated_block_ids)}"
            )
            for g, group_blocks in enumerate(new_block_ids):
                self.allocated_block_ids[g].extend(group_blocks)
        elif isinstance(new_block_ids, list):
            # Legacy flat-list: extend group 0
            self.allocated_block_ids[0].extend(new_block_ids)
        else:
            raise ValueError(
                f"Unsupported new_block_ids type {type(new_block_ids)}"
            )
```

Replace `ReqMeta.block_ids` typing in the dataclass (currently line 171):

```python
    block_ids: list[list[int]]
```

In `ReqMeta.from_request_tracker` (the `block_ids=tracker.allocated_block_ids` field at line 238) is already correct shape — no change needed.

- [ ] **Step 2.4: Update scheduler `update_state_after_alloc`, `build_connector_meta`, `request_finished` to use per-group shape with clipping**

In `mooncake_store_scheduler.py`:

Replace `update_state_after_alloc` (lines 110-143) with:

```python
def update_state_after_alloc(
    self,
    request: Request,
    blocks: KVCacheBlocks,
    num_external_tokens: int,
):
    """Update state after block allocation."""
    if num_external_tokens > 0:
        # Always per-group (tuple) → list[list[int]]; clip SWA groups.
        local_block_ids = self.get_sw_clipped_blocks(blocks.get_block_ids())
    else:
        # Initialize as one empty group per detected KV cache group, so
        # subsequent .update() calls have correct shape.
        local_block_ids = [[] for _ in range(len(self.blocks_per_sw))]

    self._unfinished_requests[request.request_id] = (request, local_block_ids)
    self._unfinished_request_ids.add(request.request_id)

    if request.request_id not in self.load_specs:
        return

    if num_external_tokens == 0:
        self.load_specs[request.request_id].can_load = False
        return

    assert (
        num_external_tokens > 0
        and num_external_tokens
        == self.load_specs[request.request_id].kvpool_cached_tokens
        - self.load_specs[request.request_id].vllm_cached_tokens
    ), (
        f"Mismatch in number of tokens: {num_external_tokens} vs "
        f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
        f"{self.load_specs[request.request_id].vllm_cached_tokens}"
        f" for request {request.request_id}"
    )

    self.load_specs[request.request_id].can_load = True
```

Add a helper at the top of `MooncakeStoreScheduler` (above `update_state_after_alloc`):

```python
def _normalize_block_ids(
    self, block_ids: tuple[list[int], ...] | list[int] | list[list[int]],
) -> list[list[int]]:
    """Return a per-group list[list[int]] regardless of input shape."""
    if isinstance(block_ids, tuple):
        return self.get_sw_clipped_blocks(block_ids)
    if (
        isinstance(block_ids, list)
        and len(block_ids) > 0
        and isinstance(block_ids[0], list)
    ):
        return self.get_sw_clipped_blocks(block_ids)
    # Flat list[int] → wrap as single group, no clipping
    return [list(block_ids)] if block_ids else [[]]
```

In `build_connector_meta`, replace **both** adapter blocks
(lines 177-180 and 221-224) — `if not isinstance(request.block_ids[0], list): unfolded_block_ids = request.block_ids.copy() else: unfolded_block_ids = request.block_ids[0].copy()` — with:

```python
unfolded_block_ids = self._normalize_block_ids(request.block_ids)
```

In `request_finished` (lines 346-364), replace the body with:

```python
def request_finished(
    self,
    request: Request,
    block_ids: tuple[list[int], ...] | list[int] | list[list[int]],
) -> tuple[bool, dict[str, Any] | None]:
    """Determine whether to delay freeing blocks for async save."""
    if self.kv_role == "kv_consumer":
        return False, None
    tracker = self._request_trackers.get(request.request_id)
    if tracker is not None and tracker.num_saved_tokens <= 0:
        return False, None
    normalized = self._normalize_block_ids(block_ids)
    delay_free_blocks = any(len(group) > 0 for group in normalized)
    if delay_free_blocks:
        logger.debug(
            "Delaying free of %s blocks for request %s",
            [len(g) for g in normalized], request.request_id,
        )
    return delay_free_blocks, None
```

In `mooncake_store_connector.py` line 146 (`request_finished`), widen the type hint:

```python
def request_finished(
    self,
    request: Request,
    block_ids: tuple[list[int], ...] | list[int] | list[list[int]],
) -> tuple[bool, dict[str, Any] | None]:
    assert self.connector_scheduler is not None
    return self.connector_scheduler.request_finished(request, block_ids)
```

- [ ] **Step 2.5: Run hma tests + existing store tests, verify all pass**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py -v
```

Expected: all green. If a previously-existing test asserts the legacy single-group `list[int]` shape on a tracker, fix that test to expect `[[...]]`.

- [ ] **Step 2.6: Commit**

```bash
cd /home/zhewen/repos/vllm-mooncake && git add \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py
git commit -m "$(cat <<'EOF'
feat(mooncake-store): per-group block_ids in scheduler/data classes

RequestTracker.allocated_block_ids and ReqMeta.block_ids become
list[list[int]] (one per KV cache group). Scheduler clips SWA groups
via get_sw_clipped_blocks before storing. Single-group models still
work — they just see [[...]].

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `ChunkedTokenDatabase` — `GroupLayout` and group-aware `prepare_value`

**Goal:** Worker-side address resolution can pick the right per-group block ID for each registered layer/segment. Single-group still resolves the same way as today.

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py`
- Modify: `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`

- [ ] **Step 3.1: Write failing test for `prepare_value` per-group resolution**

Append to `test_mooncake_store_connector_hma.py`:

```python
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    GroupLayout,
    KeyMetadata,
)


@pytest.mark.cpu_test
def test_prepare_value_picks_right_group_block_ids():
    """A layer in the SWA group must address through SWA's clipped block ids."""
    block_size = 16
    metadata = KeyMetadata(
        model_name="m", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0,
    )
    db = ChunkedTokenDatabase(metadata, block_size=block_size)

    # Two groups, each with one layer (one segment per layer for simplicity).
    fa_layout = GroupLayout(base_addrs=[0x1000], block_lens=[256])
    sw_layout = GroupLayout(base_addrs=[0x2000], block_lens=[256])
    db.set_groups([fa_layout, sw_layout], layer_to_group=[0, 1])

    # 20 chunks total. SWA holds last 9.
    fa_block_ids = list(range(20))
    sw_block_ids = list(range(100, 109))  # only last 9
    block_ids_per_group = [fa_block_ids, sw_block_ids]

    # Chunk 19: the LAST chunk → both groups must resolve.
    # SWA's chunk 19 maps to its index (19 - (20 - 9)) = 8 → sw_block_ids[8] = 108.
    addr_list, size_list, _ = db.prepare_value(
        start=19 * block_size, end=20 * block_size,
        block_ids=block_ids_per_group,
    )
    # Two segments registered (one per layer), so two entries.
    assert len(addr_list) == 2
    assert addr_list[0] == 0x1000 + 19 * 256        # FA: block 19
    assert addr_list[1] == 0x2000 + 108 * 256       # SWA: block 108


@pytest.mark.cpu_test
def test_prepare_value_single_group_unchanged():
    """N=1 (single full-attention group) must produce the same addresses as before."""
    block_size = 16
    metadata = KeyMetadata(
        model_name="m", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0,
    )
    db = ChunkedTokenDatabase(metadata, block_size=block_size)

    layout = GroupLayout(base_addrs=[0x1000, 0x2000], block_lens=[256, 256])
    db.set_groups([layout], layer_to_group=[0, 0])

    block_ids_per_group = [[5, 6, 7, 8]]
    addr_list, size_list, _ = db.prepare_value(
        start=2 * block_size, end=3 * block_size,
        block_ids=block_ids_per_group,
    )
    assert addr_list == [0x1000 + 7 * 256, 0x2000 + 7 * 256]
```

- [ ] **Step 3.2: Run, verify fails**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py::test_prepare_value_picks_right_group_block_ids -v
```

Expected: ImportError on `GroupLayout` or AttributeError on `set_groups`.

- [ ] **Step 3.3: Implement `GroupLayout`, `layer_to_group`, `set_groups`, group-aware `prepare_value`**

In `mooncake_store_data.py`:

Add the `GroupLayout` dataclass near the top (after the `PoolKey` class):

```python
@dataclass
class GroupLayout:
    """Per-KV-cache-group memory layout.

    base_addrs and block_lens are flattened across the group's layers and
    K/V segments; entry i is one (layer, K|V) pair.
    """
    base_addrs: list[int]
    block_lens: list[int]
```

Replace `ChunkedTokenDatabase` (lines 62-127) with:

```python
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
        # kv_caches_base_addr / block_len; entry i tells which group
        # registered segment i belongs to.
        self.groups: list[GroupLayout] = []
        self.layer_to_group: list[int] = []

    def _make_key_by_hash(self, chunk_hash: str) -> "PoolKey":
        return PoolKey(self.metadata, chunk_hash)

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        """Legacy setter — single-group flat layout."""
        self.kv_caches_base_addr = kv_caches_base_addr
        # If callers use the legacy setter, default to one group.
        if not self.groups or len(self.groups) != 1:
            self.groups = [GroupLayout(base_addrs=kv_caches_base_addr,
                                       block_lens=self.block_len)]
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
        """Group-aware setter — flattens groups into kv_caches_base_addr/block_len.

        This is the canonical API used by the HMA-aware register_kv_caches.
        """
        self.groups = groups
        self.layer_to_group = layer_to_group
        # Flatten for backwards compatibility with consumers that walk the
        # flat lists directly (e.g. logging in register_kv_caches).
        flat_addrs: list[int] = []
        flat_lens: list[int] = []
        for layout in groups:
            flat_addrs.extend(layout.base_addrs)
            flat_lens.extend(layout.block_lens)
        self.kv_caches_base_addr = flat_addrs
        self.block_len = flat_lens

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
        if block_ids and not isinstance(block_ids[0], list):
            # Legacy flat shape — wrap as single group.
            block_ids = [block_ids]  # type: ignore[list-item]

        chunk_id = start // self.block_size
        # FA group is the longest; offset trims older chunks for SWA groups.
        total_chunks = max(len(g) for g in block_ids) if block_ids else 0
        # Per-group local-index for chunk_id; -1 means out of window.
        per_group_local: list[int] = []
        for group in block_ids:
            offset = total_chunks - len(group)
            local_i = chunk_id - offset
            per_group_local.append(local_i if 0 <= local_i < len(group) else -1)

        addr_list: list[int] = []
        size_list: list[int] = []
        last_block_id = 0
        # Walk every registered segment. Use layer_to_group to pick the right
        # per-group block id. If the chunk is outside this group's window, the
        # caller should have filtered it — we still produce a placeholder so
        # the lists stay aligned with starts/ends; the worker will skip the
        # whole chunk via is_chunk_savable() before reaching here.
        for seg_idx, base_addr in enumerate(self.kv_caches_base_addr):
            g = self.layer_to_group[seg_idx] if self.layer_to_group else 0
            local_i = per_group_local[g]
            if local_i < 0:
                # Defensive: caller should have filtered.
                addr_list.append(0)
                size_list.append(0)
                continue
            block_id = block_ids[g][local_i]
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
        if block_ids and not isinstance(block_ids[0], list):
            block_ids = [block_ids]  # type: ignore[list-item]
        if not block_ids:
            return False
        chunk_id = start // self.block_size
        total_chunks = max(len(g) for g in block_ids)
        for group in block_ids:
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
    ) -> Iterable[tuple[int, int, "PoolKey"]]:
        """Process tokens and yield (start_idx, end_idx, pool_key) tuples."""
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
```

- [ ] **Step 3.4: Run new tests, verify pass**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py -v
```

Expected: all green. The legacy `set_kv_caches_base_addr` / `set_block_len` setters still keep existing tests working because they auto-populate a one-element `groups` list.

- [ ] **Step 3.5: Commit**

```bash
cd /home/zhewen/repos/vllm-mooncake && git add \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py
git commit -m "$(cat <<'EOF'
feat(mooncake-store): group-aware ChunkedTokenDatabase.prepare_value

Add GroupLayout, layer_to_group, set_groups, and is_chunk_savable.
prepare_value now resolves each registered (layer × K/V-segment)
through its KV cache group's block ids, applying the per-group
chunk-offset for SWA groups. Legacy flat-list callers and single-group
models keep their behavior via shape-detection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Worker `register_kv_caches` — bucket layers by KV cache group

**Goal:** Worker reads `kv_cache_config.kv_cache_groups`, buckets registered layers per group, builds per-group `GroupLayout`, populates `layer_to_group`. Single-group model still gets one group.

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py`
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py`
- Modify: `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`

- [ ] **Step 4.1: Write failing test for `register_kv_caches` HMA layout**

Append to `test_mooncake_store_connector_hma.py`:

```python
from unittest.mock import patch
import torch


@pytest.mark.cpu_test
def test_register_kv_caches_buckets_by_kv_cache_group():
    """Mixed FA + SWA layers must be split into two GroupLayouts."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
        mooncake_store_worker,
    )

    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=True,
    )
    # FA group claims layer0 + layer2; SWA group claims layer1 + layer3.

    # Bypass __init__ to avoid the real MooncakeDistributedStore.
    worker = mooncake_store_worker.MooncakeStoreWorker.__new__(
        mooncake_store_worker.MooncakeStoreWorker,
    )
    worker.kv_cache_config = kv_cache_config
    worker.cache_config = vllm_config.cache_config
    worker.cache_config.num_gpu_blocks = 4
    worker.num_blocks = 4
    worker.block_size = block_size
    worker.use_mla = False
    worker.tp_size = 1
    worker.tp_rank = 0
    worker.put_step = 1
    worker.kv_role = "kv_both"
    worker.metadata = mooncake_store_worker.KeyMetadata(
        model_name="m", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0,
    )
    worker.token_database = mooncake_store_worker.ChunkedTokenDatabase(
        worker.metadata, block_size=block_size,
    )
    # Stub the store to avoid real RDMA register.
    worker.store = MagicMock()
    worker.store.register_buffer = MagicMock(return_value=0)
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.enable_kv_events = False
    worker.disk_offload_buffer_budget_bytes = None
    worker.replicate_config = None
    worker.preferred_segment = None
    worker.store_replicate_config = None

    # MLA-style blocks-first layout: (num_blocks, block_size, head_size).
    fa_tensor = torch.zeros((4, 16, 64), dtype=torch.float16)
    sw_tensor = torch.zeros((4, 16, 64), dtype=torch.float16)
    kv_caches = {
        "layer0": fa_tensor, "layer2": fa_tensor.clone(),  # FA group
        "layer1": sw_tensor, "layer3": sw_tensor.clone(),  # SWA group
    }
    # Avoid touching the real send/recv threads.
    with patch.object(
        mooncake_store_worker, "KVCacheStoreSendingThread"
    ), patch.object(
        mooncake_store_worker, "KVCacheStoreRecvingThread"
    ):
        worker.register_kv_caches(kv_caches)

    # Two groups, in declaration order from kv_cache_config.kv_cache_groups.
    db = worker.token_database
    assert len(db.groups) == 2
    assert len(db.groups[0].base_addrs) == 2  # FA: 2 layers
    assert len(db.groups[1].base_addrs) == 2  # SWA: 2 layers
    # Flat layer_to_group has 4 entries (2 per group, one segment per layer
    # because MLA-style blocks-first → no K/V split).
    assert worker.token_database.layer_to_group == [0, 0, 1, 1]
```

- [ ] **Step 4.2: Run, verify fails**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py::test_register_kv_caches_buckets_by_kv_cache_group -v
```

Expected: FAIL — `worker.kv_cache_config` not used; current `register_kv_caches` produces one group.

- [ ] **Step 4.3: Update `MooncakeStoreWorker.__init__` to accept `kv_cache_config`, then update `register_kv_caches`**

In `mooncake_store_worker.py`:

Change the `MooncakeStoreWorker.__init__` signature (line 753ish — find `def __init__(self, vllm_config: VllmConfig)`) to:

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig | None = None,
):
```

Add at the top of the file (with other imports):

```python
from vllm.v1.kv_cache_interface import KVCacheConfig
```

Save the new arg into `self`:

```python
self.kv_cache_config = kv_cache_config
```

(Add this line right after `self.vllm_config = vllm_config` if it exists, or near the top of the body — wherever fits the existing style.)

Replace `register_kv_caches` (currently lines 868-947) with:

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register KV cache tensors and start transfer threads.

    Buckets layers by kv_cache_config.kv_cache_groups so HMA models
    (mixed FullAttention + SlidingWindow) get one GroupLayout per group
    and prepare_value can resolve per-group block ids.
    """
    first_kv_cache = next(iter(kv_caches.values()))

    # num_blocks from cache_config is authoritative (set after profiling,
    # before KV cache allocation).
    assert self.cache_config.num_gpu_blocks is not None
    self.num_blocks = self.cache_config.num_gpu_blocks

    # Detect the KV cache memory layout using the stride-based approach.
    # Same logic as before, but applied per group below.
    storage = first_kv_cache.untyped_storage()
    el = first_kv_cache.element_size()
    page_size_bytes = storage.nbytes() // self.num_blocks
    outer_dims = [
        d
        for d in range(first_kv_cache.ndim)
        if first_kv_cache.stride(d) * el > page_size_bytes
    ]

    # Bucket layers by kv_cache_group. If kv_cache_config is missing
    # (legacy callers / tests), fall back to a single group containing
    # everything.
    if self.kv_cache_config is not None:
        layer_to_group_idx: dict[str, int] = {}
        for g_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                layer_to_group_idx[layer_name] = g_idx
        num_groups = len(self.kv_cache_config.kv_cache_groups)
    else:
        layer_to_group_idx = {name: 0 for name in kv_caches}
        num_groups = 1

    # Group caches by their KV cache group, preserving registration order.
    grouped_caches: list[list[tuple[str, torch.Tensor]]] = [
        [] for _ in range(num_groups)
    ]
    for name, cache in kv_caches.items():
        g_idx = layer_to_group_idx.get(name, 0)
        grouped_caches[g_idx].append((name, cache))

    seen_ptrs: set[int] = set()
    groups: list[GroupLayout] = []
    flat_layer_to_group: list[int] = []

    for g_idx, layers in enumerate(grouped_caches):
        group_base_addrs: list[int] = []
        group_block_lens: list[int] = []
        for _, cache in layers:
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            region_len = cache_storage.nbytes()

            if base_addr not in seen_ptrs:
                seen_ptrs.add(base_addr)
                ret = self.store.register_buffer(base_addr, region_len)
                if ret != 0:
                    logger.error(
                        "register_buffer failed for addr %#x len %d: %d",
                        base_addr, region_len, ret,
                    )

            if not outer_dims:
                # Blocks-first layout (FlashInfer / MLA): one segment.
                group_base_addrs.append(base_addr)
                group_block_lens.append(page_size_bytes)
                flat_layer_to_group.append(g_idx)
            else:
                # K/V-first layout (FlashAttn / ROCm): split segments.
                seg_stride = cache.stride(outer_dims[0]) * el
                for seg in range(cache.shape[outer_dims[0]]):
                    group_base_addrs.append(base_addr + seg * seg_stride)
                    group_block_lens.append(seg_stride // self.num_blocks)
                    flat_layer_to_group.append(g_idx)
        groups.append(GroupLayout(
            base_addrs=group_base_addrs, block_lens=group_block_lens,
        ))

    # Push into the token_database via the new group-aware API. This also
    # populates the legacy flat lists (kv_caches_base_addr / block_len).
    self.token_database.set_groups(groups, flat_layer_to_group)
    self.kv_caches_base_addr = self.token_database.kv_caches_base_addr
    self.block_len = self.token_database.block_len

    logger.info(
        "Registering KV_Caches. use_mla: %s, num_groups: %d, "
        "shape %s, num_blocks: %d, block_len: %s, "
        "per_key_bytes: %d, num_segments: %d",
        self.use_mla, num_groups, first_kv_cache.shape, self.num_blocks,
        list(set(self.block_len)), sum(self.block_len),
        len(self.kv_caches_base_addr),
    )

    # Start transfer threads (existing code below this point — leave
    # KVCacheStoreSendingThread / KVCacheStoreRecvingThread setup unchanged).
```

Add the import for `GroupLayout` at the top of the worker file (alongside the other `mooncake_store_data` imports):

```python
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_store_data import (
    ChunkedTokenDatabase,
    GroupLayout,            # <-- new
    KeyMetadata,
    MooncakeStoreConnectorMetadata,
    ReqMeta,
)
```

In `mooncake_store_connector.py` change line 107 (`MooncakeStoreWorker(vllm_config)`) to:

```python
self.connector_worker = MooncakeStoreWorker(
    vllm_config, kv_cache_config=kv_cache_config,
)
```

- [ ] **Step 4.4: Run the new test, verify pass**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py::test_register_kv_caches_buckets_by_kv_cache_group -v \
  tests/v1/kv_connector/unit/test_mooncake_store_worker.py -v
```

Expected: green. The existing worker tests still pass because the legacy single-group fallback runs when `kv_cache_config is None`.

- [ ] **Step 4.5: Commit**

```bash
cd /home/zhewen/repos/vllm-mooncake && git add \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py
git commit -m "$(cat <<'EOF'
feat(mooncake-store): bucket layers by KV cache group in register_kv_caches

Worker now consults kv_cache_config.kv_cache_groups to bucket each
registered layer into a GroupLayout, populates token_database via the
new set_groups API. layer_to_group lets prepare_value resolve the
right per-group block id for HMA models. Drops the
"# TODO(yifan): we haven't supported HMA yet." comment.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Worker save/load threads — per-group block_ids, group-count assertion, SWA chunk gating

**Goal:** Both `KVCacheStoreSendingThread._handle_request` and `KVCacheStoreRecvingThread._handle_request` pass per-group block_ids to `prepare_value` and skip chunks outside any group's window. Group-count mismatch triggers a logged drop, never a crash.

**Files:**
- Modify: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py`
- Modify: `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`

- [ ] **Step 5.1: Write failing tests for chunk gating + group-count assertion**

Append to `test_mooncake_store_connector_hma.py`:

```python
@pytest.mark.cpu_test
def test_is_chunk_savable_intersection_of_groups():
    """A chunk is savable iff it lies within every group's window."""
    block_size = 16
    metadata = KeyMetadata(
        model_name="m", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0,
    )
    db = ChunkedTokenDatabase(metadata, block_size=block_size)
    db.set_groups(
        [
            GroupLayout(base_addrs=[0x1000], block_lens=[256]),  # FA
            GroupLayout(base_addrs=[0x2000], block_lens=[256]),  # SWA
        ],
        layer_to_group=[0, 1],
    )

    fa = list(range(20))
    sw = list(range(100, 109))  # last 9 → group_start_chunk = 11
    block_ids = [fa, sw]

    # Chunk 5: only FA has it. Not savable.
    assert db.is_chunk_savable(start=5 * block_size, block_ids=block_ids) is False
    # Chunk 11: SWA's first valid chunk → savable.
    assert db.is_chunk_savable(start=11 * block_size, block_ids=block_ids) is True
    # Chunk 19: last → savable.
    assert db.is_chunk_savable(start=19 * block_size, block_ids=block_ids) is True
    # Chunk 20: past end → not savable.
    assert db.is_chunk_savable(start=20 * block_size, block_ids=block_ids) is False
```

- [ ] **Step 5.2: Run, verify the chunk-savable test passes (already implemented in Task 3)**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py::test_is_chunk_savable_intersection_of_groups -v
```

Expected: PASS — `is_chunk_savable` was added in Task 3.

- [ ] **Step 5.3: Update `KVCacheStoreSendingThread._handle_request` to skip non-savable chunks and assert group count**

In `mooncake_store_worker.py`, find `KVCacheStoreSendingThread._handle_request` (starts around line 337). Replace the chunk-collection loop (lines ~356-372) and the `prepare_value` call site (lines ~406-411) with the group-aware version.

Specifically — change this section:

```python
        starts = []
        ends = []
        keys = []
        block_hashes: list[BlockHash] = []
        for index, (start, end, key) in enumerate(
            self.token_database.process_tokens(token_len, req_meta.block_hashes)
        ):
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])
```

to:

```python
        # HMA: assert per-group block_ids has the right shape.
        num_groups = len(self.token_database.groups) or 1
        if len(block_ids) != num_groups:
            logger.error(
                "req %s: KV group count mismatch: got %d, expected %d. "
                "Dropping save.",
                req_id, len(block_ids), num_groups,
            )
            self.dec_stored_request(req_id)
            self.request_queue.task_done()
            return

        starts = []
        ends = []
        keys = []
        block_hashes: list[BlockHash] = []
        for index, (start, end, key) in enumerate(
            self.token_database.process_tokens(token_len, req_meta.block_hashes)
        ):
            # Skip chunks outside any group's SWA window.
            if not self.token_database.is_chunk_savable(start, block_ids):
                continue
            starts.append(start)
            ends.append(end)
            keys.append(key.to_string())
            block_hashes.append(req_meta.block_hashes[index])
```

The existing `prepare_value(start, ends[index], block_ids)` call at line ~407 already passes `block_ids`, which is now `list[list[int]]` thanks to Task 2. No change needed there — `prepare_value` handles per-group resolution internally.

- [ ] **Step 5.4: Update `KVCacheStoreRecvingThread._handle_request` symmetrically**

Find `KVCacheStoreRecvingThread._handle_request` (starts around line 507). Replace the loop (lines ~519-525) with:

```python
        num_groups = len(self.token_database.groups) or 1
        if len(req_meta.block_ids) != num_groups:
            logger.error(
                "req %s: KV group count mismatch on load: got %d, expected %d. "
                "Dropping load.",
                req_id, len(req_meta.block_ids), num_groups,
            )
            return

        addr_list = []
        size_list = []
        key_list = []
        for start, end, key in self.token_database.process_tokens(
            token_len, req_meta.block_hashes, mask_num
        ):
            if not self.token_database.is_chunk_savable(start, req_meta.block_ids):
                continue
            addr, size, _ = self.token_database.prepare_value(
                start, end, req_meta.block_ids,
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
```

(Replace the existing `for start, end, key in self.token_database.process_tokens(...)` loop body but keep the rest of the method — TP rotation, batch get, error handling — unchanged.)

- [ ] **Step 5.5: Run all unit tests, verify pass**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/unit/ -v -k "mooncake"
```

Expected: all green. If a worker test mocks `process_tokens` directly and asserts every chunk is processed, update it to use a savable single-group block_ids tuple.

- [ ] **Step 5.6: Commit**

```bash
cd /home/zhewen/repos/vllm-mooncake && git add \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py
git commit -m "$(cat <<'EOF'
feat(mooncake-store): per-group block_ids in send/recv threads

Both transfer threads now skip chunks outside any KV group's SWA
window via ChunkedTokenDatabase.is_chunk_savable, and assert
per-group block_ids matches the number of registered groups before
prepare_value runs. Group-count mismatch logs and drops the request,
never crashes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Lint, smoke-check, and final commit

**Goal:** Run linters as configured by AGENTS.md, run the full unit-test suite for KV connectors, document smoke-test commands for HMA + DeepSeek end-to-end runs.

**Files:**
- Modify (if lint complains): any of the files touched in tasks 1-5.
- Modify: `docs/superpowers/specs/2026-04-28-mooncake-store-hma-design.md` (optional — record actual smoke-run results).

- [ ] **Step 6.1: Run pre-commit on all changed files**

```bash
cd /home/zhewen/repos/vllm-mooncake && pre-commit run --files \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py \
  tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py
```

Expected: all hooks pass. If ruff or formatter rewrites a file, re-stage and commit as `style(mooncake-store): ...`.

- [ ] **Step 6.2: Run mypy as in CI**

```bash
cd /home/zhewen/repos/vllm-mooncake && pre-commit run mypy-3.10 --files \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_scheduler.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_data.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_worker.py \
  vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_store_connector.py \
  --hook-stage manual
```

Expected: clean. Common issues: missing `kv_cache_config` parameter type hints, narrow return types when accepting unions. Fix inline.

- [ ] **Step 6.3: Run the full KV connector unit-test suite**

```bash
cd /home/zhewen/repos/vllm-mooncake && .venv/bin/python -m pytest tests/v1/kv_connector/ -v
```

Expected: all green. Investigate any failures; do NOT commit on red.

- [ ] **Step 6.4: Smoke-test commands (manual — record results in PR description)**

For HMA (Gemma-3 or Llama-4):
```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
.venv/bin/python -m vllm serve google/gemma-3-12b-it \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
# Issue a long prompt, verify save/load logs report num_groups=2 and
# successful key counts on prefix-cache hits.
```

For DeepSeek (regression):
```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
.venv/bin/python -m vllm serve deepseek-ai/DeepSeek-V3 \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
# Single full-attention group → num_groups=1, behavior unchanged.
```

- [ ] **Step 6.5: Final lint/style commit if anything changed**

```bash
cd /home/zhewen/repos/vllm-mooncake && git status
# If pre-commit modified files in 6.1/6.2:
git add -A
git commit -m "style(mooncake-store): apply pre-commit fixes"
# Otherwise: skip this step.
```

- [ ] **Step 6.6: Push the branch and open PR (only if user explicitly approves)**

```bash
cd /home/zhewen/repos/vllm-mooncake && git log --oneline feat/mooncake-store-int..HEAD
# Verify ~6 atomic commits. Pause and confirm with user before pushing.
```

Do NOT push without user confirmation. PR description must follow AGENTS.md (duplicate-work checks, AI-assisted disclosure, test-plan with results).

---

## Self-review (filled in by author)

- **Spec coverage:** Each spec section maps to at least one task — §3 contract → Tasks 2 + 3 + 5; §4.1 connector → Task 1, 4; §4.2 scheduler → Tasks 1, 2; §4.3 data → Tasks 2, 3; §4.4 worker → Tasks 4, 5; §5 error handling → group-count drop in Task 5; §6 testing → distributed across each task.
- **Placeholder scan:** No "TBD", "TODO", "implement later", "fill in details" in any step.
- **Type consistency:** `RequestTracker.allocated_block_ids: list[list[int]]` introduced in Task 2 is consumed identically in Tasks 3, 4, 5. `GroupLayout` defined in Task 3 is imported and used in Task 4. `set_groups` signature matches between definition and call. `is_chunk_savable` defined in Task 3 is called in Task 5.
- **Backwards-compat path:** legacy `set_kv_caches_base_addr` / `set_block_len` setters keep one-group behavior; `prepare_value` accepts flat `list[int]` by wrapping; `_normalize_block_ids` handles tuple/list-of-lists/flat-list at the scheduler boundary.
