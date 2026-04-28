# MooncakeStoreConnector — HMA Support Design

**Date:** 2026-04-28
**Repo:** `vllm-mooncake` (https://github.com/ivanium/vllm), branch `feat/mooncake-store-int`
**Reference:** vllm-svf PR #128 (HMA support for the P2P MooncakeConnector)
**Files in scope:**
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/{mooncake_store_connector,mooncake_store_scheduler,mooncake_store_data,mooncake_store_worker}.py`
plus tests under `tests/v1/kv_connector/unit/`.

---

## 1. Problem

The MooncakeStoreConnector (vllm upstream PR #40900) registers KV cache layers
with a Mooncake distributed store and addresses them with a single flat
`block_ids: list[int]`. This implicitly assumes one KV cache group:

- `mooncake_store_scheduler.py` lines 177-180 / 221-224 call
  `request.block_ids[0]` and discard everything else — an HMA model with
  `(FA_blocks, SWA_blocks)` would silently drop the SWA group.
- `mooncake_store_worker.py:870` carries an explicit
  `# TODO(yifan): we haven't supported HMA yet.` in `register_kv_caches`,
  which loops over every layer assuming one shared block-id space.
- `mooncake_store_data.py:80-97` `prepare_value` indexes
  `block_ids[start // block_size]` and applies the same block id to every
  registered (layer × K/V-segment) tuple; no notion of "this layer belongs
  to group g, use `block_ids[g]`."
- No sliding-window pruning, so SWA blocks outside the window would be saved
  as garbage and waste store keys.

DeepSeek (single full-attention MLA group) works today because the assumed
shape is `N=1`. HMA models (Gemma-3, Llama-4, hybrid Mistral) — which mix
`FullAttentionSpec` and `SlidingWindowSpec` layers — do not.

PR #128 already solved the equivalent problem for the P2P
`MooncakeConnector` by propagating `list[list[int]]` through the entire
pipeline and clipping SWA groups in the scheduler. This spec adapts that
design to the store connector.

## 2. Goals / Non-goals

**Goals**

- HMA correctness for save and load through the Mooncake store, parity with
  PR #128's contract on the P2P side.
- Unified data shape: block IDs are always `list[list[int]]`; single-group
  models simply have `N=1`. No branching on `_is_hma_required` at the data
  layer.
- DeepSeek / dense-FA models must keep working without behavior change.
- Tests at the same granularity as PR #128's `test_mooncake_connector_hma.py`.

**Non-goals**

- Per-layer cross-group block-id mapping at the store-key level. The store
  key (chunk_hash) is intentionally group-agnostic; only address resolution
  becomes per-group.
- Cross-instance HMA interop with a non-HMA-aware producer/consumer. If one
  side is on a stale connector, behavior is undefined; we'll document it but
  not gate at runtime.

## 3. Contract

```
Scheduler:
  request.block_ids: tuple[list[int], ...]                  (N groups)
    → MooncakeStoreScheduler.get_sw_clipped_blocks(...)     (clip SWA)
    → RequestTracker.allocated_block_ids: list[list[int]]
    → ReqMeta.block_ids: list[list[int]]

  ─── ZMQ / IPC ───

Worker:
  ReqMeta.block_ids: list[list[int]]
  register_kv_caches(kv_caches):
    bucket layer_names by kv_cache_config.kv_cache_groups
    → groups: list[GroupLayout]
    → layer_to_group: list[int]    (per registered (layer × seg) entry)
  prepare_value(start, end, block_ids_per_group):
    chunk_id  = start // block_size
    total     = max(len(g) for g in block_ids_per_group)   # FA-group length
    for each registered segment s:
       g       = layer_to_group[s]
       offset  = total - len(block_ids_per_group[g])       # 0 for FA, >0 for SWA
       local_i = chunk_id - offset                         # caller has already
                                                           # filtered chunks where
                                                           # local_i < 0 for any group
       block_id = block_ids_per_group[g][local_i]
       addr     = base_addrs[s] + block_id * block_lens[s]
```

Single-group models stay correct because `N=1`, `offset = 0`,
`layer_to_group = [0] * S`, and the loop reduces to today's behavior.

## 4. Components

### 4.1 `mooncake_store_connector.py`

- Constructor already accepts `kv_cache_config: KVCacheConfig | None`. Pass
  it to `MooncakeStoreScheduler` and `MooncakeStoreWorker`.
- `request_finished(self, request, block_ids)`:
  - Accept `block_ids: list[int] | tuple[list[int], ...]`. If flat list,
    wrap to `(block_ids,)`.
  - Forward to scheduler.

### 4.2 `mooncake_store_scheduler.py` (MooncakeStoreScheduler)

New attributes computed in `__init__` from `kv_cache_config`:

- `_is_hma_required: bool` — True iff
  `not disable_hybrid_kv_cache_manager` and any group is not a
  `FullAttentionSpec`.
- `blocks_per_sw: list[int]` — for each group: `cdiv(sliding_window,
  block_size) + 1` if `SlidingWindowSpec`, else `0`.

New method (mirrors #128):

```python
def get_sw_clipped_blocks(
    self, block_ids: tuple[list[int], ...] | list[list[int]]
) -> list[list[int]]:
    if not block_ids or not self._is_hma_required:
        return list(block_ids)
    return [
        blocks[-self.blocks_per_sw[i]:] if self.blocks_per_sw[i] > 0 else blocks
        for i, blocks in enumerate(block_ids)
    ]
```

Updates:

- `_unfinished_requests`, `_request_trackers` typed `list[list[int]]`.
- `update_state_after_alloc`: replace `blocks.get_block_ids()[0]` with
  `blocks.get_block_ids()`, then `get_sw_clipped_blocks(...)`.
- `build_connector_meta`: collapse the two `if not isinstance(request.block_ids[0], list)`
  adapter blocks (lines 177-180, 221-224) into a single
  normalize-to-tuple-then-clip helper.
- `request_finished`: `delay_free_blocks = any(len(g) > 0 for g in block_ids)`.

### 4.3 `mooncake_store_data.py`

`RequestTracker`:

- `allocated_block_ids: list[list[int]]`.
- `update(new_block_ids)`: accept the same tuple/list-of-lists shape, extend
  each group independently. Reject unknown shapes loudly.

`ReqMeta`:

- `block_ids: list[list[int]]`.

`ChunkedTokenDatabase`:

- New per-group layout state:
  ```python
  @dataclass
  class GroupLayout:
      base_addrs: list[int]   # per (layer × K/V-seg)
      block_lens: list[int]   # per (layer × K/V-seg)
  ```
- `groups: list[GroupLayout]`.
- `layer_to_group: list[int]` — index into `kv_caches_base_addr` (the
  flattened-across-groups list) to its group index.
- `set_groups(groups: list[GroupLayout])` replaces `set_kv_caches_base_addr`
  / `set_block_len` (which we keep as thin shims for now to keep the diff
  small).
- `prepare_value(start, end, block_ids_per_group: list[list[int]])`:
  - For each registered segment `s`, look up
    `g = layer_to_group[s]`, then
    `block_id = block_ids_per_group[g][start // block_size]`.
  - `addr = base_addrs[s] + block_id * block_lens[s]`.
  - Returns the same `(addr_list, size_list, last_block_id)` tuple shape so
    callers don't need to change.

### 4.4 `mooncake_store_worker.py` (MooncakeStoreWorker)

Constructor:

- Accept `kv_cache_config: KVCacheConfig`. Persist as `self.kv_cache_config`.

`register_kv_caches`:

- Drop the `# TODO(yifan): we haven't supported HMA yet.` comment.
- Bucket `kv_caches.items()` by `kv_cache_config.kv_cache_groups[g].layer_names`.
- Per-bucket, run the existing stride-based segment detection
  (page_size_bytes, outer_dims) to decide blocks-first vs K/V-first. Each
  group emits its own `(base_addrs, block_lens)`.
- The flat `kv_caches_base_addr` / `block_len` lists become group-0 +
  group-1 + … concatenation, so the existing `batch_put_from_multi_buffers`
  call site stays unchanged. We additionally record `layer_to_group` of
  matching length.
- Logging: include `num_groups` and per-group segment count.

`KVCacheStoreSendingThread._handle_request` and
`KVCacheStoreRecvingThread._handle_request`:

- Pass `req_meta.block_ids` (now `list[list[int]]`) to `prepare_value`.
- Group-count assertion against `len(self.token_database.groups)`. On
  mismatch: log error, drop the request, mark task done. Mirrors #128's
  `KV group count mismatch`.
- SWA save/load gating. After scheduler clipping, an SWA group's
  `block_ids[g]` holds only the last `blocks_per_sw[g]` entries — these
  correspond to the **trailing** chunks of the request, not the leading
  ones. We compute a per-group offset:
  ```python
  total_chunks = max(len(group) for group in block_ids)
  group_start_chunk[g] = total_chunks - len(block_ids[g])
  ```
  For a chunk at logical position `chunk_id = start // block_size`, group
  `g` is valid iff `chunk_id >= group_start_chunk[g]`, and its local
  block id is `block_ids[g][chunk_id - group_start_chunk[g]]`. A chunk is
  saveable / loadable only if it is valid for **every** group
  (intersection). Chunks outside any group's window are skipped on both
  paths.

## 5. Error handling

| Failure                                  | Behavior                                                        |
|------------------------------------------|-----------------------------------------------------------------|
| Group-count mismatch (req vs registered) | Log error, drop request, mark queue task done.                  |
| Empty group (fully clipped)              | Valid; chunks for that group are skipped, no crash.             |
| Flat `list[int]` from legacy caller      | Connector wraps to 1-tuple; downstream sees `N=1`.              |
| `kv_cache_config is None`                | Worker falls back to single-group (legacy) layout. Logged once. |

## 6. Testing

New file: `tests/v1/kv_connector/unit/test_mooncake_store_connector_hma.py`
mirroring `test_mooncake_connector_hma.py` from #128:

- `test_sw_sizes` — `blocks_per_sw` parametrized over swa_enabled.
- `test_is_hma_required` — derived from groups + scheduler config flag.
- `test_get_sw_clipped_blocks` — FA untouched, SWA clipped.
- `test_get_sw_clipped_blocks_noop_no_hma` — single-group no-op.
- `test_register_kv_caches_groups_by_kv_group` — for a mixed FA+SWA layer
  set, per-group base_addrs / block_lens correct, `layer_to_group` matches.
- `test_prepare_value_picks_right_group` — layer in SWA group resolves to
  SWA's block IDs, not FA's.
- `test_save_skips_chunks_outside_swa_window`.
- `test_load_skips_chunks_outside_swa_window`.
- `test_request_finished_with_hma_groups` — mirrors #128.

Update existing `test_mooncake_store_connector.py` and
`test_mooncake_store_worker.py` to use the unified `list[list[int]]` shape.
DeepSeek single-group cases stay valid (N=1 tuples).

Manual smoke:
- HMA model end-to-end save+load (Gemma-3 / Llama-4).
- DeepSeek-V3 regression on the unified path.

## 7. Rollout

One PR off `feat/mooncake-store-int`, ~6 atomic commits along the natural
seams below. Each commit compiles and passes the existing test suite; the
final commit lands the new HMA tests.

1. Plumb `kv_cache_config` to scheduler/worker; add `_is_hma_required`,
   `blocks_per_sw`, `get_sw_clipped_blocks` (no behavior change yet — the
   helper is unused).
2. Convert `RequestTracker` / `ReqMeta` / metadata structures to
   `list[list[int]]`. Update both branches of
   `MooncakeStoreScheduler.build_connector_meta`.
3. `ChunkedTokenDatabase` refactor: introduce `GroupLayout`,
   `layer_to_group`, group-aware `prepare_value`. Single-group still works
   because `N=1`.
4. `MooncakeStoreWorker.register_kv_caches`: bucket layers by
   `kv_cache_config.kv_cache_groups`. Drop the TODO comment.
5. Worker save/load threads: per-group block IDs, group-count assertion,
   SWA skip on chunks outside any group's window.
6. Tests: new HMA test file + updates to existing tests.

## 8. Open questions

(none — answered during brainstorming)

- HMA scope: full parity with #128 ✓
- Group identity source: worker re-derives from `kv_cache_config` ✓
- SW clipping location: scheduler-side, mirroring #128 ✓
- Compat: unified path, single-group is N=1 ✓
- Rollout: one PR, layered atomic commits ✓
