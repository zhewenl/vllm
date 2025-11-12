# ROCm test_prefix_prefill.py - Final Fix Summary

## Issues and Fixes

### 1. Type Mismatch: torch.long → torch.int ✅
**Lines**: 221-231
**Problem**: ROCm kernel expects `int32` but test used `torch.long` (int64)
**Fix**: Changed all tensor dtypes to `torch.int`:
```python
values = torch.arange(0, cache_size, dtype=torch.int)
block_table = values[...].view(BS, max_block_per_request)
b_seq_len = torch.tensor(seq_lens, dtype=torch.int)
b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int)
b_start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.int), dim=0)
b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1], dtype=torch.int), dim=0)
```

### 2. Unsupported fp8_e5m2 Format ✅
**Lines**: 170-175
**Problem**: ROCm custom kernel only supports fp8_e4m3, not fp8_e5m2
**Fix**: Added skip condition:
```python
if (
    current_platform.is_rocm()
    and op is chunked_prefill_paged_decode
    and kv_cache_dtype == "fp8_e5m2"
):
    pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")
```

### 3. Cache Population Bug (Root Cause) ✅
**Lines**: 236-246 and 484-494 (both test functions)
**Problem**: Test didn't correctly distinguish between prefill and decode cache requirements

**Root Cause Analysis**:

`chunked_prefill_paged_decode` handles TWO kernel paths:

#### Prefill Path (query_len > 1)
```python
context_attention_fwd(
    k=key,           # NEW tokens passed here
    v=value,         # NEW tokens passed here
    k_cache=key_cache,  # Only CONTEXT tokens in cache
    v_cache=value_cache,
)
```
- Cache should contain ONLY `ctx_len` context tokens
- New query tokens passed separately via `k`/`v` parameters
- Kernel combines both sources

#### Decode Path (query_len == 1)
```python
kernel_paged_attention(
    key_cache_ptr=key_cache,    # ONLY source of K/V
    value_cache_ptr=value_cache,
    # No separate k/v parameters!
)
```
- Cache must contain ALL `seq_len` tokens (context + current)
- No separate `k`/`v` parameters - everything from cache
- Kernel filters to process only `query_len==1` sequences (line 73-78)

**The Fix**:
```python
# For decode sequences ONLY (query_len==1), populate all seq_len tokens
# For prefill sequences, populate only ctx_len tokens
cache_len = (
    seq_lens[i]
    if (op is chunked_prefill_paged_decode and query_lens[i] == 1)
    else b_ctx_len[i]
)
while cur_ctx < cache_len:
    # ... copy to cache ...
```

**Why Previous Fix Failed**:
First attempt set `cache_len = seq_lens[i]` for ALL sequences when using `chunked_prefill_paged_decode`. This incorrectly added query tokens to cache for prefill sequences, when those tokens should only be passed via `k`/`v` parameters.

## Test Configuration That Failed

Only this specific combination triggered ROCm custom kernel and exposed the bug:
- `op = chunked_prefill_paged_decode`
- `num_queries_per_kv = 1` (GQA ratio = 64)
- `sliding_window = 0`
- `head_size = 128`
- `kv_cache_dtype = auto or fp8_e4m3`

Other configurations used Triton kernel with different semantics.

## Verification Command

```bash
HF_HUB_DISABLE_XET=1 wp pytest -s -v 'tests/kernels/attention/test_prefix_prefill.py'
```

Expected result: All 4 previously failing tests should now pass:
- `test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:0-auto-dtype0-128-1-64]`
- `test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:0-fp8-dtype0-128-1-64]`
- `test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:1-auto-dtype0-128-1-64]`
- `test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:1-fp8-dtype0-128-1-64]`

## Files Modified

1. `tests/kernels/attention/test_prefix_prefill.py`:
   - Lines 170-175: Added fp8_e5m2 skip
   - Lines 221-231: Changed dtype to torch.int (9 occurrences)
   - Lines 236-246: Fixed cache population for decode sequences
   - Lines 396-401: Added fp8_e5m2 skip (alibi test)
   - Lines 469-479: Changed dtype to torch.int (alibi test)
   - Lines 484-494: Fixed cache population for decode sequences (alibi test)

## Key Insight

This was a **TEST BUG**, not a kernel bug. The test incorrectly simulated the cache state that production code would provide. The ROCm custom paged attention kernel is correct when given properly populated cache according to its contract.
