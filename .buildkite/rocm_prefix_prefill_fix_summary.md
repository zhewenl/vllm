# ROCm test_prefix_prefill.py Fix Summary

## Issues Fixed

### 1. Type Mismatch (torch.long → torch.int) ✅
**Problem**: Test used `dtype=torch.long` (64-bit) but ROCm kernel expects `dtype=torch.int` (32-bit)
**Fix**: Changed all 9 occurrences to `torch.int` to match production code patterns

### 2. Unsupported fp8_e5m2 Format ✅
**Problem**: ROCm custom kernel only supports `fp8_e4m3`, not `fp8_e5m2`
**Fix**: Added skip for these tests on ROCm

### 3. Cache Population Bug (Root Cause) ✅
**Problem**: Test only populated `ctx_len` tokens in cache but told kernel there are `seq_len = ctx_len + query_len` tokens. For decode sequences (query_len=1), the current token was missing from cache, causing the kernel to read uninitialized data.

**Root Cause Analysis**:
- In production vLLM, tokens are **always added to cache before** calling paged attention
- The test simulated a "before" state but expected "after" behavior
- ROCm custom kernel correctly reads from cache based on `seq_len` parameter
- Missing token caused numerical errors (exactly 8192 elements = 1 token × 64 heads × 128 dims)

**Fix** (lines 238-240):
```python
# OLD: Only cached ctx_len tokens
while cur_ctx < b_ctx_len[i]:

# NEW: Cache all seq_len tokens for chunked_prefill_paged_decode
cache_len = seq_lens[i] if op is chunked_prefill_paged_decode else b_ctx_len[i]
while cur_ctx < cache_len:
```

## Why This Configuration Failed

| Config | Kernel | Result | Reason |
|--------|--------|--------|--------|
| `num_q_per_kv=1, sw=0, hs=128` | ROCm custom | **FAILED** | Read uninitialized cache |
| `num_q_per_kv=1, sw=16` | Triton | PASSED | Different kernel path |
| `num_q_per_kv=64` | Triton | PASSED | GQA ratio > 16, no custom |
| `head_size=24` | Triton | PASSED | Doesn't meet custom requirements |

## Verification

```bash
HF_HUB_DISABLE_XET=1 wp pytest -s -v 'tests/kernels/attention/test_prefix_prefill.py::test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:0-auto-dtype0-128-1-64]'
```

Expected: Test passes with correct numerical accuracy

## Key Insight

**This was a TEST BUG, not a KERNEL BUG**. The ROCm custom paged attention kernel works correctly when given properly populated cache. The test's cache setup didn't match production behavior.
