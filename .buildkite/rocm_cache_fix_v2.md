# ROCm test_prefix_prefill Cache Population Fix (v2)

## The Actual Bug

My first fix was **WRONG**. I populated cache with `seq_lens[i]` tokens for ALL sequences, but this broke prefill sequences.

## Two Kernel Paths with Different Requirements

### Prefill Sequences (query_len > 1)
```python
# vllm/attention/ops/chunked_prefill_paged_decode.py:253-275
context_attention_fwd(
    q=query,
    k=key,           # NEW query tokens passed separately!
    v=value,         # NEW query tokens passed separately!
    k_cache=key_cache,  # Only CONTEXT tokens
    v_cache=value_cache,
    ...
    skip_decode=True
)
```

**Contract**: Cache should have ONLY `ctx_len` tokens. New tokens passed via `k`/`v` parameters.

### Decode Sequences (query_len == 1)
```python
# Lines 334-354 (ROCm) or 356-401 (Triton)
kernel_paged_attention(
    query_ptr=query,
    key_cache_ptr=key_cache,   # ONLY source of K/V!
    value_cache_ptr=value_cache,
    ...
)
```

**Contract**: Cache must have ALL `seq_len` tokens (including current). No separate `k`/`v` parameters.

## The Correct Fix

```python
# For decode sequences ONLY, populate cache with all seq_len tokens
cache_len = (
    seq_lens[i]
    if (op is chunked_prefill_paged_decode and query_lens[i] == 1)
    else b_ctx_len[i]
)
```

This ensures:
- **Prefill**: Cache has `ctx_len` tokens, new tokens via `k`/`v` → `context_attention_fwd` gets both
- **Decode**: Cache has `seq_len` tokens → decode kernel gets everything from cache

## Why This Matters

The test has:
- 9 prefill sequences with `query_len ∈ [16, 1024]`
- 1 decode sequence with `query_len = 1` (line 194: `query_lens[-1] = 1`)

My first fix broke prefill sequences by adding their query tokens to cache when they should be passed separately.

## Verification

The failing tests all have `query_len=1` in their name pattern:
```
test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:0-auto-dtype0-128-1-64]
                                                                              ^^^^
                            This is num_queries_per_kv, not query_len, but the test setup ensures
                            one sequence has query_len=1
```

The error (8192 elements = 1 token × 64 heads × 128 dims) corresponds to that single decode token.
