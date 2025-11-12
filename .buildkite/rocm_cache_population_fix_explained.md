# Why The Cache Population Fix Is Correct

## Executive Summary

The fix `cache_len = seq_lens[i] if op is chunked_prefill_paged_decode else b_ctx_len[i]` is **CORRECT**.

The decode kernel expects ALL `seq_len` tokens in cache (including the current query token), not just `ctx_len` context tokens. The test was buggy - it only populated `ctx_len` tokens but told the kernel there are `seq_len` tokens.

## Background: Two Kernel Paths

`chunked_prefill_paged_decode` handles mixed batches with both prefill and decode sequences:

### Path 1: Prefill Sequences (query_len > 1)
```python
# vllm/attention/ops/chunked_prefill_paged_decode.py:253-275
if max_query_len > 1:
    context_attention_fwd(
        q=query,
        k=key,           # NEW query tokens
        v=value,         # NEW query tokens
        k_cache=key_cache,  # CONTEXT tokens in cache
        v_cache=value_cache,
        ...
        skip_decode=True  # Skips sequences with query_len == 1
    )
```

This kernel gets separate `key`/`value` parameters for new tokens.

### Path 2: Decode Sequences (query_len == 1)
```python
# Line 334-354: ROCm OR Line 356-399: Triton
kernel_paged_attention_2d(
    query_ptr=query,
    key_cache_ptr=key_cache,    # ONLY source of K/V!
    value_cache_ptr=value_cache,
    ...
    seq_lens_ptr=seq_lens,  # Total sequence length
)
```

**CRITICAL**: Decode kernel does NOT get separate `key`/`value` parameters! It ONLY gets the cache.

## The Test Bug

### What The Test Did (WRONG)
```python
# Line 238 (BEFORE fix)
while cur_ctx < b_ctx_len[i]:  # Only ctx_len tokens
    ...
    k_cache[...].copy_(key[start_loc:end_loc])
```

For a decode sequence with `ctx_len=100, query_len=1`:
- Cache contains: 100 tokens (positions [0:100])
- Test passes `seq_lens[i] = 101` to kernel
- Kernel tries to load 101 tokens from cache
- Token at position 100 is MISSING → reads garbage → numerical error

### What The Test Should Do (CORRECT)
```python
# Line 240 (AFTER fix)
cache_len = seq_lens[i] if op is chunked_prefill_paged_decode else b_ctx_len[i]
while cur_ctx < cache_len:  # All seq_len tokens
    ...
    k_cache[...].copy_(key[start_loc:end_loc])
```

Now cache contains all 101 tokens as expected.

## Proof: How Decode Kernel Uses seq_len

### Loads seq_len Blocks From Cache
```python
# vllm/attention/ops/chunked_prefill_paged_decode.py:118,126
seq_len = tl.load(seq_lens_ptr + seq_idx)
num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)  # Loads ceil(seq_len/block_size) blocks!

for j in range(0, num_blocks):
    physical_block_idx = tl.load(block_tables_ptr + ...)
    K_load = tl.load(key_cache_ptr + k_offset, ...)  # Loads from cache
```

If cache only has `ctx_len` tokens but kernel loads `seq_len` blocks, it reads past valid data!

### Computes context_len For Attention
```python
# Line 174
context_len = seq_len - 1  # Excludes current token from attention

# Line 180 (ALiBi example)
S += alibi_slope[:, None] * (seq_offset - context_len)
```

So the kernel:
1. **Loads** `seq_len` tokens from cache (including current)
2. **Attends** to `context_len = seq_len - 1` tokens (excluding current)

This maintains causality while keeping all data in cache.

## Why Only ROCm Custom Kernel Failed

Different parameter combinations trigger different kernels:

| Config | Kernel Used | Has Bug? |
|--------|-------------|----------|
| `num_q_per_kv=1, sw=0, hs=128` | **ROCm custom** | **YES** - reads past cache |
| `num_q_per_kv=1, sw=16` | Triton | NO - sliding window disables ROCm |
| `num_q_per_kv=64` | Triton | NO - gqa_ratio=64 > 16 max |
| `head_size=24` | Triton | NO - ROCm requires hs=128 |

All decode kernels have this requirement, but only ROCm custom was triggered by test parameters.

## Error Signature Confirms Root Cause

```
Mismatched elements: 8192 / 43720704 (0.0%)
Greatest absolute difference: 1.685546875 at index (5336, 0, 0)
```

`8192 elements = 1 token × 64 heads × 128 head_dim` - exactly the missing decode token!

## Production Behavior

In production vLLM, the attention backend adds tokens to cache **before** calling paged attention:

1. New token KVs are computed
2. KVs are written to cache
3. Paged attention is called with updated `seq_len`
4. Kernel reads all tokens from cache

The test should simulate this same flow.

## Conclusion

The fix is **correct and necessary**. It makes the test match production semantics where:
- For decode: ALL `seq_len` tokens must be in cache before calling paged attention
- For prefill: NEW query tokens are passed separately via `key`/`value` parameters

The test was violating the kernel's contract by providing incomplete cache data.
