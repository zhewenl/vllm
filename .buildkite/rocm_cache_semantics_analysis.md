# ROCm Cache Population Semantics Analysis

## Question
Is my fix correct: `cache_len = seq_lens[i] if op is chunked_prefill_paged_decode else b_ctx_len[i]`?

This populates the cache with ALL seq_len tokens (context + query) instead of just ctx_len tokens (context only).

## Test Setup

```python
# Line 192-197
query_lens[i] = # NEW tokens being processed (1 for decode)
ctx_lens[i] = # CONTEXT tokens (already processed)
seq_lens[i] = query_lens[i] + ctx_lens[i]  # TOTAL sequence length
```

```python
# Line 219-220: NEW tokens only
k = torch.zeros(sum(query_lens), ...)  # shape: [sum(query_lens), ...]
v = torch.zeros(sum(query_lens), ...)  # shape: [sum(query_lens), ...]

# Line 233-235: Copy query tokens from full KV to k, v
k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
# This copies tokens at positions [ctx_len : seq_len] from key -> k
```

## Key Discovery: Different Kernel Paths

### Path 1: max_query_len > 1 (Has Prefill)
```python
# Line 253-275: Called FIRST
context_attention_fwd(
    q=query,
    k=key,        # NEW tokens
    v=value,      # NEW tokens
    k_cache=key_cache,  # CONTEXT tokens
    v_cache=value_cache,
    ...
    skip_decode=True  # Skips decode sequences!
)
```

### Path 2: Decode Kernel (ROCm or Triton)
```python
# Line 334-354: ROCm path
ops.paged_attention_rocm(
    query,
    key_cache,     # ONLY cache, no separate key/value!
    value_cache,
    ...
    seq_lens,      # Total sequence length
)

# Line 356-399: Triton path
kernel_paged_attention_2d(
    query_ptr=query,
    key_cache_ptr=key_cache,   # ONLY cache!
    value_cache_ptr=value_cache,
    ...
    seq_lens_ptr=seq_lens,
)
```

**CRITICAL**: Both decode kernels do NOT receive separate `key`/`value` parameters for new tokens! They ONLY get the cache.

### Decode Kernel Filtering
```python
# Line 73-78 in kernel_paged_attention_2d
if filter_by_query_len:
    if cur_batch_query_len > 1:
        return  # Skip prefill sequences
```

So the decode kernel ONLY processes sequences with `query_len == 1`.

## The Problem

For decode sequences (`query_len == 1`):
1. `context_attention_fwd` is NOT called for them (`skip_decode=True`)
2. The decode kernel processes them
3. The decode kernel expects ALL KV tokens in cache (no separate `key`/`value` params)
4. But the test only populates `ctx_len` tokens in cache
5. **The decode kernel tries to attend to `seq_len` tokens but only `ctx_len` are in cache**
6. Missing token → reads uninitialized data → numerical error

## Evidence

Error affects exactly **8192 elements = 1 token × 64 heads × 128 dims**, confirming it's the missing decode token.

## Counter-Argument: Shouldn't Query Token Be Separate?

**Question**: In attention, we compute attention FOR the query token, not OVER it. Why would the query token be in cache?

**Answer**: The semantics of `seq_lens` parameter:
- For prefill: `seq_lens[i]` = total tokens to attend OVER (context only, query tokens handled separately in k/v params)
- For decode: `seq_lens[i]` = total tokens INCLUDING the current token being processed

This is because decode uses a different kernel that expects everything in cache.

##  TODO: Verify Production Behavior

Need to check how production vLLM handles decode:
1. Are tokens added to cache BEFORE calling paged_attention?
2. Or AFTER?
3. What is `seq_lens` for a decode request?

Let me search for this...
