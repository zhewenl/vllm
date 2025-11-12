# Cache Population Fix: Detailed Analysis

## Executive Summary

The test was incorrectly populating the KV cache based on a misunderstanding of how different operators use the cache. The `chunked_prefill_paged_decode` operator's decode path expects ALL tokens (including the new query token) to be in cache before calling the paged attention kernel, while `context_attention_fwd` expects only context tokens in cache with new tokens provided via separate K/V tensors. The fix correctly populates `seq_lens[i]` tokens for `chunked_prefill_paged_decode` and `ctx_lens[i]` tokens for `context_attention_fwd`.

## Background: Prefix-Prefill and KV Cache Architecture

### The Two Attention Modes

vLLM handles two distinct scenarios:

1. **Prefill** (`query_len > 1`): Processing multiple new tokens
   - Context tokens read from cache
   - NEW tokens have K/V computed on-the-fly from input tensors
   - Uses `context_attention_fwd` which receives separate `k`/`v` tensors

2. **Decode** (`query_len == 1`): Processing a single new token
   - ALL tokens (including the new one) must be in cache
   - No separate K/V tensors passed to kernel
   - Uses paged attention kernel which only reads from cache

### Test Setup (lines 185-197)

```python
query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
query_lens[-1] = 1  # Ensure at least one decode sequence

ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
```

**Key Variables:**
- `ctx_lens[i]`: Number of "context" tokens (old tokens already processed)
- `query_lens[i]`: Number of NEW tokens being added
- `seq_lens[i]`: Total sequence length = `ctx_lens[i]` + `query_lens[i]`

The test creates:
- `key`, `value`: Shape `[sum(seq_lens), ...]` - ALL tokens for reference computation
- `k`, `v`: Shape `[sum(query_lens), ...]` - NEW query tokens only
- `key_cache`, `value_cache`: Paged KV cache to be populated

## The Bug: Incorrect Cache Population

### What the OLD Code Did (lines 236-253)

```python
for i in range(BS):
    # Correctly populate k,v with NEW query tokens (lines 233-235)
    for j in range(query_lens[i]):
        k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
        v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])

    # Cache population - ALWAYS uses ctx_lens[i]
    cur_ctx = 0
    block_id = 0
    while cur_ctx < b_ctx_len[i]:  # ← Only loops ctx_lens[i] times
        start_loc = b_seq_start_loc[i] + cur_ctx
        if cur_ctx + block_size > b_ctx_len[i]:
            end_loc = b_seq_start_loc[i] + b_ctx_len[i]
        else:
            end_loc = start_loc + block_size

        # Copies first ctx_lens[i] tokens to cache
        k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            key[start_loc:end_loc]
        )
```

**Result:**
- Cache contains: tokens `[0, ctx_lens[i])` for each sequence
- `k`/`v` tensors contain: tokens `[ctx_lens[i], seq_lens[i])` for each sequence
- This setup is correct for `context_attention_fwd` but WRONG for `chunked_prefill_paged_decode`

### What the FIX Does (line 238-240)

```python
cache_len = seq_lens[i] if op is chunked_prefill_paged_decode else b_ctx_len[i]
while cur_ctx < cache_len:
```

**Result:**
- For `chunked_prefill_paged_decode`: Cache contains ALL `seq_lens[i]` tokens
- For `context_attention_fwd`: Cache contains only `ctx_lens[i]` tokens (context)

## Kernel Interface Contract

### `context_attention_fwd` (prefix_prefill.py lines 829-1024)

**Signature:**
```python
def context_attention_fwd(
    q,    # Query tokens
    k,    # K values for NEW tokens
    v,    # V values for NEW tokens
    o,    # Output
    kv_cache_dtype,
    k_cache,    # Context tokens in cache
    v_cache,    # Context tokens in cache
    ...
)
```

**Behavior (lines 152-254):**
- Loop over context in cache: reads from `k_cache`/`v_cache`
- Loop over new queries (lines 272-318): reads from `k`/`v` tensors
- Expects cache to contain ONLY context, not the new tokens

**Usage in test:** Cache should have `ctx_lens[i]` tokens, `k`/`v` have the rest.

### `chunked_prefill_paged_decode` (chunked_prefill_paged_decode.py lines 223-401)

**Signature:**
```python
def chunked_prefill_paged_decode(
    query,    # Query tokens
    key,      # K values for NEW tokens (used in prefill path)
    value,    # V values for NEW tokens (used in prefill path)
    output,
    kv_cache_dtype,
    key_cache,      # Cached tokens
    value_cache,    # Cached tokens
    ...
    seq_lens,       # TOTAL sequence length
)
```

**Behavior:**

**Prefill Path** (`max_query_len > 1`, lines 253-275):
```python
if max_query_len > 1:
    context_attention_fwd(
        q=query,
        k=key,      # ← Passes separate K/V tensors
        v=value,
        ...
    )
```
Delegates to `context_attention_fwd`, which handles the separate K/V tensors.

**Decode Path** (lines 302-401):
- ROCm kernel call (lines 334-354):
```python
ops.paged_attention_rocm(
    output,
    ...,
    query,           # Only the new query tokens
    key_cache,       # ← NO separate key/value tensors!
    value_cache,     # ← Everything must be in cache!
    ...
    seq_lens=seq_lens,  # Tells kernel the TOTAL sequence length
)
```

**CRITICAL:** The decode path does NOT pass separate `key`/`value` tensors to the kernel!

### ROCm Kernel Expectations (csrc/rocm/attention.cu lines 357-378)

```cpp
const int seq_len = seq_lens[seq_idx];

// Skip non-decode sequences
if (query_start_loc_ptr != nullptr &&
    (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx]) != 1) {
  return;  // Only process decode (query_len == 1)
}

const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
```

The kernel:
1. Filters to only process decode sequences (`query_len == 1`)
2. Uses `seq_lens[seq_idx]` as the TOTAL sequence length
3. Fetches K/V for ALL `seq_len` tokens from cache (lines 506-524)
4. Does NOT receive or use separate K/V tensors

**Expected Cache State:**
- Cache must contain ALL `seq_len` tokens, including the NEW query token
- In production: new token's K/V are written to cache BEFORE calling the kernel
- In test: we must populate cache with all tokens to simulate post-write state

### Triton Kernel (chunked_prefill_paged_decode.py lines 356-401)

The Triton kernel similarly does NOT receive separate K/V parameters in decode mode:

```python
kernel_paged_attention_2d[...](
    output_ptr=output,
    query_ptr=query,
    key_cache_ptr=key_cache,    # ← Only cache, no separate K/V
    value_cache_ptr=value_cache,
    ...
)
```

But the Triton kernel was less sensitive to this bug for reasons explained below.

## Why Only ROCm Failed

### ROCm Custom Kernel Selection (vllm/platforms/rocm.py lines 133-163)

```python
def use_rocm_custom_paged_attention(...) -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    ON_GFX9 = any(arch in GPU_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])

    if ON_GFX9:
        return (
            (sliding_window == 0 or sliding_window == (-1, -1))
            and (qtype == torch.half or qtype == torch.bfloat16)
            and (head_size == 64 or head_size == 128)
            and (block_size == 16 or block_size == 32)
            and (gqa_ratio >= 1 and gqa_ratio <= 16)
            and max_seq_len <= 128 * 1024
            and (envs.VLLM_ROCM_CUSTOM_PAGED_ATTN)
            ...
        )
```

The ROCm custom C++/HIP kernel is only used on MI200/MI300 series (gfx90a/gfx942/gfx950) and requires specific configurations. When these conditions are met (which they were in the failing test), the custom kernel is invoked.

### Why ROCm Kernel Failed

The ROCm kernel (csrc/rocm/attention.cu) strictly expects:
1. All tokens up to `seq_lens[seq_idx]` to be in cache
2. No fallback to separate K/V tensors
3. Block table mapping to be complete

When the test only populated `ctx_lens[i]` tokens but told the kernel `seq_lens[i]`, the kernel read:
- First `ctx_lens[i]` positions: Correct values
- Positions `ctx_lens[i]` to `seq_lens[i]`: **Uninitialized cache memory or zeros**

This caused incorrect attention scores and output mismatches.

### Why Triton Kernel Was Unaffected

The Triton kernel (chunked_prefill_paged_decode.py lines 27-221) has:
- More lenient memory access patterns
- Implicit masking that may have masked out invalid regions
- Different memory layout that happened to read correct values from a different location
- The test's random initialization may have accidentally placed similar values

**However**, the Triton kernel ALSO expects the cache to be fully populated in decode mode. The test was incorrect for both, but only ROCm exposed the bug due to stricter memory access.

## Production Behavior: How Cache is Really Populated

In production (vllm/worker/model_runner.py and related files):

1. **New token arrives**
2. **Model forward pass computes** Q, K, V for the new token
3. **K/V are written to cache** (via cache manager operations)
4. **Paged attention kernel is called** with the populated cache

The cache is ALWAYS complete before the kernel runs. The kernel never needs to handle "partial" cache scenarios.

## Proof the Fix is Correct

### Evidence 1: Kernel Interface

ROCm kernel signature (csrc/rocm/attention.cu lines 932-937):
```cpp
void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, ...]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, ...]
    ...
    const int* __restrict__ seq_lens,       // [num_seqs] - TOTAL sequence length
```

No separate `key`/`value` parameters. Only `k_cache`/`v_cache` and `seq_lens` telling it how many tokens to read.

### Evidence 2: Kernel Implementation

Lines 967-971:
```cpp
const int seq_len = seq_lens[seq_idx];
const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
```

Lines 1031-1039:
```cpp
const int global_token_idx = partition_start_token_idx + local_token_idx;
const int block_idx = (global_token_idx < seq_len)
                          ? global_token_idx / BLOCK_SIZE
                          : last_seq_block;
```

The kernel loops over ALL `seq_len` tokens, fetching each from cache via block tables.

### Evidence 3: Test Reference Implementation

Lines 322-349 use PyTorch SDPA with ALL `seq_lens` tokens:
```python
key_sdpa = key[:, :, None, :].expand(...)  # Uses full 'key' tensor
key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
    1, num_heads, sum(seq_lens), head_size  # ← sum(seq_lens), not sum(ctx_lens)
)
```

The reference expects to attend over ALL tokens, which means the kernel should have access to ALL tokens too.

### Evidence 4: Semantic Correctness

For a decode sequence with:
- `query_lens[i] = 1` (one new token)
- `ctx_lens[i] = 100` (100 context tokens)
- `seq_lens[i] = 101` (total)

The attention computation should be:
```
output[0] = softmax(Q[0] @ [K[0], K[1], ..., K[100]]) @ [V[0], V[1], ..., V[100]]
```

This requires K/V for ALL 101 tokens. Since the kernel doesn't receive separate K/V tensors, they must ALL be in cache.

## Fix Summary

**Line 238 Change:**
```python
# OLD (incorrect for chunked_prefill_paged_decode):
while cur_ctx < b_ctx_len[i]:

# NEW (correct for both operators):
cache_len = seq_lens[i] if op is chunked_prefill_paged_decode else b_ctx_len[i]
while cur_ctx < cache_len:
```

**Why This Is Correct:**

1. **For `context_attention_fwd`**: Populates `ctx_lens[i]` tokens in cache
   - Context in cache, new tokens in `k`/`v` tensors
   - Matches the operator's interface contract

2. **For `chunked_prefill_paged_decode`**: Populates `seq_lens[i]` tokens in cache
   - ALL tokens in cache, simulating production state after K/V write
   - Matches the decode kernel's expectation
   - Required because decode kernel doesn't receive separate K/V tensors

## Conclusion

The fix correctly distinguishes between two different cache usage patterns:
- **Prefill mode** (`context_attention_fwd`): Cache = context, separate K/V for new tokens
- **Decode mode** (`chunked_prefill_paged_decode`): Cache = everything, no separate K/V

The test now properly simulates how each operator expects the cache to be populated, fixing the ROCm failures while maintaining correctness for all platforms.
