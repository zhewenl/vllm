# ROCm Custom Paged Attention: Numerical Accuracy Root Cause Analysis

## Executive Summary

**Root Cause**: The test setup has a critical bug where the KV cache is not fully populated for decode sequences. The ROCm custom kernel expects the cache to contain ALL tokens (0...seq_len-1), but the test only populates the first ctx_len tokens, leaving the query token's KV missing from the cache.

**Why it only fails for this specific config**: The issue only manifests with the custom ROCm kernel (`gqa_ratio=1`, `head_size=128`, `sliding_window=0`) because:
1. Other configs use the Triton kernel which may handle boundary conditions differently
2. The ROCm custom kernel reads from uninitialized cache blocks, producing NaN or incorrect values

## Detailed Analysis

### 1. Understanding When ROCm Custom Kernel is Used

From `/data/users/zhewenli/gitrepos/vllm-fork/vllm/platforms/rocm.py:133-176`:

**On GFX9 (gfx90a, gfx942, gfx950)**:
- Uses custom kernel when `gqa_ratio >= 1 and gqa_ratio <= 16`
- Also requires: `sliding_window == 0`, `head_size` in `[64, 128]`, `block_size` in `[16, 32]`

**On GFX11/GFX12**:
- Uses custom kernel when `gqa_ratio >= 3 and gqa_ratio <= 16`
- Also requires: `sliding_window == 0`, `head_size == 128`, `block_size == 16`

**Test parameter mapping**:
- `num_queries_per_kv = 1` → `num_kv_heads = 64`, `gqa_ratio = 1` → Custom kernel used on GFX9 only
- `num_queries_per_kv = 64` → `num_kv_heads = 1`, `gqa_ratio = 64` → Custom kernel NOT used (exceeds max 16)
- `sliding_window = 16` → Custom kernel NOT used (requires sliding_window == 0)

### 2. Test Setup Analysis

From `tests/kernels/attention/test_prefix_prefill.py:195-262`:

```python
BS = 10
query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
query_lens[-1] = 1  # Force one decode sequence

ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]  # Total length

# Cache population - ONLY populates ctx_len tokens!
while cur_ctx < b_ctx_len[i]:  # ← BUG: Should be seq_lens[i]
    start_loc = b_seq_start_loc[i] + cur_ctx
    ...
    k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
        key[start_loc:end_loc]
    )
```

**The Problem**:
- Cache is populated with only `ctx_lens[i]` tokens (positions 0...ctx_len-1)
- But `seq_lens[i] = ctx_lens[i] + query_lens[i]`
- For decode (seq 9): cache contains ctx_len tokens, but seq_len = ctx_len + 1
- **The query token at position ctx_len is MISSING from the cache!**

### 3. Kernel Expectation vs Reality

**ROCm Custom Kernel** (`csrc/rocm/attention.cu:967-1033`):
```cpp
const int seq_len = seq_lens[seq_idx];  // Expects seq_len tokens in cache

// Tries to access tokens 0...seq_len-1
const int global_token_idx = partition_start_token_idx + local_token_idx;
const int block_idx = (global_token_idx < seq_len)
                          ? global_token_idx / BLOCK_SIZE
                          : last_seq_block;

const int64_t physical_block_number =
    static_cast<int64_t>(block_table[block_idx]);

// Reads from cache - may read UNINITIALIZED data for token at position ctx_len!
const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride + ...;
```

**What happens**:
1. For decode sequence 9 with `ctx_len = 500`, `query_len = 1`:
   - Cache contains tokens 0...499 (500 tokens)
   - `seq_len = 501`
   - Kernel tries to access token 500 (at position ctx_len)
   - Token 500 is NOT in cache → reads uninitialized/zero data
2. This uninitialized data propagates through QK matmul and softmax
3. Results in numerical errors at specific output positions

### 4. Why Only This Configuration Fails

**Configuration matrix**:

| Config | gqa_ratio | sliding_window | Kernel Used | Result |
|--------|-----------|----------------|-------------|--------|
| `num_queries_per_kv=1, sw=0` | 1 | 0 | ROCm custom | **FAIL** |
| `num_queries_per_kv=1, sw=16` | 1 | 16 | Triton | PASS |
| `num_queries_per_kv=64, sw=0` | 64 | 0 | Triton (gqa_ratio > 16) | PASS |
| `head_size=24, sw=0` | 1 | 0 | Triton (head_size != 128) | PASS |

**Why Triton passes**:
- The Triton kernel may have better boundary handling
- OR: There's a subtle difference in how it accesses cache blocks
- OR: The Triton kernel has implicit zero-padding for out-of-bounds accesses

### 5. Error Pattern Analysis

```
Mismatched elements: 8192 / 43720704 (0.0%)
Greatest absolute difference: 1.685546875 at index (5336, 0, 0)
```

**Calculation**:
- Total elements: `43720704 ≈ sum(query_lens) * 64 heads * 128 dim`
- Mismatched: `8192 = 1 token * 64 heads * 128 dim`
- **Exactly 1 token's worth of data is incorrect!**

This strongly suggests the decode token (query_lens[-1] = 1) has corrupted output due to reading uninitialized cache data.

## Root Cause: Cache vs seq_len Mismatch

### Semantic Confusion

The test conflates two different concepts:
1. **Context length** (`ctx_lens[i]`): Tokens already in cache
2. **Sequence length** (`seq_lens[i]`): Total tokens to attend to

For **prefill** (query_len > 1):
- Cache contains `ctx_lens[i]` tokens
- New tokens passed separately as `k`, `v`
- `context_attention_fwd` combines both → ✓ Works correctly

For **decode** (query_len = 1):
- Cache should contain `seq_lens[i] = ctx_lens[i] + 1` tokens
- Paged attention kernel expects full cache → ✗ **Missing last token**

### Why This Design Exists

In production vLLM:
1. New token's KV is computed
2. New token's KV is **added to cache** via `cache_kernels`
3. Paged attention is called with updated cache
4. Paged attention sees complete cache with all tokens

In the test:
1. Full KV pre-generated for testing
2. Only context added to cache (simulating "before" state)
3. Paged attention called with incomplete cache
4. **Test expects kernel to handle this, but it doesn't!**

## Proposed Fix

### Option 1: Fix Test Setup (Recommended)

Populate cache with ALL tokens including query tokens for decode sequences:

```python
# After line 262 in test_prefix_prefill.py
for i in range(BS):
    # Add query tokens to cache for decode sequences
    if query_lens[i] == 1:  # Decode
        # Add the single query token to cache
        query_token_idx = b_ctx_len[i]  # Position in sequence
        block_idx = query_token_idx // block_size
        slot_in_block = query_token_idx % block_size

        physical_block = block_table[i, block_idx]
        cache_slot = physical_block * block_size + slot_in_block

        # Copy query token's KV to cache
        src_idx = b_seq_start_loc[i] + query_token_idx
        k_cache.view(-1, num_kv_heads, head_size)[cache_slot].copy_(key[src_idx])
        v_cache.view(-1, num_kv_heads, head_size)[cache_slot].copy_(value[src_idx])
```

### Option 2: Fix Kernel to Use ctx_lens

Modify kernel to use context length instead of sequence length for decode:

```python
# In chunked_prefill_paged_decode.py, pass ctx_lens separately
# Kernel uses ctx_lens for decode (query_len = 1) and seq_lens for prefill
```

This is more invasive and changes the kernel contract.

### Option 3: Document and Skip (Current Approach)

Keep the skip but document the limitation. Not recommended as it hides the real issue.

## Verification Steps

After applying Option 1 fix:

```bash
# Run the specific failing test
HF_HUB_DISABLE_XET=1 wp pytest -s -v \
  'tests/kernels/attention/test_prefix_prefill.py::test_contexted_kv_attention[chunked_prefill_paged_decode-0-cuda:0-auto-dtype0-128-1-64]'

# Should now PASS without the skip
```

## Additional Notes

### Why gqa_ratio=1 is Special

With `gqa_ratio=1`:
- Each KV head has exactly 1 query head
- Kernel processes 1 head per workgroup
- Less redundancy to mask errors compared to `gqa_ratio > 1` where multiple query heads share KV

### Platform Differences

**v0 vs v1**: This is a v0 code path (not v1), so it's lower priority for production, but still should be fixed for test correctness.

**CUDA vs ROCm**: CUDA may use different kernels that handle boundary conditions differently. The test should be platform-agnostic.

## Conclusion

The numerical accuracy issue is caused by a test setup bug, not a kernel bug. The test incorrectly assumes the paged attention kernel can handle incomplete caches, but the kernel expects the cache to contain all tokens up to `seq_len-1`.

The fix is straightforward: populate the cache with the decode token's KV before calling the kernel. This matches production behavior where new tokens are always added to the cache before computing attention.
