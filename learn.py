"""
Final Refactored test_prefix_prefill.py using PyTorch SDPA
===========================================================
This replaces xformers with PyTorch's native scaled_dot_product_attention
"""

import math
import random
import time
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.platforms import current_platform

# Test parameters
NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96, 24]
DTYPES = [torch.float16, torch.bfloat16]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
SLIDING_WINDOW = [0, 16, 64, 128, 256, 512, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]


def compute_ground_truth_with_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    context_len: int,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute ground truth attention using PyTorch's SDPA.
    This replaces the xformers implementation.
    
    Args:
        query: Query tensor [num_queries, num_heads, head_size]
        key: Key tensor [total_tokens, num_kv_heads, head_size]
        value: Value tensor [total_tokens, num_kv_heads, head_size]
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA/MQA)
        scale: Attention scale factor
        context_len: Length of the context/prefix
        sliding_window: Optional sliding window size
    
    Returns:
        Attention output [num_queries, num_heads, head_size]
    """
    num_queries = query.shape[0]
    total_tokens = key.shape[0]
    head_size = query.shape[-1]
    
    # Calculate batch size and sequence lengths
    batch_size = num_queries // (total_tokens - context_len)
    new_len = total_tokens - context_len
    
    # Reshape tensors to [batch, heads, seq_len, head_dim]
    # Query is only for new tokens
    query_reshaped = query.view(batch_size, new_len, num_heads, head_size)
    query_reshaped = query_reshaped.transpose(1, 2)  # [batch, heads, new_len, head_dim]
    
    # Key and Value include all tokens (context + new)
    key_reshaped = key.view(batch_size, total_tokens // batch_size, num_kv_heads, head_size)
    value_reshaped = value.view(batch_size, total_tokens // batch_size, num_kv_heads, head_size)
    
    key_reshaped = key_reshaped.transpose(1, 2)  # [batch, kv_heads, total_len, head_dim]
    value_reshaped = value_reshaped.transpose(1, 2)  # [batch, kv_heads, total_len, head_dim]
    
    # Handle GQA/MQA by repeating KV heads if necessary
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        key_reshaped = key_reshaped.repeat_interleave(repeat_factor, dim=1)
        value_reshaped = value_reshaped.repeat_interleave(repeat_factor, dim=1)
    
    # Create attention mask for prefix-prefill pattern
    # New tokens can see all prefix tokens and previous new tokens (causal)
    seq_len_per_batch = total_tokens // batch_size
    attn_mask = None
    
    if sliding_window is not None and sliding_window > 0:
        # Create sliding window mask
        attn_mask = torch.full(
            (batch_size, 1, new_len, seq_len_per_batch),
            float('-inf'),
            dtype=query.dtype,
            device=query.device
        )
        
        for i in range(new_len):
            query_pos = context_len + i
            # Can see all prefix tokens
            attn_mask[:, :, i, :context_len] = 0
            # Can see new tokens within sliding window
            start = max(context_len, query_pos - sliding_window + 1)
            end = query_pos + 1
            attn_mask[:, :, i, start:end] = 0
    else:
        # Standard causal mask for new tokens attending to prefix + previous new
        attn_mask = torch.zeros(
            (batch_size, 1, new_len, seq_len_per_batch),
            dtype=query.dtype,
            device=query.device
        )
        
        for i in range(new_len):
            current_pos = context_len + i
            # Block future positions
            attn_mask[:, :, i, current_pos + 1:] = float('-inf')
    
    # Apply SDPA
    output = F.scaled_dot_product_attention(
        query_reshaped,
        key_reshaped,
        value_reshaped,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
        is_causal=False,  # We handle causality through our mask
    )
    
    # Reshape output back to original format
    output = output.transpose(1, 2)  # [batch, new_len, heads, head_dim]
    output = output.reshape(num_queries, num_heads, head_size)
    
    return output


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
    kv_cache_dtype: str,
    sliding_window: int,
):
    """Test prefix-prefill attention using SDPA as ground truth."""
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Skip FP8 tests if not supported
    if kv_cache_dtype in ["fp8", "fp8_e5m2"]:
        if not current_platform.has_device_capability(89):
            pytest.skip("FP8 not supported on this GPU")
    
    # Test configuration
    random.seed(42)
    torch.manual_seed(42)
    
    batch_size = 4
    block_size = 16
    context_len = 128
    num_new_tokens = 32
    num_kv_heads = num_heads // num_queries_per_kv
    
    max_len = context_len + num_new_tokens
    num_queries = batch_size * num_new_tokens
    total_tokens = batch_size * max_len
    
    # Create random input tensors
    scale = float(1.0 / math.sqrt(head_size))
    
    # Query is only for new tokens
    query = torch.randn(
        num_queries, num_heads, head_size,
        dtype=dtype, device=device
    )
    
    # Key and Value are for all tokens (context + new)
    key = torch.randn(
        total_tokens, num_kv_heads, head_size,
        dtype=dtype, device=device
    )
    value = torch.randn(
        total_tokens, num_kv_heads, head_size,
        dtype=dtype, device=device
    )
    
    # Prepare KV cache for custom kernel
    num_blocks = (max_len + block_size - 1) // block_size
    
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    elif kv_cache_dtype == "fp8":
        cache_dtype = torch.float8_e4m3fn
    elif kv_cache_dtype == "fp8_e5m2":
        cache_dtype = torch.float8_e5m2
    else:
        cache_dtype = dtype
    
    k_cache = torch.zeros(
        (batch_size * num_blocks, num_kv_heads, block_size, head_size),
        dtype=cache_dtype, device=device
    )
    v_cache = torch.zeros(
        (batch_size * num_blocks, num_kv_heads, block_size, head_size),
        dtype=cache_dtype, device=device
    )
    
    # Fill cache with context tokens
    block_table = torch.arange(
        batch_size * num_blocks, dtype=torch.int32, device=device
    ).reshape(batch_size, num_blocks)
    
    # Fill the cache with context data
    for batch_idx in range(batch_size):
        for pos in range(context_len):
            block_idx = pos // block_size
            block_offset = pos % block_size
            global_block_idx = block_table[batch_idx, block_idx]
            
            token_idx = batch_idx * max_len + pos
            k_cache[global_block_idx, :, block_offset, :] = key[token_idx].to(cache_dtype)
            v_cache[global_block_idx, :, block_offset, :] = value[token_idx].to(cache_dtype)
    
    # Prepare metadata for custom kernel
    b_start_loc = torch.arange(
        0, batch_size * num_new_tokens + 1, num_new_tokens,
        dtype=torch.int32, device=device
    )
    b_seq_len = torch.full(
        (batch_size,), max_len, dtype=torch.int32, device=device
    )
    b_ctx_len = torch.full(
        (batch_size,), context_len, dtype=torch.int32, device=device
    )
    
    # Scale tensors for FP8
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)
    
    # Extract new token keys and values for custom kernel
    key_new = []
    value_new = []
    for batch_idx in range(batch_size):
        start = batch_idx * max_len + context_len
        end = start + num_new_tokens
        key_new.append(key[start:end])
        value_new.append(value[start:end])
    
    key_new = torch.cat(key_new, dim=0)
    value_new = torch.cat(value_new, dim=0)
    
    # Run custom kernel
    output_custom = torch.zeros_like(query)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    context_attention_fwd(
        query,
        key_new,
        value_new,
        output_custom,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        b_ctx_len,
        max_len,
        num_new_tokens,
        k_scale,
        v_scale,
        alibi_slopes=None,
        sliding_window=sliding_window if sliding_window > 0 else None,
        sm_scale=scale,
    )
    
    torch.cuda.synchronize()
    custom_time = (time.time() - start_time) * 1000
    
    # Compute ground truth with SDPA
    torch.cuda.synchronize()
    start_time = time.time()
    
    output_sdpa = compute_ground_truth_with_sdpa(
        query=query,
        key=key,
        value=value,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        scale=scale,
        context_len=batch_size * context_len,  # Total context tokens
        sliding_window=sliding_window if sliding_window > 0 else None,
    )
    
    torch.cuda.synchronize()
    sdpa_time = (time.time() - start_time) * 1000
    
    # Compare outputs
    torch.testing.assert_close(
        output_custom,
        output_sdpa,
        rtol=1e-3 if dtype != torch.float32 else 1e-5,
        atol=1e-3 if dtype != torch.float32 else 1e-5,
        msg=f"Mismatch for config: heads={num_heads}, kv_heads={num_kv_heads}, "
        f"head_size={head_size}, dtype={dtype}, kv_cache={kv_cache_dtype}, "
        f"sliding_window={sliding_window}"
    )
    
    print(f"\n✅ Test passed! Custom: {custom_time:.2f}ms, SDPA: {sdpa_time:.2f}ms")


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.optional
def test_contexted_kv_attention_f32(dtype: torch.dtype):
    """Optional float32 test for Turing GPUs."""
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Simple test configuration
    test_contexted_kv_attention(
        num_heads=8,
        num_queries_per_kv=1,
        head_size=64,
        dtype=dtype,
        device="cuda:0",
        kv_cache_dtype="auto",
        sliding_window=0,
    )


if __name__ == "__main__":
    """Run a simple test when executed directly."""
    print("Running prefix-prefill test with SDPA ground truth...")
    
    if torch.cuda.is_available():
        test_contexted_kv_attention(
            num_heads=8,
            num_queries_per_kv=1,
            head_size=64,
            dtype=torch.float16,
            device="cuda:0",
            kv_cache_dtype="auto",
            sliding_window=0,
        )
        print("\n✅ All tests passed!")
    else:
        print("CUDA not available for testing")
