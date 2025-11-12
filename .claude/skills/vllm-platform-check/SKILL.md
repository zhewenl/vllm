---
name: vllm-platform-check
description: Check if issues are platform-specific (ROCm/CUDA) or v0 code path related
---

# vLLM Platform Check Skill

This skill helps you quickly identify whether an issue is platform-specific and whether it affects the v0 or v1 code path.

## What This Skill Does

When analyzing bugs, test failures, or code changes, this skill determines:

1. **Platform Specificity**: Is it AMD/ROCm only, NVIDIA/CUDA only, or both?
2. **Code Path**: Does it affect v0, v1, or both?
3. **Priority**: Should this issue be deprioritized (v0 issues are lower priority)?

## Platform Indicators

### ROCm/AMD Specific
- Code in files with `rocm`, `hip`, or `amd` in the path
- Use of `torch.hip` or ROCm-specific APIs
- Triton attention backend issues
- References to `gfx` architectures (gfx90a, gfx942, etc.)

### CUDA/NVIDIA Specific
- Code in files with `cuda`, `nvcc`, or `nvidia` in the path
- Use of FlashAttention, FlashInfer, or xFormers
- CUTLASS kernels
- References to compute capabilities (sm_80, sm_90, etc.)

### Cross-Platform
- Generic attention code in `vllm/attention/`
- Model implementations in `vllm/model_executor/models/`
- Engine and worker code (usually platform-agnostic)

## v0 vs v1 Detection

### v0 Code Path Indicators
- Files in `vllm/` that predate v1 refactoring
- Older engine implementations
- Legacy attention mechanisms
- Code with `VLLM_USE_V1` guards set to False

### v1 Code Path Indicators
- Files in `vllm/v1/` directory
- New engine architecture
- Unified prefix-caching
- Code with `VLLM_USE_V1` guards set to True
- References to disaggregated prefill

## Instructions

When the user invokes this skill or when analyzing an issue:

1. **Examine the code paths involved**:
   - Check file paths for platform-specific directories
   - Look for imports from ROCm/CUDA specific modules
   - Search for `torch.cuda` vs `torch.hip` usage

2. **Check for v0/v1 indicators**:
   - Look for `vllm/v1/` in file paths
   - Search for `VLLM_USE_V1` environment variable usage
   - Check if tests use `@skip_v1` markers

3. **Provide early assessment**:
   - State platform specificity clearly
   - Indicate if it's a v0 issue
   - **If it's a v0 issue, note that it's not important/lower priority**

4. **Suggest appropriate action**:
   - Platform-specific: Test on that platform only
   - v0 issue: Deprioritize or skip
   - Cross-platform: Test on both AMD and NVIDIA

## Example Output

```
Platform Check Results:
- Platform: ROCm/AMD specific
- Code Path: v0 only
- Priority: Low (v0 issues are not important)
- Reason: File is in legacy attention path, uses torch.hip APIs

Recommendation: This is a v0 ROCm-specific issue and can be deprioritized.
```

## Notes

- v1 is the current focus of vLLM development
- v0 issues are lower priority unless critical
- Platform-specific issues should be verified on the appropriate hardware
- Cross-platform issues require testing on both ROCm and CUDA
