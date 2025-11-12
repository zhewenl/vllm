---
name: vllm-ci-analysis
description: Analyze AMD/NVIDIA CI failures from Buildkite and create detailed failure reports
---

# vLLM CI Analysis Skill

This skill helps you analyze CI failures from Buildkite for both AMD and NVIDIA platforms, create comprehensive failure reports, and provide reproduction steps.

## CI Configuration Files

**NVIDIA Tests**: `.buildkite/test-pipeline.yaml`
**AMD Tests**: `.buildkite/test-amd.yaml`

## Analysis Workflow

When investigating CI failures:

1. **Identify the platform**: AMD (ROCm) or NVIDIA (CUDA)
2. **Locate the relevant Buildkite YAML file** and understand the test configuration
3. **Analyze the failure logs** to identify:
   - Test name and location
   - Error messages and stack traces
   - Platform-specific code paths
   - Whether it's a v0 issue (if so, deprioritize)
4. **Create a detailed analysis report** (only on initial analysis)
5. **Provide reproduction steps** using pytest commands

## Output Location

Analysis reports should be created in:
```
.claude/ci-analysis/<date-ci-name>/
```

For example:
- `.claude/ci-analysis/2025-01-15-amd-tool-parser/rocm_failure.md`
- `.claude/ci-analysis/2025-01-15-nvidia-attention/cuda_failure.md`

## Report Structure

Each analysis markdown file should include:

### 1. Summary
- Brief description of the failure
- Affected platform(s)
- Test name and file location

### 2. Findings
- Root cause analysis
- Relevant code paths
- Platform-specific considerations
- v0 vs v1 determination

### 3. Reproduction Steps
- Exact pytest commands to reproduce locally
- Required environment setup
- Expected vs actual behavior

### 4. CI Configuration
- Relevant sections from Buildkite YAML
- Test invocation details

## Instructions

When the user invokes this skill to analyze a CI failure:

1. **ALWAYS use extended thinking (ultrathink)** for thorough analysis
2. **First check**: Is this a platform-specific issue (ROCm/CUDA)?
3. **Check for v0 code path**: If it's a v0 issue, note that it's not important
4. **Read the appropriate Buildkite YAML file** to understand test configuration
5. **Analyze failure logs** (if provided) or examine test files
6. **Create analysis markdown** in `.claude/ci-analysis/<date-ci-name>/` with:
   - `rocm_failure.md` for AMD issues
   - `cuda_failure.md` for NVIDIA issues
   - Use descriptive date and CI name format
7. **IMPORTANT**: Only create the markdown file on **initial analysis**
   - For subsequent iterations or follow-up questions, provide information directly
   - Do not recreate or overwrite the analysis file
8. **Provide pytest reproduction commands** using the vllm-test skill format:
   ```bash
   PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest \
     'tests/path/to/test.py::test_function' \
     -v -s
   ```

## Example Workflow

```bash
# 1. User reports: "AMD CI is failing on test_tool_calls"

# 2. Claude analyzes:
# - Reads .buildkite/test-amd.yaml
# - Examines tests/tool_use/test_tool_calls.py
# - Checks if it's platform-specific
# - Uses ultrathink for deep analysis

# 3. Claude creates:
# .claude/ci-analysis/2025-01-15-amd-tool-parser/rocm_failure.md

# 4. Claude provides reproduction command:
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest \
  'tests/tool_use/test_tool_calls.py::test_tool_call_and_choice' \
  -v -s
```

## Notes

- Always check both AMD and NVIDIA YAML files if the issue might affect both platforms
- Include relevant code snippets in the analysis
- Link to related issues or PRs if available
- Mention if the failure is intermittent or consistent
- Consider numerical precision differences between platforms
- Check for attention backend differences (FlashAttention vs Triton)
