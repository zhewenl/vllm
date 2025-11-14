---
name: vllm-ci-analysis
description: Analyze AMD/NVIDIA CI failures from Buildkite and create detailed failure reports
---

# vLLM CI Analysis Skill

This skill helps you analyze CI failures from Buildkite for both AMD and NVIDIA platforms, create comprehensive failure reports, and provide reproduction steps.

## Integration with CI Log Downloader

This skill works together with the `vllm-ci-log-downloader` skill:
1. **First**: Use `vllm-ci-log-downloader` to download logs to `.claude/ci-logs/<date>-<platform>-build-<number>/`
2. **Then**: Use this skill to analyze the logs and create reports in `.claude/ci-analysis/<date>-<platform>/`

## Input: Where to Find Logs

**Log Location**: `.claude/ci-logs/`

The CI logs are organized by date and platform:
```
.claude/ci-logs/
├── 2025-11-13-cuda-build-38778/
│   ├── build_38778_extended_pooling.log
│   ├── build_38778_mteb.log
│   ├── build_38778_multimodal_extended.log
│   └── summary.md
├── 2025-11-13-amd-build-38778/
│   └── ...
└── 2025-11-13-torch-nightly-build-38778/
    └── ...
```

**When analyzing**: Check this directory first for downloaded logs before asking the user for log files.

## Output Location

Analysis reports should be created in:
```
.claude/ci-analysis/<date>-<platform>/<test_failure>.md
```

**Directory Structure**:
- `<date>`: Build date in format YYYY-MM-DD (e.g., `2025-11-13`)
- `<platform>`: Platform name: `cuda`, `amd`, `torch-nightly`, `cpu`, etc.
- `<test_failure>.md`: Descriptive name of the failing test

**Examples**:
```
.claude/ci-analysis/
├── 2025-11-13-cuda/
│   ├── extended_pooling_failure.md
│   ├── mteb_failure.md
│   ├── multimodal_extended_failure.md
│   ├── nixlconnector_distributed_failure.md
│   └── h200_distributed_failure.md
├── 2025-11-13-amd/
│   ├── basic_correctness_failure.md
│   └── distributed_4gpu_failure.md
└── 2025-11-10-cuda/
    └── flash_attention_failure.md
```

**File Naming Convention**:
- Use snake_case for test names
- Be descriptive but concise (max 50 chars)
- Include test type if helpful (e.g., `distributed_`, `multimodal_`, `kernel_`)
- Examples:
  - `language_models_extended_pooling.md`
  - `distributed_nixlconnector_deepseek.md`
  - `flash_attention_h100.md`

## CI Configuration Files

**NVIDIA Tests**: `.buildkite/test-pipeline.yaml`
**AMD Tests**: `.buildkite/test-amd.yaml`

## Analysis Workflow

When investigating CI failures:

1. **Check for existing logs first**:
   - Look in `.claude/ci-logs/<date>-<platform>-build-<number>/` for downloaded logs
   - If user selects a date/build option (e.g., "yesterday", "latest", specific date):
     - Check if corresponding logs exist in `.claude/ci-logs/`
     - If NOT found, automatically invoke the `vllm-ci-log-downloader` skill to download them
     - Then proceed with analysis
2. **Identify the platform**: AMD (ROCm), NVIDIA (CUDA), Torch Nightly, CPU, etc.
3. **Locate the relevant Buildkite YAML file** and understand the test configuration
4. **Analyze the failure logs** to identify:
   - Test name and location
   - Error messages and stack traces
   - Platform-specific code paths
   - Whether it's a v0 issue (if so, deprioritize)
5. **Create a detailed analysis report** in `.claude/ci-analysis/<date>-<platform>/<test_failure>.md`
6. **Provide reproduction steps** using pytest commands

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
2. **Check for downloaded logs first**:
   - Look in `.claude/ci-logs/` for logs organized by date and platform
   - **If user specifies a date/build option** (e.g., "yesterday", "latest", specific date):
     - Check if logs exist in `.claude/ci-logs/<date>-<platform>-build-<number>/`
     - If logs **DO NOT exist**, automatically invoke `vllm-ci-log-downloader` skill to download them first
     - Wait for download to complete, then proceed with analysis
   - Read the `summary.md` file to understand available failures
   - Use existing logs if available
3. **Identify the platform**: Is this ROCm (AMD), CUDA (NVIDIA), Torch Nightly, or other?
4. **Check for v0 code path**: If it's a v0 issue, note that it's not a priority
5. **Read the appropriate Buildkite YAML file** to understand test configuration
6. **Analyze failure logs** in detail:
   - Extract error messages and stack traces
   - Identify root cause
   - Determine if platform-specific or general issue
7. **Create analysis markdown** in `.claude/ci-analysis/<date>-<platform>/<test_failure>.md`:
   - Use descriptive filename (e.g., `extended_pooling_failure.md`)
   - Follow the report structure (Summary, Findings, Reproduction, CI Config)
   - Include relevant code snippets
8. **IMPORTANT**: Only create the markdown file on **initial analysis**
   - For subsequent iterations or follow-up questions, provide information directly
   - Do not recreate or overwrite the analysis file
9. **Provide pytest reproduction commands** using the vllm-test skill format:
   ```bash
   PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest \
     'tests/path/to/test.py::test_function' \
     -v -s
   ```

## Directory Organization

**Input** (logs downloaded by vllm-ci-log-downloader):
```
.claude/ci-logs/<date>-<platform>-build-<number>/
├── <test_name>.log
├── <test_name>.log
└── summary.md
```

**Output** (analysis reports created by this skill):
```
.claude/ci-analysis/<date>-<platform>/
├── <test_failure_1>.md
├── <test_failure_2>.md
└── <test_failure_3>.md
```

**Example Flow**:
```
1. Download logs:
   .claude/ci-logs/2025-11-13-cuda-build-38778/
   ├── build_38778_extended_pooling.log
   ├── build_38778_mteb.log
   └── summary.md

2. Analyze and create reports:
   .claude/ci-analysis/2025-11-13-cuda/
   ├── extended_pooling_failure.md
   └── mteb_failure.md
```

## Example Workflow

```bash
# 1. User: "Analyze the Extended Pooling failure from build 38778"

# 2. Claude checks for existing logs:
ls .claude/ci-logs/2025-11-13-cuda-build-38778/
# Found: build_38778_extended_pooling.log

# 3. Claude analyzes:
# - Reads .buildkite/test-pipeline.yaml
# - Analyzes build_38778_extended_pooling.log
# - Examines tests/models/test_pooling.py (or relevant test file)
# - Checks if it's platform-specific
# - Uses ultrathink for deep analysis

# 4. Claude creates analysis report:
.claude/ci-analysis/2025-11-13-cuda/extended_pooling_failure.md

# 5. Claude provides reproduction command:
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest \
  'tests/models/test_pooling.py::test_extended_pooling' \
  -v -s
```

## Complete Workflow Example (Download + Analyze)

```bash
# Step 1: Download logs (using vllm-ci-log-downloader skill)
User: "Download CI logs from build 38778"
→ Creates: .claude/ci-logs/2025-11-13-cuda-build-38778/

# Step 2: Analyze failures (using this skill)
User: "Analyze the Extended Pooling and MTEB failures"
→ Creates:
  .claude/ci-analysis/2025-11-13-cuda/extended_pooling_failure.md
  .claude/ci-analysis/2025-11-13-cuda/mteb_failure.md

# Step 3: Follow-up questions (no new files created)
User: "What's the root cause of the Extended Pooling failure?"
→ Claude answers based on existing analysis, doesn't recreate file
```

## Notes

- Always check both AMD and NVIDIA YAML files if the issue might affect both platforms
- Include relevant code snippets in the analysis
- Link to related issues or PRs if available
- Mention if the failure is intermittent or consistent
- Consider numerical precision differences between platforms
- Check for attention backend differences (FlashAttention vs Triton)
