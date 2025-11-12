---
name: vllm-test
description: Run vLLM tests with proper environment setup (uv venv, proxy, device detection)
---

# vLLM Testing Skill

This skill helps you run vLLM tests with the correct environment configuration for Facebook's infrastructure.

## Environment Setup

**CRITICAL**: Always use the uv-managed virtual environment at `/home/zhewenli/uv_env/vllm-fork/`

1. **Virtual Environment**: The uv venv contains all vLLM dependencies and must be used for all test runs
2. **Proxy**: Use `HF_HUB_DISABLE_XET=1 wp` prefix for commands that need HuggingFace access
3. **Device Detection**: Check available GPUs with `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA)

## Common Commands

### Check Available Devices

For AMD GPUs:
```bash
rocm-smi
```

For NVIDIA GPUs:
```bash
nvidia-smi
```

### Run Bash Scripts with Proxy

For scripts that require network access (lm-eval, model downloads, etc.):
```bash
HF_HUB_DISABLE_XET=1 with-proxy bash ./script.sh [args]
```

Example:
```bash
HF_HUB_DISABLE_XET=1 with-proxy bash ./run-lm-eval-gsm-vllm-baseline.sh \
  -m deepseek-ai/DeepSeek-V2-Lite-Chat \
  -b "auto" \
  -l 1000 \
  -f 5 \
  -t 2
```

### Run Tests

**PREFERRED METHOD** (with PATH modification):
```bash
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest [test_path] [options]
```

**Alternative METHOD** (direct Python path):
```bash
HF_HUB_DISABLE_XET=1 /home/zhewenli/uv_env/vllm-fork/bin/python -m pytest [test_path] [options]
```

Examples:
```bash
# Run specific test (PREFERRED)
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest 'tests/tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]' -v -s

# Run test file
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest tests/tool_use/test_tool_calls.py -v -s

# Run with specific marker
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest -v -s -m core_model tests/

# Alternative with direct python path
HF_HUB_DISABLE_XET=1 /home/zhewenli/uv_env/vllm-fork/bin/python -m pytest -v -s 'tests/tool_use/test_tool_calls.py::test_function'
```

## Instructions

When the user invokes this skill:

1. **First**, check which GPU platform is available by running both `rocm-smi` and `nvidia-smi`
2. **Then**, if running tests:
   - **ALWAYS** use the uv venv by either:
     - Modifying PATH: `PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH` (PREFERRED)
     - Using full Python path: `/home/zhewenli/uv_env/vllm-fork/bin/python`
   - Always prefix with `HF_HUB_DISABLE_XET=1` for HuggingFace model downloads
   - Use `wp` (with-proxy) for network access when using PATH modification
   - Add `-v -s` for verbose output
   - Optionally add `--log-cli-level=WARNING` for cleaner output
   - Capture output to a log file using `tee` for later analysis

3. **Report** the platform (ROCm/CUDA), available devices, and test results

## Example Workflow

```bash
# 1. Check devices
echo "Checking AMD GPUs..."
rocm-smi 2>/dev/null || echo "No AMD GPUs found"
echo "Checking NVIDIA GPUs..."
nvidia-smi 2>/dev/null || echo "No NVIDIA GPUs found"

# 2. Run test with full setup (PREFERRED METHOD)
PATH=/home/zhewenli/uv_env/vllm-fork/bin:$PATH HF_HUB_DISABLE_XET=1 wp pytest \
  'tests/tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]' \
  -v -s \
  --log-cli-level=WARNING \
  2>&1 | tee /tmp/test_output.log

# Alternative: Run test with direct python path
HF_HUB_DISABLE_XET=1 /home/zhewenli/uv_env/vllm-fork/bin/python -m pytest \
  'tests/tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]' \
  -v -s \
  --log-cli-level=WARNING \
  2>&1 | tee /tmp/test_output.log
```

## Notes

- ROCm platform uses Triton attention backend
- NVIDIA platform may use FlashAttention
- Numerical differences between platforms can cause test failures
- Always capture logs for debugging AMD-specific issues
