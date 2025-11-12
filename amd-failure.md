# ROCm Granite-3.0-8b Tool Use Test Failure Analysis

## Summary
The OpenAI-Compatible Tool Use test `test_tool_call_and_choice[granite-3.0-8b]` is failing on AMD/ROCm but likely passing on NVIDIA.

## Test Failure Details

**Failing Test:** `tests/tool_use/test_tool_calls.py::test_tool_call_and_choice[granite-3.0-8b]`

**Error:**
```python
assert len(tool_calls) == 1
E  assert 0 == 1
E   +  where 0 = len([])

tool_use/test_tool_calls.py:39: AssertionError
```

**Location:** `/data/users/zhewenli/gitrepos/vllm-fork/tests/tool_use/test_tool_calls.py:39`

## What the Test Does

The test (`test_tool_call_and_choice`) performs the following:
1. Sends a request asking for weather in Dallas, Texas
2. Expects the model to return a tool call to `get_current_weather` function
3. Expects exactly 1 tool call with specific arguments: `{"city": "Dallas", "state": "TX", "unit": "fahrenheit"}`
4. Verifies both streaming and non-streaming responses match

**Test Input:**
```python
MESSAGES_ASKING_FOR_TOOLS = [
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"}
]
```

**Available Tools:**
- `get_current_weather` (WEATHER_TOOL)
- `web_search` (SEARCH_TOOL)

## Interesting Observations

### Passing Tests
The following granite-3.0-8b tests PASS on the same AMD CI run:
- `test_chat_completion_without_tools[granite-3.0-8b]` ✓ PASSED
- `test_chat_completion_with_tools[granite-3.0-8b]` ✓ PASSED
- `test_parallel_tool_calls[granite-3.0-8b]` ✓ PASSED
- `test_parallel_tool_calls_with_results[granite-3.0-8b]` ✓ PASSED
- `test_tool_call_with_results[granite-3.0-8b]` ✓ PASSED

### Only Failing Test
- `test_tool_call_and_choice[granite-3.0-8b]` ✗ FAILED

## Model Configuration

**Model:** `ibm-granite/granite-3.0-8b-instruct`
**Tool Parser:** `granite` (GraniteToolParser)
**Chat Template:** `/vllm-workspace/examples/tool_chat_template_granite.jinja`
**Server Args:**
```
--enable-auto-tool-choice
--max-model-len 1024
--max-num-seqs 256
--enforce-eager
--no-enable-prefix-caching
--tool-call-parser granite
--chat-template /vllm-workspace/examples/tool_chat_template_granite.jinja
--seed 0
```

**AMD-Specific Configuration:**
- Using Triton Attention backend (log: "Using Triton Attention backend")
- ROCm device

## Granite Tool Parser Logic

The GraniteToolParser (vllm/entrypoints/openai/tool_parsers/granite_tool_parser.py) expects:
1. Model output starting with `<|tool_call|>` (granite 3.0) or `<tool_call>` (granite 3.1)
2. Followed by JSON array: `[{"name": "func_name", "arguments": {...}}]`

Example expected output:
```
<|tool_call|>[{"name": "get_current_weather", "arguments": {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}}]
```

The parser returns `tools_called=False` and empty tool_calls list if:
- The output doesn't start with the expected prefix
- The output doesn't start with `[` after stripping prefix
- JSON parsing fails

## Chat Template Format

From `examples/tool_chat_template_granite.jinja`:
- Tools are provided at the beginning: `<|start_of_role|>available_tools<|end_of_role|>\n{tools as JSON}\n<|end_of_text|>`
- User message: `<|start_of_role|>user<|end_of_role|>What is the weather in Dallas, Texas in Fahrenheit?<|end_of_text|>`
- Expected assistant response: `<|start_of_role|>assistant<|end_of_role|><|tool_call|>[{...}]`

If the model generates text instead (e.g., "I'll help you check the weather..."), the parser will correctly return `tools_called=False` with empty tool_calls list, but the test expects exactly 1 tool call.

## Potential Root Causes

### 1. Numerical Accuracy Issue (Most Likely)
- AMD/ROCm using Triton Attention backend vs NVIDIA using different backend
- With `seed=0`, generation should be deterministic, but numerical differences in attention computation could cause different token selections
- The model might be generating slightly different logits, causing it to generate text content instead of tool calls

### 2. Attention Backend Difference
- AMD log shows: "Using Triton Attention backend"
- NVIDIA might be using FlashAttention or other optimized backend
- Numerical differences in attention computation could affect generation

### 3. Test Flakiness
- The test might be flaky and sensitive to minor numerical variations
- Other tool use tests pass, suggesting the infrastructure works
- Only this specific test with this specific prompt fails

## Differences from Passing Tests

The `test_chat_completion_with_tools` test also tests tool calls and PASSES. The key difference is:
- `test_chat_completion_with_tools`: simpler validation, just checks that tool calls are returned
- `test_tool_call_and_choice`: stricter validation including:
  - Exact argument matching
  - Streaming vs non-streaming consistency
  - Specific tool call format validation

This suggests the issue is related to generation quality/consistency rather than infrastructure.

## Reproduction Steps

To reproduce locally on AMD:
```bash
pytest -v -s tests/tool_use/test_tool_calls.py::test_tool_call_and_choice \
  -k "granite-3.0-8b"
```

To debug what the model actually generates:
```python
# Add logging to granite_tool_parser.py extract_tool_calls method
logger.info(f"Raw model output: {model_output}")
logger.info(f"After stripping: {stripped}")
```

## Recommended Next Steps

1. **Compare with NVIDIA**: Check if this test passes on NVIDIA to confirm it's AMD-specific

2. **Capture Model Output**: Add debug logging to see what the model actually generates on AMD:
   ```python
   # In vllm/entrypoints/openai/tool_parsers/granite_tool_parser.py
   logger.warning(f"[DEBUG] model_output: {repr(model_output)}")
   ```

3. **Check Attention Backend**: Verify if forcing the same attention backend on both platforms fixes the issue

4. **Test with Different Seeds**: Try different seeds to see if this is a specific seed issue

5. **Tolerance Test**: Consider if this test should allow for minor generation differences between platforms, or if strict determinism is required

## Platform Information

- **Device**: AMD GPU (ROCm)
- **Attention Backend**: Triton
- **vLLM Version**: 0.11.1rc7.dev12+gbca74e32b
- **Eager Mode**: Enabled (--enforce-eager)
- **CUDA Graph**: Disabled in eager mode

## Related Files

- Test: `tests/tool_use/test_tool_calls.py`
- Parser: `vllm/entrypoints/openai/tool_parsers/granite_tool_parser.py`
- Template: `examples/tool_chat_template_granite.jinja`
- CI Config: `.buildkite/test-amd.yaml`

## Conclusion

This is likely a **numerical accuracy/generation consistency issue** specific to AMD/ROCm with Triton Attention backend. The model is generating different output (possibly text instead of tool calls) compared to NVIDIA, despite using the same seed. The infrastructure is working correctly (other tests pass), but the specific combination of model, prompt, and backend is producing different results.

**Action Required**: Need to capture actual model output on AMD to confirm what it's generating, and compare with NVIDIA output to understand the divergence.
