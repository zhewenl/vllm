# Gemma3 Test Failure Analysis

## Summary

The `gemma3` tests in `tests/models/multimodal/generation/test_common.py` are failing with placeholder detection issues. Despite being an NVIDIA test run, these failures appear to be platform-independent issues with the custom Gemma3-MM implementation.

## Failing Tests

**File**: `/home/zhewenli/logs/vllm-ci/amd.log` (actually NVIDIA L4 GPU with CUDA 12.9)

1. `test_single_image_models[gemma3-test_case94]`
   - **Error**: `RuntimeError: Expected there to be 1 prompt placeholders corresponding to 1 image items, but instead found 0 prompt placeholders!`

2. `test_multi_image_models[gemma3-test_case80]`
   - **Error**: `RuntimeError: Expected there to be 2 prompt placeholders corresponding to 2 image items, but instead found 1 prompt placeholders!`

3. `test_custom_inputs_models[llava_onevision-multiple-images-test_case5]`
   - **Error**: `AssertionError` on token matching

## History

### Timeline of Changes

1. **PR #26715** (commit 8c017b349 - Oct 2025)
   - Moved PaliGemma and Gemma3-MM to always use Transformers backend
   - Removed custom "gemma3" implementation
   - Only kept "gemma3-transformers" (marked as `core_model`)

2. **PR #27309** (commit e05a6754a - Oct 22, 2025)
   - **REVERTED PR #26715**
   - Restored custom PaliGemma and Gemma3-MM implementations
   - Brought back "gemma3" test entry (WITHOUT `core_model` mark)
   - Kept "gemma3-transformers" alongside it

3. **PR #27538** (commit 23ad82055 - Oct 27, 2025)
   - Fixed mm placeholder replacement issue in `vllm/model_executor/models/gemma3_mm.py:404`
   - Changed newline token order from `[newline_1, newline_2]` to `[newline_2, newline_1]`

## Root Cause

### Placeholder Detection Issue

The error occurs in `vllm/multimodal/processing.py:2012-2019` during validation that checks if the number of found placeholders matches the number of multimodal items.

**Expected behavior:**
- Prompt: `<bos><start_of_turn>user\n<start_of_image>What's the content in the center of the image?<end_of_turn>\n<start_of_turn>model\n`
- Should find 1 `<start_of_image>` placeholder

**Actual behavior:**
- Found 0 placeholders for single image test
- Found 1 placeholder for multi-image test (expected 2)

### Why This Happens

The `_get_prompt_updates` method in `gemma3_mm.py:326` returns a `PromptReplacement` targeting `hf_processor.boi_token` (which should be `<start_of_image>`). However, the HF processor is not finding these tokens in the prompt during `_call_hf_processor`.

## Test Configuration

**File**: `tests/models/multimodal/generation/test_common.py:372`

```python
"gemma3": VLMTestInfo(
    models=["google/gemma-3-4b-it"],
    test_type=(VLMTestType.IMAGE, VLMTestType.MULTI_IMAGE),
    prompt_formatter=lambda img_prompt: f"<bos><start_of_turn>user\n{img_prompt}<end_of_turn>\n<start_of_turn>model\n",
    single_image_prompts=IMAGE_ASSETS.prompts({
        "stop_sign": "<start_of_image>What's the content in the center of the image?",
        "cherry_blossom": "<start_of_image>What is the season?",
    }),
    multi_image_prompt="<start_of_image><start_of_image>Describe the two images in detail.",
    max_model_len=4096,
    max_num_seqs=2,
    auto_cls=AutoModelForImageTextToText,
    vllm_runner_kwargs={"mm_processor_kwargs": {"do_pan_and_scan": True}},
    patch_hf_runner=model_utils.gemma3_patch_hf_runner,
    num_logprobs=10,
),
```

**Note**: NO `marks=` field, so NOT a `core_model` test

## CI Test Selection

- **NVIDIA CI**: Runs `core_model` tests → runs `gemma3-transformers` (passes)
- **AMD CI**: Runs `-m 'not core_model'` tests → runs `gemma3` custom impl (FAILS)

From `.buildkite/test-amd.yaml`:
```bash
pytest -v -s models/multimodal/generation/test_common.py -m 'split(group=0) and not core_model'
```

## Why Tests Never Pass

The `gemma3` custom implementation tests were **restored broken** in PR #27309 and the subsequent fix in PR #27538 was **insufficient**. The placeholder detection logic has a fundamental issue that wasn't resolved by just swapping newline token order.

## Comparison: gemma3 vs gemma3-transformers

| Feature | gemma3 | gemma3-transformers |
|---------|--------|---------------------|
| Backend | Custom implementation | Transformers backend |
| Mark | None | `pytest.mark.core_model` |
| Test Type | IMAGE, MULTI_IMAGE | IMAGE only |
| Runs in AMD CI | Yes (FAILS) | No |
| Runs in NVIDIA CI | No | Yes (PASSES) |
| do_pan_and_scan | Yes | No |

## Recommendation

**Option 1**: Skip `gemma3` tests until the placeholder detection is fixed
- Add `marks=[pytest.mark.skip(reason="Placeholder detection broken - see #27538")]`

**Option 2**: Use transformers backend only
- Remove "gemma3" entry, keep only "gemma3-transformers"
- Similar to what PR #26715 did before being reverted

**Option 3**: Fix the placeholder detection properly
- Debug why `hf_processor.boi_token` is not being found in prompts
- Verify HF processor behavior matches expectations
- May require coordination with transformers library changes

## Reproduction

```bash
pytest tests/models/multimodal/generation/test_common.py::test_single_image_models -k "gemma3-test_case94" -v
```

Expected error:
```
RuntimeError: Expected there to be 1 prompt placeholders corresponding to 1 image items,
but instead found 0 prompt placeholders! Make sure the implementation of `_call_hf_processor`
and `_get_mm_fields_config` are consistent with each other.
```

## Related Files

- `vllm/model_executor/models/gemma3_mm.py` - Custom Gemma3 implementation
- `tests/models/multimodal/generation/test_common.py:372` - Test configuration
- `tests/models/multimodal/generation/vlm_utils/model_utils.py` - Helper functions
- `vllm/multimodal/processing.py:2012-2019` - Validation code that raises error
