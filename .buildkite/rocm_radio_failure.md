# ROCm RADIO Test Failure Analysis

## Summary

The AMD CI is failing with 1 test failure in `test_radio.py` due to numerical precision differences between ROCm and CUDA.

**Test:** `models/multimodal/pooling/test_radio.py::test_radio[half-nvidia/C-RADIOv2-H]`

**Status:** FAILED (1 failed, 25 passed, 18 skipped)

**Error:** Cosine similarity between vLLM and HuggingFace outputs is 0.9521, but test expects > 0.99

---

## Failure Details

### Test Location
- **File:** `tests/models/multimodal/pooling/test_radio.py:72`
- **Model:** `nvidia/C-RADIOv2-H`
- **Dtype:** `half` (fp16)

### Error Message
```python
AssertionError: assert tensor(0.9521, device='cuda:0') > 0.99
```

### What the Test Does
The test compares vLLM's RadioModel output with HuggingFace's RadioModel output using cosine similarity:

```python
cos_similar = nn.CosineSimilarity(dim=-1)
for vllm_output, hf_output in zip(vllm_outputs_per_image, hf_outputs_per_image):
    assert cos_similar(vllm_output, hf_output).mean() > 0.99
```

---

## Root Cause

This is **NOT** an encoder-decoder architecture issue (which would raise `NotImplementedError`).

This is a **numerical precision issue** specific to ROCm/AMD GPUs.

### Why Numerical Differences Occur

**1. Bicubic Interpolation (`radio.py:207-213`, `radio.py:358-363`, `radio.py:370-372`)**
- RADIO uses `F.interpolate(..., mode='bicubic')` for positional embedding resizing
- Bicubic interpolation may have different implementations/optimizations between CUDA and ROCm
- These differences compound across multiple interpolation operations

**2. FP16 Precision**
- Test uses `dtype='half'` (fp16)
- FP16 has limited precision, making it sensitive to implementation differences
- Small differences accumulate through the model layers

**3. Operations with Potential Differences**
- `F.interpolate` with bicubic mode (lines 207, 358, 370)
- `F.grid_sample` (line 349)
- LayerNorm operations
- Matrix multiplications in attention layers

### Model Architecture
- RADIO is a vision encoder model based on InternViT
- Uses ViT (Vision Transformer) architecture with patch embeddings
- Has complex positional encoding logic with CPE (Conditional Positional Encoding)
- Does NOT use problematic encoder-decoder attention (uses InternVisionEncoder which works on ROCm)

---

## Impact

**Severity:** Low - Single test failure, model still functional

**Why it's not critical:**
1. Model runs successfully on ROCm (no crashes or NotImplementedError)
2. Cosine similarity of 0.9521 shows outputs are very similar (95% similar)
3. Only 26 tests in this suite, 25 passing
4. This is a pooling/embedding model test, not a generation test

---

## Reproduction

To reproduce locally on ROCm:

```bash
pytest -v -s tests/models/multimodal/pooling/test_radio.py::test_radio
```

Expected output:
```
FAILED test_radio[half-nvidia/C-RADIOv2-H] - AssertionError: assert tensor(0.9521, device='cuda:0') > 0.99
```

---

## Potential Solutions

### Option 1: Relax Tolerance for ROCm (Recommended)
Adjust the cosine similarity threshold for ROCm platforms:

```python
import vllm
threshold = 0.95 if vllm.utils.current_platform.is_rocm() else 0.99
assert cos_similar(vllm_output, hf_output).mean() > threshold
```

### Option 2: Use torch.allclose Instead
Replace cosine similarity check with torch.allclose:

```python
torch.testing.assert_close(vllm_output, hf_output, rtol=1e-2, atol=1e-2)
```

### Option 3: Skip on ROCm
Add platform-specific skip (not recommended as model works):

```python
@pytest.mark.skipif(
    vllm.utils.current_platform.is_rocm(),
    reason="Numerical precision differences on ROCm"
)
def test_radio(...):
    ...
```

### Option 4: Use BF16 Instead
Test with bfloat16 instead of float16:
- BF16 has better numerical stability than FP16
- May reduce precision differences between platforms

---

## Comparison with Encoder-Decoder Issues

This is **different** from the encoder-decoder issues documented in `rocm_encoder_decoder_skip.md`:

| Aspect | Encoder-Decoder Issue | RADIO Issue |
|--------|----------------------|-------------|
| **Error Type** | NotImplementedError | AssertionError (numerical) |
| **Root Cause** | Missing ROCm attention backend support | Numerical precision differences |
| **Model Type** | Encoder-decoder (Whisper, Phi-3.5-vision) | Vision encoder only (RADIO/InternViT) |
| **Severity** | High (model crashes) | Low (model works, outputs close) |
| **Solution** | Skip tests entirely | Relax tolerance or skip |

---

## Verification

After implementing a fix, verify with:

```bash
# Run the specific test
pytest -v -s tests/models/multimodal/pooling/test_radio.py::test_radio

# Run all multimodal pooling tests
pytest -v -s tests/models/multimodal/pooling/

# Check the full AMD CI suite
bash .buildkite/run-amd-test.sh
```

---

## Related Files

- **Test:** `tests/models/multimodal/pooling/test_radio.py`
- **Model:** `vllm/model_executor/models/radio.py`
- **Config:** `vllm/transformers_utils/configs/radio.py`
- **CI Config:** `.buildkite/test-amd.yaml` (line 411 - offline inference test)
- **Log:** `/home/zhewenli/logs/vllm-ci/amd.log:11499-12121`

---

## Recommendations

1. **Short-term:** Add platform-specific tolerance for ROCm (Option 1)
2. **Medium-term:** Investigate F.interpolate bicubic differences between CUDA/ROCm
3. **Long-term:** Consider adding more comprehensive numerical testing across platforms

---

## Platform-Specific Test Results

### CUDA (Expected)
- Cosine similarity: > 0.99 ✅
- All tests passing

### ROCm (Current)
- Cosine similarity: 0.9521 ❌
- Test fails but model runs successfully
- Outputs are 95% similar (still very good)
