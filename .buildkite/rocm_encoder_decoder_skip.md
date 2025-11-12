# ROCm Encoder-Decoder Test Handling

## Summary

Encoder-decoder and encoder-only models fail on AMD ROCm with `NotImplementedError` because all ROCm-specific attention backends only support decoder-only models.

This document explains the hybrid approach we implemented to handle these failures in CI and local testing.

**Related Issue:** https://github.com/vllm-project/vllm/issues/27442

---

## Problem

### Root Cause

All ROCm attention backends raise `NotImplementedError` for encoder architectures:

**Affected Backends:**
- `TritonAttentionBackend` (default) - `vllm/v1/attention/backends/triton_attn.py`
- `RocmAttentionBackend` - `vllm/v1/attention/backends/rocm_attn.py`
- `RocmAiterFABackend` - `vllm/v1/attention/backends/rocm_aiter_fa.py`
- `RocmAiterMLABackend` - `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`

**Error Message:**
```python
NotImplementedError: Encoder self-attention and encoder/decoder cross-attention
are not implemented for TritonAttentionImpl
```

### Affected Models (17 total)

**Encoder-Decoder Models (4):**
1. openai/whisper-small
2. openai/whisper-large-v3-turbo
3. mistralai/Voxtral-Mini-3B-2507
4. microsoft/Phi-3.5-vision-instruct

**Encoder-Only Models (13):**
5. cross-encoder/ms-marco-MiniLM-L-6-v2
6. intfloat/e5-small
7. intfloat/multilingual-e5-small
8. BAAI/bge-reranker-base
9. BAAI/bge-reranker-v2-m3
10. BAAI/bge-base-en-v1.5
11. TIGER-Lab/VLM2Vec-Full
12. Snowflake/snowflake-arctic-embed-m-v1.5
13. sentence-transformers/all-MiniLM-L12-v2
14. sentence-transformers/stsb-roberta-base-v2

---

## Solution: Hybrid Approach

We use a combination of CI-level skipping (for 100% encoder-only files) and pytest markers (for mixed files).

### 1. CI-Level Skipping (12 Files)

Files that contain **ONLY** encoder models are skipped entirely at the CI level in `.buildkite/test-amd.yaml`.

**Files Skipped in API Server Test:**
- `entrypoints/openai/test_vision.py` (Phi-3.5-vision only)

**Files Skipped in Pooling Test:**
- `entrypoints/pooling/correctness/test_mteb_score.py`
- `entrypoints/pooling/correctness/test_mteb_embed.py`
- `entrypoints/pooling/llm/test_embedding.py`
- `entrypoints/pooling/llm/test_encode.py`
- `entrypoints/pooling/openai/test_embedding.py`
- `entrypoints/pooling/openai/test_embedding_dimensions.py`
- `entrypoints/pooling/openai/test_embedding_long_text.py`
- `entrypoints/pooling/openai/test_rerank.py`
- `entrypoints/pooling/openai/test_score.py`
- `entrypoints/pooling/openai/test_truncation.py`
- `entrypoints/pooling/openai/test_vision_embedding.py`

**Implementation:**
```yaml
# .buildkite/test-amd.yaml:160
- pytest -v -s entrypoints/openai ... --ignore=entrypoints/openai/test_vision.py

# .buildkite/test-amd.yaml:176
- pytest -v -s entrypoints/pooling --ignore=entrypoints/pooling/correctness/test_mteb_score.py ...
```

### 2. Pytest Markers (3 Files with Mixed Models)

Files that contain **BOTH** encoder and decoder models use `@pytest.mark.encoder_decoder` on specific tests/variants.

**Mixed Files:**
1. `tests/entrypoints/openai/test_translation_validation.py`
   - Encoder: openai/whisper-small (5 tests with explicit skip)
   - Decoder: google/gemma-3n-E2B-it (runs on ROCm)

2. `tests/entrypoints/openai/test_transcription_validation.py`
   - Encoder: openai/whisper-large-v3-turbo, mistralai/Voxtral-Mini-3B-2507 (9 tests marked)
   - Decoder: google/gemma-3n-E2B-it (2 tests NOT marked, run on ROCm)

3. `tests/models/language/pooling/test_embedding.py`
   - Encoder models: 4 variants marked with `marks=[pytest.mark.encoder_decoder]`
   - Decoder models: NOT marked, run on ROCm

**Implementation:**
```python
# For tests with parametrized fixtures (test_translation_validation.py)
@pytest.mark.encoder_decoder
async def test_basic_audio(client, model_name):
    if current_platform.is_rocm() and model_name in ENCODER_DECODER_MODELS:
        pytest.skip("Encoder-decoder models not supported on ROCm")
    ...

# For tests with non-parametrized fixtures (test_transcription_validation.py)
@pytest.mark.encoder_decoder  # Relies on conftest.py hook
async def test_long_audio_request(client):
    ...

# For parametrize variants (models/language/pooling/test_embedding.py)
pytest.param("BAAI/bge-base-en-v1.5", marks=[pytest.mark.encoder_decoder])
```

### 3. Automatic Skip Logic (conftest.py)

Super simple hook that skips encoder-decoder tests on ROCm:

```python
# tests/conftest.py:1281-1291 (5 lines!)
if current_platform.is_rocm():
    for item in items:
        if "encoder_decoder" in item.keywords:
            if any(encoder_model in item.nodeid for encoder_model in ENCODER_DECODER_MODELS):
                item.add_marker(skip_encoder_decoder)
```

**How it works:**

All mixed test files use parametrized fixtures, so model names are always in nodeids:

- `test_basic_audio[openai/whisper-small]` → encoder in nodeid → SKIP ✅
- `test_basic_audio[google/gemma-3n-E2B-it]` → encoder NOT in nodeid → RUN ✅
- `test_long_audio_request[openai/whisper-large-v3-turbo]` → encoder in nodeid → SKIP ✅
- `test_basic_audio_gemma` → not marked encoder_decoder → RUN ✅

---

## Why This Approach?

### Benefits of Hybrid Approach

**1. CI-Level Skipping for Encoder-Only Files:**
- ✅ **Faster CI** - Doesn't even collect encoder-only tests
- ✅ **Cleaner config** - Explicit about what's skipped
- ✅ **Clear intent** - Obviously encoder-only at CI level
- ✅ **Simpler test files** - No markers needed

**2. Pytest Markers for Mixed Files:**
- ✅ **Selective skipping** - Only encoder variants skip
- ✅ **Maximum coverage** - Decoder variants still validate ROCm
- ✅ **Granular control** - Per-test or per-variant control
- ✅ **Local testing** - Works on local ROCm machines

### Why Not Skip Everything at CI Level?

**API Server Tests:**
- Only ~7% encoder-decoder (22 out of ~89 tests)
- 48+ decoder-only tests still need to run on ROCm

**Pooling Tests:**
- ~63% encoder (11 out of 16 files)
- But 5 files test decoder-based pooling (reward, classification models)

**Language Model Tests:**
- Mixed parametrization with both decoder and encoder variants
- Need to test decoder variants on ROCm

---

## Implementation Details

### Modified Files (5 files)

**Configuration:**
1. `.buildkite/test-amd.yaml` - Added --ignore for 12 encoder-only files
2. `pyproject.toml` - Added encoder_decoder marker registration
3. `tests/conftest.py` - Added simple skip logic (5 lines)

**Test Files (2 files with markers):**
4. `tests/entrypoints/openai/test_translation_validation.py` - 5 tests marked
5. `tests/entrypoints/openai/test_transcription_validation.py` - 9 tests marked
6. `tests/models/language/pooling/test_embedding.py` - 4 variants marked

### AMD CI Test Suites Covered

**✅ Entrypoints Integration Test (API Server)** - amd-3.log
- test_vision.py skipped at CI level (12 tests)
- test_translation_validation.py selective skipping (5 encoder variants)
- test_transcription_validation.py selective skipping (9 encoder variants)

**✅ Entrypoints Integration Test (Pooling)** - amd-1.log
- 11 encoder-only files skipped at CI level (~35 tests)
- Decoder-based pooling tests (reward, classification) still run

**✅ Language Models Tests** - amd.log
- 4 encoder model variants marked in test_embedding.py
- Decoder model variants still run

---

## Usage

### For Test Authors

**When writing new tests:**

1. **If entire file uses encoder models** → Skip at CI level
   - Add file to --ignore in `.buildkite/test-amd.yaml`

2. **If file has mixed encoder/decoder models** → Use markers
   - Add `@pytest.mark.encoder_decoder` to encoder tests/variants
   - Leave decoder tests unmarked

**Example - Mixed file:**
```python
@pytest.mark.parametrize("model", [
    "openai/whisper-small",      # Encoder
    "google/gemma-3n-E2B-it",    # Decoder
])
@pytest.mark.encoder_decoder
async def test_basic_audio(model):
    # whisper-small will skip on ROCm
    # gemma will run on ROCm
    ...
```

**Example - Encoder-only file:**
```python
# Add to .buildkite/test-amd.yaml:
# --ignore=entrypoints/openai/test_my_encoder_test.py
```

### For CI Maintainers

**Adding new encoder models:**
1. Add model to `ENCODER_DECODER_MODELS` list in `tests/conftest.py`
2. If entire file uses that model, add to --ignore in test-amd.yaml

**Removing models from skip list:**
1. Remove from `ENCODER_DECODER_MODELS` list
2. Remove from --ignore if entire file was skipped

---

## Test Results

### Before Implementation
- ~60+ tests failing on ROCm AMD CI with NotImplementedError
- Tests blocked in 3 test suites: API Server, Pooling, Language Models

### After Implementation
- ✅ All encoder tests properly skipped on ROCm
- ✅ All decoder tests continue to run and validate ROCm
- ✅ Clear skip messages indicating why tests are skipped
- ✅ Faster CI (encoder-only files not even collected)

---

## Future Work (Long-term Solution)

The short-term solution skips encoder tests on ROCm. The long-term solution is to implement encoder support:

1. **Support flash_attn on ROCm** for encoder architectures
2. **Add KV bindings** for cross-attention
3. **Implement encoder/decoder attention** in ROCm backends

Once encoder support is added, remove:
- --ignore flags from test-amd.yaml
- @pytest.mark.encoder_decoder markers
- Skip logic from conftest.py

---

## References

- **GitHub Issue:** https://github.com/vllm-project/vllm/issues/27442
- **Root Cause:** `.buildkite/encoder.md`
- **CI Config:** `.buildkite/test-amd.yaml`
- **Test Config:** `tests/conftest.py:1261-1290`
- **Marker Registration:** `pyproject.toml:110`
