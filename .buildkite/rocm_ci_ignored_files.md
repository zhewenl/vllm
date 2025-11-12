# ROCm CI Ignored Files - Detailed Analysis

This document provides detailed analysis of all test files ignored in `.buildkite/test-amd.yaml` due to encoder/encoder-decoder model limitations on ROCm.

**Related Documentation:** `.buildkite/rocm_encoder_decoder_skip.md`
**GitHub Issue:** https://github.com/vllm-project/vllm/issues/27442

---

## Root Cause

All ROCm attention backends only support **decoder-only** models and raise `NotImplementedError` for:
- Encoder-decoder architectures (e.g., Whisper, T5, BART)
- Encoder-only architectures (e.g., BERT-based embedding models, cross-encoders)

**Error:**
```
NotImplementedError: Encoder self-attention and encoder/decoder cross-attention
are not implemented for TritonAttentionImpl
```

---

## Files Ignored at CI Level (13 Files)

### API Server Tests (2 Files)

#### 1. `test_vision.py`
- **Model:** `microsoft/Phi-3.5-vision-instruct`
- **Architecture:** Encoder-decoder (vision-language model)
- **Tests:** 11 tests covering chat, completions, streaming, multi-image
- **Why Encoder-Only:** All tests use Phi-3.5-vision-instruct exclusively
- **CI Location:** `.buildkite/test-amd.yaml:168`

#### 2. `test_optional_middleware.py`
- **Model:** `intfloat/multilingual-e5-small`
- **Architecture:** Encoder-only (BERT-based embedding model)
- **Tests:** 7 tests covering API keys, request IDs, middleware features
- **Why Encoder-Only:** Uses encoder model for "faster startup and smaller memory footprint" (not testing model functionality)
- **CI Location:** `.buildkite/test-amd.yaml:169`

---

### Pooling Tests (11 Files)

#### Correctness Tests (2 Files)

**3. `pooling/correctness/test_mteb_score.py`**
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Architecture:** Cross-encoder (encoder-only)
- **Tests:** 2 tests (`test_mteb_score`, `test_mteb_rerank`)
- **Purpose:** MTEB reranking benchmark validation
- **CI Location:** `.buildkite/test-amd.yaml:186`

**4. `pooling/correctness/test_mteb_embed.py`**
- **Model:** `intfloat/e5-small`
- **Architecture:** Encoder-only (embedding model)
- **Tests:** 1 test (`test_mteb_embed`)
- **Purpose:** MTEB embedding benchmark validation
- **CI Location:** `.buildkite/test-amd.yaml:187`

#### LLM API Tests (2 Files)

**5. `pooling/llm/test_embedding.py`**
- **Model:** `intfloat/multilingual-e5-small`
- **Architecture:** Encoder-only (embedding model)
- **Tests:** 3 tests covering LLM.embed() API with pooling parameters
- **Purpose:** Test vLLM's Python LLM API for embeddings
- **CI Location:** `.buildkite/test-amd.yaml:188`

**6. `pooling/llm/test_encode.py`**
- **Model:** `intfloat/multilingual-e5-small`
- **Architecture:** Encoder-only (embedding model)
- **Tests:** 2 tests covering LLM.encode() API with truncation
- **Purpose:** Test vLLM's Python LLM API for encoding
- **CI Location:** `.buildkite/test-amd.yaml:189`

#### OpenAI API Tests (7 Files)

**7. `pooling/openai/test_embedding.py`**
- **Model:** `intfloat/multilingual-e5-small`
- **Architecture:** Encoder-only (embedding model)
- **Tests:** ~15 tests covering OpenAI embeddings API
- **Features Tested:** Single/batch embeddings, base64 encoding, truncation, normalization, pooling
- **CI Location:** `.buildkite/test-amd.yaml:190`

**8. `pooling/openai/test_embedding_dimensions.py`**
- **Models:**
  - `intfloat/multilingual-e5-small` (standard embedding)
  - `Snowflake/snowflake-arctic-embed-m-v1.5` (matryoshka embedding)
- **Architecture:** Encoder-only (embedding models)
- **Tests:** 1 test (`test_matryoshka`) parametrized across 2 models
- **Purpose:** Test matryoshka embedding dimensions
- **CI Location:** `.buildkite/test-amd.yaml:191`

**9. `pooling/openai/test_embedding_long_text.py`**
- **Model:** `intfloat/multilingual-e5-small`
- **Architecture:** Encoder-only (embedding model)
- **Tests:** 4 tests covering automatic chunking for long text
- **Purpose:** Test automatic text chunking when exceeding max token length (512)
- **CI Location:** `.buildkite/test-amd.yaml:192`

**10. `pooling/openai/test_rerank.py`**
- **Model:** `BAAI/bge-reranker-base`
- **Architecture:** Cross-encoder (encoder-only)
- **Tests:** ~10 tests covering reranking API
- **Features Tested:** Rerank endpoint, top_n parameter, max_model_len, activation functions
- **CI Location:** `.buildkite/test-amd.yaml:193`

**11. `pooling/openai/test_score.py`**
- **Models:**
  - `BAAI/bge-reranker-v2-m3` (cross-encoder)
  - `BAAI/bge-base-en-v1.5` (encoder-only, used for cosine similarity scoring)
- **Architecture:** Both encoder-only
- **Tests:** ~10 tests covering score API with both models
- **Features Tested:** Text pair scoring, different input formats, max_model_len, activation
- **CI Location:** `.buildkite/test-amd.yaml:194`

**12. `pooling/openai/test_truncation.py`**
- **Model:** `sentence-transformers/all-MiniLM-L12-v2`
- **Architecture:** Encoder-only (sentence transformer)
- **Tests:** 4 tests covering truncation behavior
- **Purpose:** Test different truncation sizes (smaller, zero, bigger, max)
- **CI Location:** `.buildkite/test-amd.yaml:195`

**13. `pooling/openai/test_vision_embedding.py`**
- **Model:** `TIGER-Lab/VLM2Vec-Full`
- **Architecture:** Encoder (vision-language embedding model)
- **Tests:** ~4 tests covering vision embeddings with different image formats
- **Purpose:** Test image embedding generation
- **CI Location:** `.buildkite/test-amd.yaml:196`

---

## Summary by Model Type

### Encoder-Decoder Models (1 file)
- **microsoft/Phi-3.5-vision-instruct** → test_vision.py

### Cross-Encoders (2 files)
- **cross-encoder/ms-marco-MiniLM-L-6-v2** → test_mteb_score.py
- **BAAI/bge-reranker-base** → test_rerank.py
- **BAAI/bge-reranker-v2-m3** → test_score.py

### Sentence Transformers / Embedding Models (10 files)
- **intfloat/multilingual-e5-small** → test_optional_middleware.py, test_embedding.py (llm + openai), test_encode.py, test_embedding_long_text.py, test_embedding_dimensions.py
- **intfloat/e5-small** → test_mteb_embed.py
- **BAAI/bge-base-en-v1.5** → test_score.py
- **Snowflake/snowflake-arctic-embed-m-v1.5** → test_embedding_dimensions.py
- **sentence-transformers/all-MiniLM-L12-v2** → test_truncation.py
- **TIGER-Lab/VLM2Vec-Full** → test_vision_embedding.py

---

## Verification

All 13 files use **exclusively** encoder-only or encoder-decoder models. No decoder-only models are tested in these files, making them safe to skip entirely at the CI level.

### Test Coverage Impact

**Total tests skipped:** ~80+ tests
**Decoder-only tests still running:** ~150+ tests in other files

This maintains good test coverage for decoder-only models while cleanly handling encoder limitations.

---

## Alternative Approaches Considered

### Why Not Use Pytest Markers for These Files?

**Considered:** Add `pytestmark = pytest.mark.encoder_decoder` and rely on conftest.py hook

**Rejected because:**
1. CI still needs to collect/parse these files (slower)
2. Less explicit - not clear from CI config what's being skipped
3. More complex conftest.py logic needed for non-parametrized tests
4. No benefit since 100% of tests in these files use encoder models

### Why CI-Level --ignore is Better

**Benefits:**
- ✅ Faster CI - files never collected
- ✅ Explicit - CI config shows exactly what's skipped
- ✅ Simpler conftest.py - only handles mixed files
- ✅ Clear separation - encoder-only vs mixed files

---

## Maintenance

### Adding New Encoder-Only Test Files

1. Verify file uses ONLY encoder/encoder-decoder models
2. Add to `--ignore` list in `.buildkite/test-amd.yaml`
3. Update this documentation

### If File Becomes Mixed (adds decoder tests)

1. Remove from `--ignore` list in `.buildkite/test-amd.yaml`
2. Add `@pytest.mark.encoder_decoder` to encoder tests
3. Ensure fixtures use parametrization for simple conftest.py logic

---

## CI Configuration

### API Server (`.buildkite/test-amd.yaml:160-169`)
```yaml
- >-
    pytest -v -s entrypoints/openai
    ... (other ignores)
    --ignore=entrypoints/openai/test_vision.py
    --ignore=entrypoints/openai/test_optional_middleware.py
```

### Pooling (`.buildkite/test-amd.yaml:186-196`)
```yaml
- >-
    pytest -v -s entrypoints/pooling
    --ignore=entrypoints/pooling/correctness/test_mteb_score.py
    --ignore=entrypoints/pooling/correctness/test_mteb_embed.py
    --ignore=entrypoints/pooling/llm/test_embedding.py
    --ignore=entrypoints/pooling/llm/test_encode.py
    --ignore=entrypoints/pooling/openai/test_embedding.py
    --ignore=entrypoints/pooling/openai/test_embedding_dimensions.py
    --ignore=entrypoints/pooling/openai/test_embedding_long_text.py
    --ignore=entrypoints/pooling/openai/test_rerank.py
    --ignore=entrypoints/pooling/openai/test_score.py
    --ignore=entrypoints/pooling/openai/test_truncation.py
    --ignore=entrypoints/pooling/openai/test_vision_embedding.py
```
