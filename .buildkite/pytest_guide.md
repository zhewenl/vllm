# Pytest Guide: Decorators, Conftest, and Parametrization

A comprehensive guide to pytest's core features, with practical examples from vLLM's encoder-decoder skip implementation.

---

## Table of Contents

1. [Pytest Markers (Decorators)](#pytest-markers-decorators)
2. [Parametrization](#parametrization)
3. [conftest.py and Hooks](#conftestpy-and-hooks)
4. [Fixtures](#fixtures)
5. [Real-World Example: Encoder-Decoder Skip](#real-world-example-encoder-decoder-skip)

---

## Pytest Markers (Decorators)

Markers are pytest's way to add metadata to tests. They're like tags that you can use to categorize, skip, or modify test behavior.

### Basic Marker Usage

```python
import pytest

@pytest.mark.slow_test
def test_something():
    # This test is marked as "slow"
    pass

@pytest.mark.encoder_decoder
def test_encoder_model():
    # This test is marked as "encoder_decoder"
    pass
```

### Running Tests by Marker

```bash
# Run only tests marked as slow_test
pytest -m slow_test

# Run all tests EXCEPT slow_test
pytest -m 'not slow_test'

# Combine markers
pytest -m 'slow_test and core_model'
```

### Registering Markers (Best Practice)

Define markers in `pyproject.toml` to avoid warnings:

```toml
[tool.pytest.ini_options]
markers = [
    "slow_test: mark test as slow",
    "encoder_decoder: tests that use encoder-decoder models",
    "core_model: enable this model test in each PR",
]
```

**Without registration:**
```bash
$ pytest
PytestUnknownMarkWarning: Unknown pytest.mark.encoder_decoder
```

**With registration:**
```bash
$ pytest
# No warnings ✅
```

### Module-Level Markers

Apply a marker to ALL tests in a file:

```python
import pytest

pytestmark = pytest.mark.encoder_decoder

def test_foo():
    # Automatically has encoder_decoder marker
    pass

def test_bar():
    # Also has encoder_decoder marker
    pass
```

### Combining Multiple Markers

```python
# On a single test
@pytest.mark.slow_test
@pytest.mark.encoder_decoder
def test_something():
    pass

# Module-level with multiple markers
pytestmark = [pytest.mark.slow_test, pytest.mark.encoder_decoder]
```

---

## Parametrization

Parametrization runs the same test with different inputs. Pytest creates separate test items for each parameter value.

### Function Parametrization

**Basic Example:**

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected
```

**What pytest creates:**
```
test_square[2-4]      # Test 1
test_square[3-9]      # Test 2
test_square[4-16]     # Test 3
```

**Key insight:** Each parametrized variant is a **separate test item** with its own nodeid.

### Single Parameter

```python
@pytest.mark.parametrize("model", [
    "openai/whisper-small",
    "google/gemma-3n-E2B-it",
])
def test_model(model):
    print(f"Testing {model}")
```

**Creates:**
```
test_model[openai/whisper-small]
test_model[google/gemma-3n-E2B-it]
```

### Parametrize with Marks

You can add markers to specific parameter values:

```python
@pytest.mark.parametrize("model", [
    pytest.param("openai/whisper-small", marks=[pytest.mark.encoder_decoder]),
    pytest.param("google/gemma-3n-E2B-it"),  # No marker
    pytest.param("BAAI/bge-base-en-v1.5", marks=[pytest.mark.encoder_decoder, pytest.mark.slow_test]),
])
def test_model(model):
    pass
```

**Result:**
- `test_model[openai/whisper-small]` has `encoder_decoder` marker
- `test_model[google/gemma-3n-E2B-it]` has NO markers
- `test_model[BAAI/bge-base-en-v1.5]` has BOTH markers

### Fixture Parametrization

You can parametrize fixtures instead of test functions:

**Non-Parametrized Fixture:**
```python
@pytest.fixture
def server():
    return start_server("my-model")  # Fixed model

def test_api(server):
    # Nodeid: test_api
    pass
```

**Parametrized Fixture:**
```python
@pytest.fixture(params=["model-A", "model-B"])
def server(request):
    return start_server(request.param)  # Uses parameter

def test_api(server):
    # Creates 2 test items:
    # - test_api[model-A]
    # - test_api[model-B]
    pass
```

**Key difference:**
- Non-parametrized: ONE test, nodeid = `test_api`
- Parametrized: TWO tests, nodeids = `test_api[model-A]`, `test_api[model-B]`

### Multiple Parameters

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [3, 4])
def test_multiply(x, y):
    print(x * y)
```

**Creates 4 tests (Cartesian product):**
```
test_multiply[1-3]
test_multiply[1-4]
test_multiply[2-3]
test_multiply[2-4]
```

### Indirect Parametrization

Use `indirect=True` to pass parameters to fixtures:

```python
@pytest.fixture
def model(request):
    # request.param contains the parameter value
    return load_model(request.param)

@pytest.mark.parametrize("model", ["model-A", "model-B"], indirect=True)
def test_something(model):
    # model is the loaded model, not the string
    pass
```

---

## conftest.py and Hooks

`conftest.py` is a special file that pytest loads automatically. It's used for:
1. Sharing fixtures across multiple test files
2. Implementing hooks to customize pytest behavior
3. Configuration that applies to entire directories

### Where conftest.py is Found

Pytest searches for `conftest.py` in:
- Test file's directory
- Parent directories (up to repo root)

**Example structure:**
```
tests/
├── conftest.py              # Available to all tests
├── unit/
│   ├── conftest.py          # Available to unit tests only
│   └── test_foo.py
└── integration/
    └── test_bar.py
```

### Common Hooks

#### `pytest_collection_modifyitems`

Modify collected test items before they run. This is where you can:
- Add markers dynamically
- Skip tests based on conditions
- Reorder tests

```python
# tests/conftest.py

def pytest_collection_modifyitems(config, items):
    """Modify test items after collection but before execution."""

    for item in items:
        # Add marker to all tests in specific directory
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow_test)

        # Skip tests based on condition
        if "encoder_decoder" in item.keywords:
            if should_skip():
                item.add_marker(pytest.mark.skip(reason="Not supported"))
```

**Real example from vLLM:**
```python
def pytest_collection_modifyitems(config, items):
    # Skip optional tests unless --optional flag is passed
    if not config.getoption("--optional"):
        skip_optional = pytest.mark.skip(reason="need --optional option to run")
        for item in items:
            if "optional" in item.keywords:
                item.add_marker(skip_optional)
```

#### `pytest_addoption`

Add custom command-line options:

```python
def pytest_addoption(parser):
    parser.addoption(
        "--optional",
        action="store_true",
        default=False,
        help="run optional tests"
    )
```

**Usage:**
```bash
pytest --optional  # Runs optional tests
pytest             # Skips optional tests
```

### Understanding `item` in Hooks

When pytest collects tests, each becomes an `item` object with:

```python
item.nodeid          # Full test path, e.g., "tests/test_foo.py::test_bar[param1]"
item.keywords        # Set of markers, e.g., {"encoder_decoder", "asyncio"}
item.callspec        # Only exists for parametrized tests (see below)
```

**Example:**
```python
@pytest.mark.encoder_decoder
@pytest.mark.parametrize("model", ["model-A"])
def test_foo(model):
    pass

# In hook:
# item.nodeid = "test_foo.py::test_foo[model-A]"
# item.keywords = {"encoder_decoder", "asyncio", "parametrize", ...}
# item.callspec.params = {"model": "model-A"}
```

---

## Fixtures

Fixtures provide reusable setup/teardown code for tests.

### Basic Fixture

```python
import pytest

@pytest.fixture
def client():
    c = create_client()
    yield c
    c.cleanup()  # Cleanup after test

def test_api(client):
    # client fixture is automatically injected
    response = client.get("/api")
    assert response.status_code == 200
```

### Fixture Scopes

Control how often fixtures are created:

```python
@pytest.fixture(scope="function")  # Default: new instance per test
def client():
    return create_client()

@pytest.fixture(scope="module")  # One instance per test module
def database():
    return setup_database()

@pytest.fixture(scope="session")  # One instance for entire test session
def app():
    return start_app()
```

### Fixture Dependencies

Fixtures can use other fixtures:

```python
@pytest.fixture
def database():
    return setup_database()

@pytest.fixture
def client(database):
    # Uses database fixture
    return create_client(database)

def test_api(client):
    # client fixture automatically gets database
    pass
```

### Parametrized Fixtures

Create multiple fixture variants:

```python
@pytest.fixture(params=["sqlite", "postgres", "mysql"])
def database(request):
    # request.param contains the current parameter value
    db = setup_database(request.param)
    yield db
    db.cleanup()

def test_query(database):
    # This test runs 3 times, once for each database type
    # Nodeids:
    # - test_query[sqlite]
    # - test_query[postgres]
    # - test_query[mysql]
    pass
```

---

## Real-World Example: Encoder-Decoder Skip

Let's walk through vLLM's encoder-decoder skip implementation to see everything in action.

### Problem

ROCm attention backends don't support encoder/encoder-decoder models. We need to:
1. Skip encoder-only test files entirely (faster CI)
2. Skip encoder variants in mixed files (selective skipping)
3. Let decoder variants run normally

### Solution Components

#### 1. Marker Registration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
markers = [
    "encoder_decoder: tests that use encoder-decoder models, skipped on ROCm",
]
```

#### 2. CI-Level Skip (`.buildkite/test-amd.yaml`)

For files that use ONLY encoder models:

```yaml
- >-
    pytest -v -s entrypoints/pooling
    --ignore=entrypoints/pooling/openai/test_embedding.py
    --ignore=entrypoints/pooling/openai/test_rerank.py
    # ... (11 files total)
```

#### 3. Mixed File with Markers (`test_translation_validation.py`)

File has BOTH encoder and decoder models:

```python
@pytest.fixture(
    scope="module",
    params=["openai/whisper-small", "google/gemma-3n-E2B-it"]
)
def server(request):
    # Parametrized fixture creates 2 variants
    with RemoteOpenAIServer(request.param, SERVER_ARGS) as remote_server:
        yield remote_server, request.param

@pytest.fixture
async def client_and_model(server):
    server, model_name = server
    async with server.get_async_client() as async_client:
        yield async_client, model_name

@pytest.mark.asyncio
@pytest.mark.encoder_decoder  # Mark encoder tests
async def test_basic_audio(foscolo, client_and_model):
    client, model_name = client_and_model
    # Test code...
```

**What pytest creates:**
```
test_basic_audio[openai/whisper-small]     # encoder_decoder marker
test_basic_audio[google/gemma-3n-E2B-it]   # encoder_decoder marker
```

Wait - both have the marker? Yes! But conftest.py will only skip the encoder one...

#### 4. Automatic Skip Logic (`tests/conftest.py`)

```python
from vllm.platforms import current_platform

def pytest_collection_modifyitems(config, items):
    # List of encoder models
    ENCODER_DECODER_MODELS = [
        "openai/whisper-small",
        "openai/whisper-large-v3-turbo",
        "mistralai/Voxtral-Mini-3B-2507",
        "microsoft/Phi-3.5-vision-instruct",
        "intfloat/multilingual-e5-small",
        "BAAI/bge-reranker-base",
        # ... (17 models total)
    ]

    if current_platform.is_rocm():
        skip_encoder_decoder = pytest.mark.skip(
            reason="Encoder-decoder models not supported on ROCm"
        )

        for item in items:
            if "encoder_decoder" in item.keywords:
                # Check if encoder model name is in the test nodeid
                if any(encoder_model in item.nodeid for encoder_model in ENCODER_DECODER_MODELS):
                    item.add_marker(skip_encoder_decoder)
```

**How it works:**

```python
# Item 1:
item.nodeid = "test_basic_audio[openai/whisper-small]"
item.keywords = {"encoder_decoder", "asyncio", ...}

# Check: "encoder_decoder" in keywords? YES
# Check: any encoder model in nodeid?
#   - "openai/whisper-small" in "test_basic_audio[openai/whisper-small]"? YES
# Action: Add skip marker ✅

# Item 2:
item.nodeid = "test_basic_audio[google/gemma-3n-E2B-it]"
item.keywords = {"encoder_decoder", "asyncio", ...}

# Check: "encoder_decoder" in keywords? YES
# Check: any encoder model in nodeid?
#   - "openai/whisper-small" in "test_basic_audio[google/gemma-3n-E2B-it]"? NO
#   - "openai/whisper-large-v3-turbo" in "test_basic_audio[google/gemma-3n-E2B-it]"? NO
#   - ... (check all 17 encoder models)
#   - None match
# Action: Don't skip, let it run ✅
```

#### 5. Result

**On ROCm:**
- `test_basic_audio[openai/whisper-small]` → **SKIPPED** ✅
- `test_basic_audio[google/gemma-3n-E2B-it]` → **RUN** ✅

**On CUDA:**
- Both tests run normally ✅

---

## Complete Examples

### Example 1: Simple Parametrization

```python
import pytest

@pytest.mark.parametrize("model", ["model-A", "model-B", "model-C"])
def test_inference(model):
    result = run_inference(model)
    assert result is not None

# Creates 3 tests:
# - test_inference[model-A]
# - test_inference[model-B]
# - test_inference[model-C]
```

### Example 2: Parametrization with Conditional Skip

```python
import pytest

@pytest.mark.parametrize("model", [
    "openai/whisper-small",      # Encoder
    "google/gemma-3n-E2B-it",    # Decoder
])
@pytest.mark.encoder_decoder
def test_model(model):
    # Simple test
    pass

# In conftest.py:
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "encoder_decoder" in item.keywords:
            if "whisper" in item.nodeid:
                item.add_marker(pytest.mark.skip(reason="Encoder not supported"))

# Result on ROCm:
# - test_model[openai/whisper-small] → SKIPPED
# - test_model[google/gemma-3n-E2B-it] → RUN
```

### Example 3: Fixture Parametrization

```python
import pytest

@pytest.fixture(scope="module", params=["sqlite", "postgres"])
def database(request):
    db = setup_database(request.param)
    yield db
    db.cleanup()

@pytest.fixture
def client(database):
    return create_client(database)

def test_query(client):
    # This test runs twice:
    # - test_query[sqlite]
    # - test_query[postgres]
    result = client.query("SELECT 1")
    assert result is not None

def test_insert(client):
    # This test ALSO runs twice:
    # - test_insert[sqlite]
    # - test_insert[postgres]
    client.insert({"id": 1})
```

**Key insight:** Fixture parametrization affects ALL tests that use the fixture (directly or indirectly).

### Example 4: Combined Function and Fixture Parametrization

```python
@pytest.fixture(params=["backend-A", "backend-B"])
def server(request):
    return start_server(request.param)

@pytest.mark.parametrize("model", ["model-1", "model-2"])
def test_inference(server, model):
    # Creates 4 tests (2 backends × 2 models):
    # - test_inference[backend-A-model-1]
    # - test_inference[backend-A-model-2]
    # - test_inference[backend-B-model-1]
    # - test_inference[backend-B-model-2]
    pass
```

---

## Understanding item.callspec

When pytest parametrizes a test, it creates a `callspec` attribute on the test item:

```python
@pytest.mark.parametrize("model", ["model-A"])
def test_foo(model):
    pass

# In hook:
item.callspec           # Exists ✅
item.callspec.params    # {"model": "model-A"}
hasattr(item, "callspec")  # True
```

**Non-parametrized test:**
```python
def test_bar():
    pass

# In hook:
item.callspec           # Does NOT exist
hasattr(item, "callspec")  # False
```

**Why this matters:**

You can detect if a test is parametrized:

```python
def pytest_collection_modifyitems(config, items):
    for item in items:
        if hasattr(item, "callspec"):
            # This is a parametrized test
            # Model name might be in nodeid
            if "encoder-model" in item.nodeid:
                skip
        else:
            # Non-parametrized test
            # Model name is NOT in nodeid (it's a module constant)
            if "encoder_decoder" in item.keywords:
                skip  # Skip all marked non-parametrized tests
```

---

## Practical Patterns

### Pattern 1: Skip Tests on Specific Platforms

```python
# tests/conftest.py
from vllm.platforms import current_platform

def pytest_collection_modifyitems(config, items):
    if current_platform.is_rocm():
        for item in items:
            if "cuda_only" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="CUDA only test"))
```

```python
# test_something.py
@pytest.mark.cuda_only
def test_cuda_feature():
    # Automatically skipped on ROCm
    pass
```

### Pattern 2: Selective Skipping with Parametrization

Mark the entire test but only skip specific variants:

```python
# test_models.py
@pytest.mark.parametrize("model", [
    "encoder-model",
    "decoder-model",
])
@pytest.mark.encoder_decoder
def test_model(model):
    pass

# conftest.py
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "encoder_decoder" in item.keywords:
            if "encoder-model" in item.nodeid:
                skip  # Only skip encoder variant
```

**Result:**
- `test_model[encoder-model]` → SKIPPED
- `test_model[decoder-model]` → RUN

### Pattern 3: Module-Level Markers with CI Skip

For files that are 100% encoder-only:

```python
# test_encoder_only.py
pytestmark = pytest.mark.encoder_decoder

def test_foo():
    pass

def test_bar():
    pass
```

**CI level:**
```yaml
# Skip entire file at CI level (faster)
pytest --ignore=test_encoder_only.py
```

**Local testing:**
```bash
# File still has markers for local ROCm developers
pytest test_encoder_only.py  # Tests auto-skip on ROCm
```

---

## Key Concepts Summary

### Markers
- **Purpose:** Tag tests with metadata
- **Scope:** Function-level, module-level, or parametrize-level
- **Usage:** Filter, skip, or modify test behavior
- **Registration:** Define in `pyproject.toml` to avoid warnings

### Parametrization
- **Purpose:** Run same test with different inputs
- **Types:** Function parametrization, fixture parametrization
- **Result:** Creates separate test items (with separate nodeids)
- **Key insight:** Each variant is independent

### conftest.py
- **Purpose:** Share fixtures and customize pytest behavior
- **Scope:** Applies to directory and subdirectories
- **Hooks:** Modify collection, add options, customize execution
- **Location:** Anywhere in test hierarchy

### item.callspec
- **Purpose:** Detect if test is parametrized
- **Exists:** Only on parametrized tests
- **Usage:** `hasattr(item, "callspec")`
- **Why useful:** Different handling for parametrized vs non-parametrized

### nodeid
- **Format:** `path/to/file.py::TestClass::test_function[param1]`
- **Contains:** File path, test name, parameters (if any)
- **Usage:** Check if specific value is in test parameters
- **Key insight:** Parametrization adds `[param]` to nodeid

---

## Common Pitfalls and Solutions

### Pitfall 1: Marker on Test But Model Not in Nodeid

**Problem:**
```python
MODEL_NAME = "encoder-model"  # Module constant

@pytest.fixture(scope="module")
def server():
    return RemoteOpenAIServer(MODEL_NAME, ...)  # Not parametrized!

@pytest.mark.encoder_decoder
def test_foo(client):
    pass

# Nodeid: "test_foo" (no model name!)
```

**Hook that fails:**
```python
if "encoder-model" in item.nodeid:  # FALSE - no model in nodeid
    skip
```

**Solution:** Use parametrized fixture:
```python
@pytest.fixture(scope="module", params=["encoder-model"])
def server(request):
    return RemoteOpenAIServer(request.param, ...)  # Parametrized!

# Nodeid: "test_foo[encoder-model]" ✅
```

### Pitfall 2: Fixture Parametrization with Dicts

**Problem:**
```python
@pytest.fixture(params=[{"name": "model-A"}, {"name": "model-B"}])
def model(request):
    return request.param

def test_foo(model):
    pass

# Nodeids: test_foo[model0], test_foo[model1]
# Model names NOT in nodeid!
```

**Solution 1:** Use string params instead:
```python
@pytest.fixture(params=["model-A", "model-B"])
def model(request):
    return {"name": request.param}

# Nodeids: test_foo[model-A], test_foo[model-B] ✅
```

**Solution 2:** Use `ids` parameter:
```python
@pytest.fixture(params=[
    {"name": "model-A"},
    {"name": "model-B"}
], ids=["model-A", "model-B"])
def model(request):
    return request.param

# Nodeids: test_foo[model-A], test_foo[model-B] ✅
```

### Pitfall 3: Forgetting to Check Platform

**Problem:**
```python
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "encoder_decoder" in item.keywords:
            skip  # Skips on ALL platforms!
```

**Solution:**
```python
from vllm.platforms import current_platform

def pytest_collection_modifyitems(config, items):
    if current_platform.is_rocm():  # Only skip on ROCm
        for item in items:
            if "encoder_decoder" in item.keywords:
                skip
```

---

## Advanced Techniques

### Conditional Fixture Parametrization

```python
import pytest
import sys

# Different params based on platform
DATABASES = ["sqlite", "postgres"]
if sys.platform == "linux":
    DATABASES.append("mysql")

@pytest.fixture(params=DATABASES)
def database(request):
    return setup_database(request.param)
```

### Marker with Arguments

```python
@pytest.mark.skipif(sys.version_info < (3, 10), reason="Requires Python 3.10+")
def test_new_syntax():
    pass
```

### Dynamic Marker Addition

```python
def pytest_collection_modifyitems(config, items):
    for item in items:
        # Add marker based on test location
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add marker based on test name
        if item.name.startswith("test_slow_"):
            item.add_marker(pytest.mark.slow_test)
```

### Accessing Marker Data in Tests

```python
@pytest.mark.custom_marker(value=42)
def test_something(request):
    marker = request.node.get_closest_marker("custom_marker")
    assert marker.kwargs["value"] == 42
```

---

## Best Practices

### 1. Register All Custom Markers
Always define markers in `pyproject.toml`:
```toml
markers = [
    "slow_test: mark test as slow",
    "encoder_decoder: tests that use encoder-decoder models",
]
```

### 2. Use Descriptive Marker Names
- ✅ Good: `@pytest.mark.encoder_decoder`
- ❌ Bad: `@pytest.mark.skip_on_amd`

### 3. Prefer Parametrization Over Loops
```python
# ❌ Bad: Loop in test
def test_models():
    for model in ["model-A", "model-B"]:
        result = run_inference(model)
        assert result  # If model-A fails, model-B never runs

# ✅ Good: Parametrization
@pytest.mark.parametrize("model", ["model-A", "model-B"])
def test_models(model):
    result = run_inference(model)
    assert result  # Both run independently, clear which failed
```

### 4. Use Fixture Parametrization for Setup Variations
When the setup changes, not just test inputs:

```python
# ✅ Good: Fixture parametrization for different setups
@pytest.fixture(params=["config-A", "config-B"])
def server(request):
    return start_server(config=request.param)

def test_api(server):
    # Runs with both server configurations
    pass
```

### 5. Keep conftest.py Logic Simple
- Extract complex logic to helper functions
- Add comments explaining non-obvious behavior
- Keep hooks focused and readable

---

## Debugging Tips

### View Collected Tests
```bash
# See all tests without running
pytest --collect-only

# See with markers
pytest --collect-only -v

# See with specific marker
pytest --collect-only -m encoder_decoder
```

### Show Skip Reasons
```bash
# Show why tests were skipped
pytest -rs

# Verbose skip info
pytest -rsv
```

### Debug Hook Execution
```python
def pytest_collection_modifyitems(config, items):
    print(f"\n=== Collected {len(items)} items ===")
    for item in items:
        print(f"  {item.nodeid}")
        print(f"    keywords: {item.keywords}")
        print(f"    has callspec: {hasattr(item, 'callspec')}")
```

### Check If Marker Is Applied
```python
def test_something(request):
    if request.node.get_closest_marker("encoder_decoder"):
        print("This test has encoder_decoder marker")
```

---

## Comparison: Different Approaches

### Approach 1: Skip in Test (Explicit)

```python
def test_foo(model_name):
    if platform.is_rocm() and model_name in ENCODER_MODELS:
        pytest.skip("Not supported")
    # Test code...
```

**Pros:** Explicit, easy to understand
**Cons:** Repetitive, skip check in every test

### Approach 2: Marker + Hook (Automatic)

```python
@pytest.mark.encoder_decoder
def test_foo(model_name):
    # No skip check!
    # Test code...

# conftest.py handles skipping
```

**Pros:** DRY, centralized logic, clean tests
**Cons:** "Magic" - need to understand conftest.py

### Approach 3: CI-Level Skip

```yaml
pytest --ignore=test_encoder_only.py
```

**Pros:** Fastest (not even collected), explicit in CI
**Cons:** Local developers need to know to skip manually

### Approach 4: Hybrid (vLLM's Approach)

- CI skip for 100% encoder files
- Marker + hook for mixed files
- Simple 5-line hook

**Pros:** Best of all approaches
**Cons:** Need to maintain both CI config and markers

---

## Real-World Decision Tree

```
Does file use encoder models?
├─ No → No action needed
└─ Yes
   ├─ 100% encoder-only?
   │  └─ Yes → Skip at CI level (--ignore)
   └─ No (mixed encoder + decoder)
      └─ Use markers:
         ├─ Parametrized? → Model name in nodeid
         │  └─ conftest.py: if encoder in nodeid: skip
         └─ Not parametrized? → Refactor to use parametrization
            └─ Makes nodeid contain model name
```

---

## References

- **Pytest Documentation:** https://docs.pytest.org/
- **Markers:** https://docs.pytest.org/en/stable/how-to/mark.html
- **Parametrization:** https://docs.pytest.org/en/stable/how-to/parametrize.html
- **conftest.py:** https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py
- **Hooks:** https://docs.pytest.org/en/stable/reference/reference.html#hooks

---

## Summary

**Markers:** Add metadata to tests for filtering and conditional behavior
**Parametrization:** Run same test with different inputs (creates separate test items)
**conftest.py:** Share fixtures and customize pytest with hooks
**Hooks:** Intercept pytest's workflow to modify behavior
**item.callspec:** Detect parametrized tests and access parameter values
**nodeid:** Unique identifier for each test, includes parameters if any

The key to understanding pytest is recognizing that **parametrization creates separate test items**, each with its own nodeid and metadata. Hooks can then inspect and modify these items before execution.
