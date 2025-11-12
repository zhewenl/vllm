---
name: vllm-codestyle
description: Enforce vLLM coding style (no fancy comments, no emojis, follow existing patterns)
---

# vLLM Code Style Skill

This skill enforces the vLLM project's coding style guidelines to maintain consistency across the codebase.

## Core Style Rules

### 1. No Fancy Comments
- Keep comments simple and informative
- Avoid decorative comment blocks or ASCII art
- No excessive separators or visual flourishes
- Use clear, concise language

**Bad:**
```python
################################
# THIS IS AN IMPORTANT FUNCTION
################################

# ========== Helper Functions ==========

# TODO: Fix this!!! URGENT!!! 🚨🚨🚨
```

**Good:**
```python
# Process the input tensor and apply attention.

# Helper functions

# TODO: Fix attention mask handling
```

### 2. No Emojis
- Never use emojis in code, comments, or docstrings
- Keep all text professional and plain
- This includes commit messages and documentation

### 3. Follow Existing Coding Style
- Match the style of surrounding code
- Use consistent naming conventions
- Follow established patterns for similar functionality
- Respect existing indentation and formatting

## Pre-Commit Checks

The project uses pre-commit hooks for automated style enforcement:

- **ruff**: Python linting and formatting
- **clang-format**: C++/CUDA formatting
- **mypy**: Type checking
- **typos**: Spell checking

Run before committing:
```bash
pre-commit run --all-files
```

## Style Guidelines by Language

### Python
- Follow PEP 8 conventions
- Use type hints consistently
- Keep functions focused and single-purpose
- Use descriptive variable names (no single letters except in loops)

### C++/CUDA
- Follow existing kernel style in `csrc/`
- Use clear variable names
- Comment complex CUDA operations
- Follow naming: `snake_case` for functions, `PascalCase` for classes

### Comments
- Explain "why", not "what"
- Keep comments concise
- Update comments when code changes
- Remove commented-out code (use git history instead)

## Instructions

When this skill is invoked or when writing/reviewing code:

1. **Review all code for style violations**:
   - Check for fancy comments or ASCII art
   - Remove any emojis
   - Ensure naming follows project conventions

2. **Match existing patterns**:
   - Look at similar functions in the same file
   - Follow the same structure and style
   - Use consistent parameter naming

3. **Run pre-commit checks**:
   ```bash
   pre-commit run --all-files
   ```

4. **Simplify where possible**:
   - Remove unnecessary complexity
   - Use clear, direct language
   - Avoid over-commenting obvious code

## Common Violations to Avoid

1. Decorative separators:
   ```python
   # ============================================================
   # DO NOT USE THIS
   # ============================================================
   ```

2. Excessive emphasis:
   ```python
   # !!! IMPORTANT !!! READ THIS !!! CRITICAL !!!
   ```

3. Emojis anywhere:
   ```python
   # Fixed the bug! 🎉
   # TODO: Refactor this 📝
   ```

4. Inconsistent naming:
   ```python
   # Mixing styles in same file
   def process_input():  # snake_case
       myVariable = 1    # camelCase - inconsistent!
   ```

## Example: Before and After

**Before (violates style):**
```python
################################
# 🚀 SUPER IMPORTANT FUNCTION 🚀
################################

def ProcessData(input_tensor):  # Inconsistent naming
    # !!! This is where the magic happens !!!
    result = input_tensor * 2  # multiply by 2 🎯
    return result
```

**After (follows style):**
```python
# Scale input tensor by factor of 2.
def process_data(input_tensor):
    # Apply scaling for normalization.
    result = input_tensor * 2
    return result
```

## Notes

- Consistency is more important than personal preference
- When in doubt, match the surrounding code
- Style enforcement helps maintainability and code review
- Run linters before creating PRs
