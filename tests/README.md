# Tests Directory

This directory contains test cases for the Agentic IR framework.

## Test Organization

Tests are organized by module and component:

- `test_core/`: Tests for core components (state, memory, thought, etc.)
- `test_tools/`: Tests for various tools
- `test_environments/`: Tests for environment implementations
- `test_llm/`: Tests for LLM integration
- `test_examples/`: Integration tests for example applications

## Running Tests

To run all tests:

```bash
cd agentic-ir
python -m pytest
```

To run specific tests:

```bash
# Run tests for core components
python -m pytest tests/test_core

# Run a specific test file
python -m pytest tests/test_core/test_state.py
``` 