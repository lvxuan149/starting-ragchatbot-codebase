# Test Configuration Guide

## Running Tests

### Prerequisites
- Python 3.13+ installed
- Virtual environment created (recommended)
- All dependencies installed

### Quick Start
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock httpx

# Run all tests
pytest

# Run with coverage
./run_tests.sh
```

### Test Categories

#### Unit Tests
```bash
# Run all unit tests
pytest tests/unit/

# Run specific unit test file
pytest tests/unit/test_document_processor.py

# Run with verbose output
pytest tests/unit/ -v
```

#### Integration Tests
```bash
# Run all integration tests
pytest tests/integration/

# Run API integration tests
pytest tests/integration/test_api.py
```

#### End-to-End Tests
```bash
# Run all e2e tests
pytest tests/e2e/

# Run RAG workflow tests
pytest tests/e2e/test_rag_workflow.py
```

### Running Tests with Marks

The test suite uses custom markers for better organization:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only e2e tests
pytest -m e2e

# Run all tests except slow ones
pytest -m "not slow"
```

### Coverage Reports

Generate detailed coverage reports:

```bash
# HTML report (opens in browser)
pytest --cov=backend --cov-report=html

# Terminal report with missing lines
pytest --cov=backend --cov-report=term-missing

# XML report for CI/CD
pytest --cov=backend --cov-report=xml
```

### Test Configuration

The `pyproject.toml` file contains pytest configuration:

- Test paths: `tests/`
- Test file pattern: `test_*.py`
- Custom markers for test categorization
- Coverage configuration

### Mocking and Fixtures

The test suite uses:
- `pytest-mock` for mocking external dependencies
- Custom fixtures in `tests/fixtures/` for test data
- `conftest.py` for shared fixtures and mocks

### Running Specific Tests

```bash
# Run tests matching a keyword
pytest -k "document_processor"

# Run tests with a specific marker
pytest -m "unit and not slow"

# Run tests until first failure
pytest --maxfail=1

# Run tests with debugging
pytest --pdb
```

### CI/CD Integration

For CI/CD pipelines, use:

```bash
# Run tests with coverage and XML report
pytest --cov=backend --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml
```