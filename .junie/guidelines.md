# CVXRisk Development Guidelines

This document provides guidelines for developing and contributing to the CVXRisk project.

## Build/Configuration Instructions

### Environment Setup

CVXRisk uses `uv` for dependency management and virtual environment creation. To set up the development environment:

1. Install `uv` if not already installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment:
   ```bash
   uv venv --python 3.12
   ```

3. Install dependencies:
   ```bash
   uv sync --dev --frozen
   ```

### Makefile Commands

The project includes a Makefile with several useful commands:

- `make install`: Create a virtual environment and install dependencies
- `make fmt`: Run code formatting and linting
- `make test`: Run all tests
- `make clean`: Clean generated files and directories
- `make marimo`: Start a Marimo server for interactive notebooks
- `make help`: Display help information

## Testing Information

### Testing Framework

CVXRisk uses pytest for testing. Tests are organized in the `tests` directory, with subdirectories corresponding to the modules in the `cvx` package:

- `tests/test_risk`: Tests for the `cvx.risk` module
- `tests/test_random`: Tests for the `cvx.random` module

### Running Tests

To run all tests:

```bash
make test
```

To run a specific test file:

```bash
uv run pytest tests/path/to/test_file.py
```

To run tests with verbose output:

```bash
uv run pytest tests/path/to/test_file.py -v
```

### Writing Tests

Tests should follow these guidelines:

1. Test files should be named with a `test_` prefix
2. Test functions should also be named with a `test_` prefix
3. Use pytest fixtures for common setup
4. Use pytest.approx for floating-point comparisons
5. Use pytest.raises for testing exceptions

Example test:

```python
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.risk.bounds import Bounds


def test_bounds_initialization():
    """Test that Bounds can be initialized with default parameters."""
    bounds = Bounds(m=5, name="test")

    # Check that the parameters are initialized correctly
    assert bounds.m == 5
    assert bounds.name == "test"

    # Check that the parameter dictionary contains the expected keys
    assert "lower_test" in bounds.parameter
    assert "upper_test" in bounds.parameter

    # Check that the parameters have the expected values
    assert bounds.parameter["lower_test"].value.shape == (5,)
    assert bounds.parameter["upper_test"].value.shape == (5,)
    assert np.all(bounds.parameter["lower_test"].value == 0)
    assert np.all(bounds.parameter["upper_test"].value == 1)
```

## Code Style and Development Practices

### Code Style

CVXRisk uses Ruff for code formatting and linting. The configuration is in the `pyproject.toml` file:

- Line length: 120 characters
- Target Python version: 3.10+
- Linting rules: E (pycodestyle errors), F (pyflakes), I (isort)

To run the formatter and linter:

```bash
make fmt
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. The hooks are configured in `.pre-commit-config.yaml` and include:

- Code formatting with Ruff
- Linting with Ruff
- Markdown linting
- Python code upgrading with pyupgrade
- JSON schema validation
- License header insertion
- GitHub Actions linting
- pyproject.toml validation
- Typo checking

To install the pre-commit hooks:

```bash
pre-commit install
```

### Project Structure

The project is organized as follows:

- `cvx/`: Main package
  - `cvx/risk/`: Risk models
  - `cvx/random/`: Random number generation
  - `cvx/portfolio/`: Portfolio optimization
- `tests/`: Test files
- `book/`: Documentation and examples

### Dependencies

The project has the following main dependencies:

- cvxpy-base: Convex optimization framework
- numpy: Numerical computing
- pandas: Data analysis
- scikit-learn: Machine learning
- scipy: Scientific computing

Development dependencies include:

- pytest: Testing framework
- pytest-cov: Test coverage
- pre-commit: Pre-commit hooks
- clarabel: Solver
- cvxsimulator: Simulation framework
