#!/bin/bash
# Development container startup script for cvxrisk

# Install uv package manager (faster alternative to pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment in the default location (.venv)
uv venv

# Install all dependencies including development dependencies and all extras
# --all-extras: Install all optional dependencies defined in pyproject.toml
# --dev: Include development dependencies
# --frozen: Use exact versions for reproducibility
uv sync --all-extras --dev --frozen

# Install marimo for interactive notebooks
# --no-cache-dir: Don't use the pip cache to ensure clean installation
uv pip install --no-cache-dir marimo
