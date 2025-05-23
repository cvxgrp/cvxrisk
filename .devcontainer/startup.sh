#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync --all-extras --dev --frozen
uv pip install --no-cache-dir marimo

uv pip install marimo
