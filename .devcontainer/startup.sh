#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync -v --extra Clarabel --frozen

uv pip install marimo
