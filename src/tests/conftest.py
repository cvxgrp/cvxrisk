"""Pytest configuration and fixtures for the cvxrisk test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resource_dir():
    """Pytest fixture that provides the path to the test resources directory.

    This fixture has session scope, meaning it's created once per test session.
    It returns the path to the 'resources' directory within the tests directory,
    which contains data files used by various tests.

    Returns:
        pathlib.Path: Path to the test resources directory

    """
    return Path(__file__).parent / "resources"


@pytest.fixture()
def readme_path() -> Path:
    """Provide the path to the project's README.md file.

    This fixture searches for the README.md file by starting in the current
    directory and moving up through parent directories until it finds the file.

    Returns:
    -------
    Path
        Path to the README.md file

    Raises:
    ------
    FileNotFoundError
        If the README.md file cannot be found in any parent directory

    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        candidate = current_dir / "README.md"
        if candidate.is_file():
            return candidate
        current_dir = current_dir.parent
    raise FileNotFoundError("README.md not found in any parent directory")
