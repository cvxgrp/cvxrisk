from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resource_dir():
    """
    Pytest fixture that provides the path to the test resources directory.

    This fixture has session scope, meaning it's created once per test session.
    It returns the path to the 'resources' directory within the tests directory,
    which contains data files used by various tests.

    Returns:
        pathlib.Path: Path to the test resources directory
    """
    return Path(__file__).parent / "resources"
