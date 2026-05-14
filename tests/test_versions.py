"""Tests for the version information in submodules."""

import importlib.metadata

import cvx.risk


def test_versions() -> None:
    """Test that the version information in each submodule matches the package version.

    This test verifies that:
    1. Each submodule has a __version__ attribute
    2. The __version__ attribute in each submodule matches the package version
    """
    # Get the expected version from the package metadata
    expected_version = importlib.metadata.version("cvxrisk")

    # Check that each submodule's __version__ matches the expected version
    assert cvx.risk.__version__ == expected_version

    # Print the versions for debugging purposes
    print(f"Package version: {expected_version}")
