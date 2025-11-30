"""Tests for module docstrings using doctest.

This module runs doctest on all modules in the cvxrisk package to ensure
that the code examples in docstrings are correct and up-to-date.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from collections.abc import Iterator
from types import ModuleType


def iter_modules(package_name: str) -> Iterator[ModuleType]:
    """Iterate over all modules in a package, including subpackages.

    This function recursively discovers and imports all modules within
    a given package.

    Args:
        package_name: The name of the package to iterate over.

    Yields:
        Imported module objects for each module in the package.

    Example:
        >>> modules = list(iter_modules("cvx.risk"))
        >>> len(modules) > 0
        True

    """
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        yield package
        return

    yield package

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        try:
            module = importlib.import_module(name)
            yield module
        except ImportError:
            continue


def test_docstrings() -> None:
    """Test that all docstrings in the cvxrisk package pass doctest.

    This test iterates over all modules in the cvx.risk package and runs
    doctest on each one. It verifies that code examples in docstrings
    are correct and produce the expected output.

    The test collects all modules with doctests and runs them, failing
    if any doctest fails.

    Raises:
        AssertionError: If any doctest in any module fails.

    """
    # Collect all modules
    modules = list(iter_modules("cvx.risk"))

    # Track results
    total_tests = 0
    total_failures = 0
    failed_modules = []

    for module in modules:
        # Run doctest on each module
        results = doctest.testmod(
            module,
            verbose=False,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        )

        total_tests += results.attempted
        total_failures += results.failed

        if results.failed > 0:
            failed_modules.append((module.__name__, results.failed, results.attempted))

    # Print summary
    print(f"\nDoctest summary: {total_tests} tests in {len(modules)} modules")
    print(f"Failures: {total_failures}")

    if failed_modules:
        print("\nFailed modules:")
        for mod_name, failures, attempted in failed_modules:
            print(f"  {mod_name}: {failures}/{attempted} failed")

    # Assert no failures
    assert total_failures == 0, f"Doctest failures in: {[m[0] for m in failed_modules]}"
    assert total_tests > 0, "No doctests were found"


def test_docstrings_source_modules() -> None:
    """Test docstrings in specific source modules individually.

    This test explicitly tests each main source module to ensure
    comprehensive doctest coverage.

    The modules tested include:
        - cvx.risk.model
        - cvx.risk.bounds
        - cvx.risk.sample.sample
        - cvx.risk.cvar.cvar
        - cvx.risk.factor.factor
        - cvx.risk.linalg.cholesky
        - cvx.risk.linalg.pca
        - cvx.risk.linalg.valid
        - cvx.risk.portfolio.min_risk
        - cvx.risk.random.rand_cov

    """
    modules_to_test = [
        "cvx.risk.model",
        "cvx.risk.bounds",
        "cvx.risk.sample.sample",
        "cvx.risk.cvar.cvar",
        "cvx.risk.factor.factor",
        "cvx.risk.linalg.cholesky",
        "cvx.risk.linalg.pca",
        "cvx.risk.linalg.valid",
        "cvx.risk.portfolio.min_risk",
        "cvx.risk.random.rand_cov",
    ]

    for module_name in modules_to_test:
        module = importlib.import_module(module_name)
        results = doctest.testmod(
            module,
            verbose=False,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        )

        print(f"{module_name}: {results.attempted} tests, {results.failed} failures")

        assert results.failed == 0, f"Doctest failures in {module_name}"


def test_docstrings_init_modules() -> None:
    """Test docstrings in __init__ modules.

    This test verifies that the package-level docstrings with examples
    in __init__.py files are correct.

    """
    init_modules = [
        "cvx.risk",
        "cvx.risk.cvar",
        "cvx.risk.factor",
        "cvx.risk.linalg",
        "cvx.risk.portfolio",
        "cvx.risk.random",
        "cvx.risk.sample",
    ]

    for module_name in init_modules:
        module = importlib.import_module(module_name)
        results = doctest.testmod(
            module,
            verbose=False,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        )

        print(f"{module_name}: {results.attempted} tests, {results.failed} failures")

        assert results.failed == 0, f"Doctest failures in {module_name}"
