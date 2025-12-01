"""Tests for module docstrings using doctest.

This module runs doctest on all modules in the source package(s) to ensure
that the code examples in docstrings are correct and up-to-date, without
hardcoding any project-specific module names.

We assume the project is structured as a Python package with a "src" directory
containing subpackages for each module.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
import sys
import warnings
from pathlib import Path
from collections.abc import Iterator
from types import ModuleType


def _find_src_dir(start: Path) -> Path | None:
    """Find the project's first package directory under src.

    Walk up from the starting path until a "src" directory is found, then
    search within it for the first directory containing an __init__.py file
    (e.g., for this project: src/cvx/risk). If none is found, return the
    "src" directory itself. Returns None if no "src" is found.
    """
    current = start.resolve()
    while True:
        src_root = current / "src"
        if src_root.is_dir():
            # Prefer the first package directory that contains an __init__.py
            init_files = sorted(src_root.rglob("__init__.py"))
            if init_files:
                return init_files[0].parent
            return src_root
        if current.parent == current:
            return None
        current = current.parent


def iter_modules(package_name: str | None = None) -> Iterator[ModuleType]:
    """Iterate over all modules in a package (recursively) or in all src packages.

    If ``package_name`` is provided, this function recursively discovers and
    imports all modules within that package. If it is omitted, the function
    discovers all top-level packages under the repository's "src" directory and
    iterates through all their submodules.

    Args:
        package_name: Optional fully-qualified package name to iterate over.

    Yields:
        Imported module objects for each discovered module.

    Example:
        Iterating the standard library doctest package (avoids project-specific names):

        >>> mods = list(iter_modules("doctest"))
        >>> len(mods) > 0
        True
    """
    if package_name is not None:
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            yield package
            return
        # Include the package itself
        yield package
        for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
            try:
                module = importlib.import_module(name)
                yield module
            except ImportError:
                continue
        return

    # No package specified: discover packages under src (supports namespace pkgs)
    src_or_pkg_dir = _find_src_dir(Path(__file__).parent)
    if src_or_pkg_dir is None:
        raise RuntimeError("Could not locate project 'src' directory for module discovery")

    # Determine the real src root and optional base package path
    probe = src_or_pkg_dir
    src_root = None
    while True:
        if probe.name == "src":
            src_root = probe
            break
        if probe.parent == probe:
            break
        probe = probe.parent
    if src_root is None:
        src_root = src_or_pkg_dir

    # Ensure src_root is on sys.path for import
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

    if src_or_pkg_dir == src_root:
        # Discover all top-level packages under src
        for _, name, ispkg in pkgutil.iter_modules([src_root_str]):
            if not ispkg:
                continue
            pkg = importlib.import_module(name)
            yield pkg
            if hasattr(pkg, "__path__"):
                for _, subname, _ in pkgutil.walk_packages(pkg.__path__, prefix=name + "."):
                    try:
                        module = importlib.import_module(subname)
                        yield module
                    except ImportError:
                        continue
    else:
        # We found a concrete package directory under src (e.g., src/cvx/risk)
        base_pkg = src_or_pkg_dir.relative_to(src_root).as_posix().replace("/", ".")
        pkg = importlib.import_module(base_pkg)
        yield pkg
        if hasattr(pkg, "__path__"):
            for _, subname, _ in pkgutil.walk_packages(pkg.__path__, prefix=base_pkg + "."):
                try:
                    module = importlib.import_module(subname)
                    yield module
                except ImportError:
                    continue


def test_docstrings() -> None:
    """Test that all docstrings in the discovered packages pass doctest.

    This test discovers all packages under the repository's src directory and
    runs doctest on each discovered module. It verifies that code examples in
    docstrings are correct and produce the expected output.

    The test collects all modules with doctests and runs them, failing if any
    doctest fails.

    Raises:
        AssertionError: If any doctest in any module fails.
    """
    # Collect all modules (auto-discover from src)
    modules = list(iter_modules())

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

    # raise a warning if no tests were found
    if total_tests == 0:
        warnings.warn("No doctests were found")


