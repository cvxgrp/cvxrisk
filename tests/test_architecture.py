"""Import-layering contract for the ``cvx`` package (#508).

The package is a deliberate two-layer design:

- ``cvx.core`` is the foundation layer (parameters, variables, bounds, the
  abstract :class:`~cvx.core.model.Model`, the cone-program builder).
- ``cvx.risk`` is the domain layer built on top of ``cvx.core``.

These tests machine-enforce that direction so a future edit cannot silently
introduce an upward ``core -> risk`` import or an import cycle. They parse the
AST of every first-party module under ``src/cvx`` and therefore run without
importing the heavy solver dependencies.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_PKG_ROOT = _SRC / "cvx"


def _module_name(path: Path) -> str:
    """Dotted module name for a file under ``src`` (``__init__.py`` -> package)."""
    parts = list(path.relative_to(_SRC).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _first_party_modules() -> dict[str, Path]:
    """Map every first-party ``cvx.*`` module name to its source file."""
    return {_module_name(path): path for path in _PKG_ROOT.rglob("*.py")}


def _resolve(target: str, known: set[str]) -> str | None:
    """Reduce an imported name to the first-party module it belongs to.

    ``from cvx.core import Model`` yields the target ``cvx.core.Model`` whose
    owning module is ``cvx.core``; walk the dotted prefixes longest-first and
    return the first one that is a real module.
    """
    parts = target.split(".")
    for stop in range(len(parts), 0, -1):
        candidate = ".".join(parts[:stop])
        if candidate in known:
            return candidate
    return None


def _imported_first_party(path: Path, known: set[str]) -> set[str]:
    """First-party ``cvx`` modules imported by ``path`` (incl. ``TYPE_CHECKING``).

    Relative imports are resolved against the importing module's package so
    they map onto the same absolute module names as the rest of the graph.
    """
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    edges: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (owner := _resolve(alias.name, known)) is not None:
                    edges.add(owner)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -> resolve against the package
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                prefix = ".".join(base)
                root = f"{prefix}.{node.module}" if node.module else prefix
            else:
                root = node.module or ""
            if not root.startswith("cvx"):
                continue
            for alias in node.names:
                if (owner := _resolve(f"{root}.{alias.name}", known)) is not None:
                    edges.add(owner)
            if (owner := _resolve(root, known)) is not None:
                edges.add(owner)
    return edges - {module}


def _import_graph() -> dict[str, set[str]]:
    modules = _first_party_modules()
    known = set(modules)
    return {name: _imported_first_party(path, known) for name, path in modules.items()}


def test_core_never_imports_risk() -> None:
    """No module in the ``cvx.core`` layer may import from ``cvx.risk``."""
    graph = _import_graph()
    violations = {
        source: sorted(t for t in targets if t.startswith("cvx.risk"))
        for source, targets in graph.items()
        if source.startswith("cvx.core") and any(t.startswith("cvx.risk") for t in targets)
    }
    assert not violations, f"Upward core -> risk imports (layering violation): {violations}"


def test_import_graph_is_acyclic() -> None:
    """The first-party ``cvx`` import graph must contain no cycles."""
    graph = _import_graph()
    visiting, done = set(), set()
    stack: list[str] = []

    def walk(node: str) -> list[str] | None:
        visiting.add(node)
        stack.append(node)
        for nxt in graph.get(node, set()):
            if nxt in visiting:
                return [*stack[stack.index(nxt) :], nxt]
            if nxt not in done and (cycle := walk(nxt)) is not None:
                return cycle
        visiting.discard(node)
        stack.pop()
        done.add(node)
        return None

    for start in graph:
        if start not in done:
            cycle = walk(start)
            assert cycle is None, f"Import cycle detected: {' -> '.join(cycle)}"
