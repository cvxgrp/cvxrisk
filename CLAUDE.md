# CLAUDE.md

Guidance for working in this repository.

## What this is

`cvxrisk` — a small risk-model engine for portfolio optimization, part of the
[cvxgrp](https://github.com/cvxgrp) ecosystem. It builds convex risk models and
hands them to Clarabel directly: `clarabel`, `cvx-linalg`, `numpy` and `scipy`
are the whole runtime dependency set. `cvxpy-base` lives in the `benchmark`
dependency group only, where `tests/benchmarks/` times this engine against the
older cvxpy-based formulation.

Two namespace packages under `src/cvx/` (there is deliberately no
`src/cvx/__init__.py` — `cvx` is a namespace shared with sibling projects such as
`cvx-linalg`):

- `cvx.core` — the solver-facing primitives: `bounds`, `conic`, `model`,
  `parameter`, `variable`.
- `cvx.risk` — the risk models, one subpackage per family: `sample` (sample
  covariance), `factor` (factor model), `cvar` (conditional value at risk), and
  `portfolio/min_risk` (the minimum-risk problem).

`stubs/clarabel/__init__.pyi` is a hand-written stub: Clarabel ships no type
information, and the strict typecheck needs it.

## Ownership: locally owned vs Rhiza-managed

This repo syncs its dev infrastructure from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The pinned
version lives in `.rhiza/template.yml` (`ref:`), and `/rhiza:update` re-applies
the template. **The authoritative, machine-generated list of synced files is the
`files:` block of `.rhiza/template.lock`** — when in doubt, consult it. The split
below summarises it.

### Locally owned — edit these freely

- `src/` — the library source, and `stubs/` alongside it
- `tests/` — the test suite
- `pyproject.toml` — project metadata, dependency groups, tool config, and the
  `[tool.rhiza-task]` table that configures the gates
- `README.md`, `introduction.md`, `CHANGELOG.md`, `mkdocs.yml`, `CLAUDE.md`
- `book/` — marimo notebooks and other project content
- `.rhiza/template.yml` — the template pin and the `profiles:`/`templates:`
  selection. The one file under `.rhiza/` this repo owns.
- `local.mk` — repo-specific make targets. The `Makefile` `-include`s it, and the
  template deliberately does not ignore it.

### Rhiza-managed — do NOT edit in place; fix upstream

These are overwritten by the next sync. To change one, open a PR against
`jebel-quant/rhiza` (or exclude the path in `.rhiza/template.yml`), then re-sync:

- `.github/workflows/rhiza_*.yml` — all CI/CD workflows
- `.github/` scaffolding — `dependabot.yml`, `release.yml`, rulesets,
  `secret_scanning.yml`
- `Makefile` — a 71-line shim that pins `RHIZA_TASK` and forwards every unmatched
  target to that CLI. Nothing goes below it; the next sync overwrites whatever was
  appended. Repo targets belong in `local.mk`.
- `.pre-commit-config.yaml`, `ruff.toml`, `pytest.ini`, `.bandit`,
  `.editorconfig`, `.python-version`, `cliff.toml` — tooling config
- `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, and the synced `docs/` pages

`SECURITY.md` in particular is synced here: an edit to it is drift the next sync
reverts, and the `check-managed-files` pre-commit hook refuses the commit.

## Quality gates

Since rhiza v1.4 the gates are tasks in the pinned `rhiza-task` CLI rather than
synced make fragments. Run them as bare `make <target>` (the shim forwards to
`uvx rhiza-task <task>`) — never call `.venv/bin/...` directly. `make help` lists
every task the pinned CLI knows, plus anything `local.mk` adds.

- `make install` — create the venv and sync dependencies
- `make fmt` — the pre-commit hooks over all files
- `make typecheck` — `ty` **and** `mypy --strict`; `[tool.rhiza-task]` sets
  `typechecker = "both"` because the default (`ty` alone) would quietly retire the
  `[tool.mypy]` block and the strict cross-check it exists for
- `make test` — the full pytest suite with the coverage gate
- `make coverage` — coverage measurement into `_tests/coverage.xml`
- `make docs-coverage` — interrogate docstring coverage
- `make deps` — deptry unused/missing dependency analysis
- `make security` — the bandit scan
- `make license` — fail on GPL/LGPL/AGPL
- `make rhiza-test` — the rhiza repository checks, from `pytest-rhiza==0.2.1`
- `make all` — everything above, in CI's order

`make benchmark` runs `tests/benchmarks/` and needs the `benchmark` group:
`uv sync --group benchmark`. Without it those tests error with
`fixture 'benchmark' not found`, which is expected rather than a regression.

Do not reach for `make mutation`. The task still exists in the CLI, but rhiza
v1.5.0 stopped offering mutation testing (Jebel-Quant/rhiza#1492) and the recipe
drives a mutmut 2.x CLI that mutmut 3 removed.

## Conventions

- The coverage gate is rhiza-task's default of 90%, not 100% — `[tool.rhiza-task]`
  deliberately leaves `coverage_fail_under` unset. Raise it there, not in
  `[tool.coverage.report]`, which the CLI outranks.
- `pytest.ini` runs with `--doctest-modules`, so a docstring example in `src/` is
  a test: keep them runnable.
- The per-test timeout is 60s (`pytest-timeout`). A test that needs longer is a
  test to reconsider.
- Three markers are declared: `stress`, `property`, `kaleido`. Use them rather
  than inventing new ones, and deselect with `-m "not stress"`.
- The `Programming Language :: Python :: 3.x` classifiers in `pyproject.toml`
  generate the CI test/typecheck matrices. Dropping one silently shrinks CI.

## Test layout

Tests mirror the source tree — `tests/core/`, `tests/risk/<family>/` — with a
handful of deliberate cross-cutting files that have no 1:1 source counterpart:

- `tests/risk/test_cvxpy_equivalence.py` — pins this engine's answers against the
  cvxpy formulation it replaced
- `tests/risk/test_properties.py` — hypothesis properties
- `tests/risk/test_reference_solvers.py`, `test_integration.py`,
  `test_failure_paths.py` — behaviour spanning several models
- `tests/test_architecture.py`, `tests/test_rhiza_packaging.py`,
  `tests/test_versions.py` — repo-level invariants

Shared fixtures live in `tests/conftest.py`; `tests/resources/stock_prices.csv` is
the common price fixture.
