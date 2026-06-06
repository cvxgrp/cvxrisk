## [1.5.1] - 2026-06-04

### 🚀 Features

- Promote cvx-linalg to core dep and use it in src
- Add profiles: github-project to template.yml

### 🐛 Bug Fixes

- Pass secrets to reusable CI workflow
- Remove tests for generate-matrix/test jobs no longer in ci workflow
- Update workflow tests to match reusable workflow delegation pattern
- Restore .rhiza core bundle files lost during v0.14.0 sync
- Remove blank leading line from .python-version
- Satisfy rhiza pyproject.toml structure requirements

### 💼 Other

- Bump version 1.5.0 → 1.5.1

### ⚙️ Miscellaneous Tasks

- Bump rhiza template to v0.11.0
- Sync with rhiza v0.11.0
- Bump rhiza CI workflow to v0.11.1 (adds workflow_call support)
- Simplify marimo and book workflows to use reusable workflow (v0.11.2)
- Simplify codeql and sync workflows to use reusable workflow (v0.11.3)
- Simplify weekly and release workflows to use reusable workflow (v0.11.4)
- Update via rhiza
- Bump rhiza version to 0.16.0
- Revert template-branch to v0.14.0 (v0.15.0 not yet published)
- Sync to rhiza v0.14.0
- Bump rhiza to v0.15.1
- Apply rhiza sync v0.15.1
- Bump rhiza to v0.15.2
- Add bundles directory
- Remove bundles directory placeholder
- Apply rhiza sync v0.15.2
- Bump rhiza to v0.15.3
- Apply rhiza sync v0.15.3
- Bump rhiza to v0.17.0
- Apply rhiza sync v0.17.0
- Bump rhiza to v0.18.4
- Apply rhiza sync v0.18.4
- Add pip dependabot entry for .rhiza/requirements
## [1.5.0] - 2026-05-14

### 🚀 Features

- Replace in-repo linalg with cvx-linalg dependency
- Upgrade cvx-linalg to 0.3.0

### 🐛 Bug Fixes

- Replace import cvxsimulator with import cvx.simulator
- Update broken documentation links in README.md
- Set MARIMO_FOLDER to book/marimo in .rhiza/.env
- Remove hardcoded sizes in marimo factor model example
- Update README factor model example for cvx-linalg 0.3.0
- Update factormodel marimo notebook for cvx-linalg 0.3.0

### 💼 Other

- Bump version 1.4.14 → 1.5.0

### 🚜 Refactor

- Remove simulator version test from test_versions.py

### 📚 Documentation

- Add coverage badge to README
- Expand notebooks and reports nav in mkdocs.yml
- Fix factor test fixture return type docstring

### 🎨 Styling

- Apply ruff import sorting
- Fix ruff import sorting in remaining files

### 🧪 Testing

- Bring test coverage to 100%

### ⚙️ Miscellaneous Tasks

- Sync rhiza template to v0.8.20
- Update rhiza template to v0.10.5
- Sync with rhiza template v0.10.5
- Add mkdocs.yml following cvxcla structure
- Add pandas to dev dependencies
## [1.4.14] - 2026-03-01

### 💼 Other

- Disable PyPI publishing in rhiza_release workflow
- Bump version 1.4.13 → 1.4.14
## [1.4.13] - 2026-02-28

### 🐛 Bug Fixes

- Resolve ruff TRY003 and shellcheck SC2059 linting errors
- Add deptry package_module_name_map to suppress warnings
- Add missing __init__.py to enable doctest discovery
- Resolve mypy type errors for src layout and indexed assignments
- Resolve mypy strict type checking errors
- Support namespace packages in doctest discovery
- Rename random subpackage to rand to avoid stdlib shadowing

### 💼 Other

- Bump version 1.4.12 → 1.4.13

### 🚜 Refactor

- Remove dotenv dependency from doctest test file

### 📚 Documentation

- Add detailed doctests across all risk model modules

### ⚙️ Miscellaneous Tasks

- Sync template files
- Sync template files
- Import rhiza templates
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
## [1.4.12] - 2025-11-30

### 🐛 Bug Fixes

- *(deps)* Update dependency python-dotenv to v1.2.1

### ⚙️ Miscellaneous Tasks

- Sync config files from .config-templates (#38) (#292)
- Sync config files from .config-templates (#295)
- Sync template files (#300)
- Sync template files (#305)
- Sync template files (#306)
- Sync template files (#307)
- Sync template files
- Sync template files
- Sync template files
## [1.4.2] - 2025-05-21

### ⚙️ Miscellaneous Tasks

- *(config)* Migrate config .github/renovate.json (#246)
## [0.0.1] - 2023-06-09
