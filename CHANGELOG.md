# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [1.6.1] - 2026-06-17

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 4 updates (#475)
- Chore(deps)(deps): bump starlette from 1.0.1 to 1.3.1 (#477)
- Chore(deps)(deps): bump python-multipart from 0.0.29 to 0.0.31 (#476)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#474)
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#473)

### Other Changes
- Sync Rhiza template v0.18.8 → v0.19.3 (#478)

## [1.6.0] - 2026-06-11

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#470)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#469)

### Other Changes
- Fix factor tracking error, dedupe solver code, harden test suite (#471)
- Bump version 1.5.1 → 1.6.0

## [1.5.1] - 2026-06-04

### New Features
- Promote cvx-linalg to core dep and use it in src
- Add profiles: github-project to template.yml

### Bug Fixes
- Pass secrets to reusable CI workflow
- Remove tests for generate-matrix/test jobs no longer in ci workflow
- Update workflow tests to match reusable workflow delegation pattern
- Restore .rhiza core bundle files lost during v0.14.0 sync
- Remove blank leading line from .python-version
- Satisfy rhiza pyproject.toml structure requirements

### Maintenance
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 4 updates
- Chore(deps)(deps): bump idna from 3.11 to 3.15
- Chore(deps)(deps): bump pymdown-extensions from 10.21.2 to 10.21.3
- Sync with rhiza v0.11.0
- Simplify marimo and book workflows to use reusable workflow (v0.11.2)
- Simplify codeql and sync workflows to use reusable workflow (v0.11.3)
- Simplify weekly and release workflows to use reusable workflow (v0.11.4)
- Update via rhiza
- Revert template-branch to v0.14.0 (v0.15.0 not yet published)
- Sync to rhiza v0.14.0
- Apply rhiza sync v0.15.1
- Add bundles directory
- Remove bundles directory placeholder
- Apply rhiza sync v0.15.2
- Chore(deps)(deps): bump the github-actions group with 7 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Apply rhiza sync v0.15.3
- Apply rhiza sync v0.17.0
- Apply rhiza sync v0.18.4
- Add pip dependabot entry for .rhiza/requirements
- Chore(deps)(deps): bump the github-actions group with 8 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 9 updates (#465)
- Chore(deps)(deps): bump starlette from 0.52.1 to 1.0.1 (#467)

### Other Changes
- Merge pull request #447 from cvxgrp/dependabot/github_actions/github-actions-bcb0c4251a
- Merge pull request #448 from cvxgrp/dependabot/uv/python-dependencies-c963573f50
- Merge pull request #449 from cvxgrp/dependabot/uv/idna-3.15
- Merge pull request #450 from cvxgrp/dependabot/uv/pymdown-extensions-10.21.3
- Update template.yml
- Workflows
- Merge pull request #451 from cvxgrp/rhiza11
- Remove generate-matrix job and add ci job
- Merge pull request #452 from cvxgrp/rhizaCI
- Merge pull request #453 from cvxgrp/rhizaCI
- Merge pull request #454 from cvxgrp/rhiza/26377674877
- Update template branch version to v0.15.0
- Merge pull request #457 from cvxgrp/rhiza_v0.15.2
- Merge pull request #458 from cvxgrp/dependabot/github_actions/github-actions-d4114beb95
- Merge pull request #459 from cvxgrp/dependabot/uv/python-dependencies-67199ecef6
- Merge pull request #460 from cvxgrp/rhiza_v0.15.3
- Merge pull request #461 from cvxgrp/rhiza_v0.17.0
- Merge pull request #462 from cvxgrp/rhiza_v0.18.4
- Merge pull request #464 from cvxgrp/dependabot/uv/python-dependencies-62474b3a21
- Merge branch 'main' into dependabot/github_actions/github-actions-f379237d3f
- Merge pull request #463 from cvxgrp/dependabot/github_actions/github-actions-f379237d3f
- Merge pull request #466 from cvxgrp/dependabot/uv/python-dependencies-6664ff0d93
- Bump version 1.5.0 → 1.5.1

## [1.5.0] - 2026-05-14

### New Features
- Replace in-repo linalg with cvx-linalg dependency
- Upgrade cvx-linalg to 0.3.0

### Bug Fixes
- Replace import cvxsimulator with import cvx.simulator
- Update broken documentation links in README.md
- Set MARIMO_FOLDER to book/marimo in .rhiza/.env
- Remove hardcoded sizes in marimo factor model example
- Update README factor model example for cvx-linalg 0.3.0
- Update factormodel marimo notebook for cvx-linalg 0.3.0

### Documentation
- Add coverage badge to README
- Expand notebooks and reports nav in mkdocs.yml
- Fix factor test fixture return type docstring

### Maintenance
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the python-dependencies group with 3 updates
- Chore(deps)(deps): bump pygments from 2.19.2 to 2.20.0
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump numpy in the python-dependencies group
- Sync rhiza template to v0.8.20
- Chore(deps)(deps): bump docker/login-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps-dev)(deps-dev): bump marimo from 0.22.4 to 0.23.0
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Update rhiza template to v0.10.5
- Sync with rhiza template v0.10.5
- Add mkdocs.yml following cvxcla structure
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 3 updates
- Bring test coverage to 100%
- Add pandas to dev dependencies
- Apply ruff import sorting
- Fix ruff import sorting in remaining files
- Remove simulator version test from test_versions.py

### Other Changes
- Merge pull request #395 from cvxgrp/dependabot/github_actions/github-actions-aa99a42152
- Merge branch 'main' into dependabot/uv/python-dependencies-ebeff0b55f
- Change template branch version to v0.8.5
- Sync
- Remove history
- Merge pull request #397 from cvxgrp/tschm-patch-110
- Merge branch 'main' into dependabot/uv/python-dependencies-ebeff0b55f
- Initial plan
- Merge pull request #398 from cvxgrp/copilot/sub-pr-396
- Merge pull request #396 from cvxgrp/dependabot/uv/python-dependencies-ebeff0b55f
- Merge pull request #401 from cvxgrp/dependabot/uv/python-dependencies-41f8a01e75
- Merge branch 'main' into dependabot/github_actions/github-actions-49563d2fa8
- Merge pull request #400 from cvxgrp/dependabot/github_actions/github-actions-49563d2fa8
- Merge pull request #404 from cvxgrp/dependabot/uv/python-dependencies-a6b05f733b
- Update template branch to v0.8.13 and modify templates
- Update template to v0.8.13: resolve conflicts and apply rhiza sync changes
- Merge pull request #405 from cvxgrp/tschm-patch-160
- Merge pull request #410 from cvxgrp/dependabot/uv/pygments-2.20.0
- Merge branch 'main' into dependabot/uv/python-dependencies-73d97db294
- Merge branch 'main' into dependabot/github_actions/github-actions-59b44d9e0a
- Merge pull request #407 from cvxgrp/dependabot/github_actions/github-actions-59b44d9e0a
- Merge branch 'main' into dependabot/uv/python-dependencies-73d97db294
- Merge pull request #408 from cvxgrp/dependabot/uv/python-dependencies-73d97db294
- Merge pull request #412 from cvxgrp/dependabot/uv/python-dependencies-1670832c3b
- Merge branch 'main' into dependabot/github_actions/github-actions-fd00acb19b
- Merge pull request #411 from cvxgrp/dependabot/github_actions/github-actions-fd00acb19b
- Update template branch to v0.8.20
- Merge pull request #413 from cvxgrp/tschm-patch-200
- Merge pull request #416 from cvxgrp/dependabot/uv/python-dependencies-111c77e3aa
- Merge branch 'main' into dependabot/github_actions/github-actions-cb5fd4910d
- Merge pull request #415 from cvxgrp/dependabot/github_actions/github-actions-cb5fd4910d
- Merge pull request #417 from cvxgrp/dependabot/uv/marimo-0.23.0
- Merge pull request #419 from cvxgrp/dependabot/github_actions/github-actions-e0563db078
- Merge pull request #420 from cvxgrp/dependabot/uv/python-dependencies-0f08c1daeb
- Merge pull request #423 from cvxgrp/dependabot/uv/python-dependencies-9f6c11c0d5
- Merge branch 'main' into dependabot/github_actions/github-actions-f3e34333ea
- Merge pull request #422 from cvxgrp/dependabot/github_actions/github-actions-f3e34333ea
- Merge pull request #424 from cvxgrp/dependabot/pip/dot-rhiza/requirements/python-dotenv-1.2.2
- Initial plan
- Merge pull request #426 from cvxgrp/copilot/update-action-workflow-configuration
- Merge pull request #427 from cvxgrp/dependabot/uv/python-dependencies-4e6ebe02a2
- Update README link to the correct cvxrisk page
- Merge pull request #428 from cvxgrp/update/rhiza-v0.10.5
- Delete .github/workflows/structure.yml
- Merge pull request #429 from cvxgrp/tschm-patch-1
- Delete .github/workflows/rhiza_benchmarks.yml
- Merge pull request #430 from cvxgrp/tschm-patch-2
- Initial plan
- Merge pull request #432 from cvxgrp/copilot/include-coverage-badge-readme
- Merge pull request #433 from cvxgrp/dependabot/github_actions/github-actions-937d73b4db
- Merge pull request #434 from cvxgrp/dependabot/uv/python-dependencies-189a206c4b
- Initial plan
- Move directly to Clarabel: remove cvxpy dependency, use direct solver
- Add clarifying comments on Clarabel solver tolerances in tests
- Add benchmarks measuring direct Clarabel vs cvxpy performance
- Add clarabel type stubs to fix ty typecheck errors
- Fix ruff N806/RUF059/RUF003 violations and stale doctests
- Update README Quick Start to show problem.status
- Fix second stale numpy scalar repr in parameter.py doctest
- Merge pull request #436 from cvxgrp/copilot/move-directly-to-clarabel
- Move Variable and Parameter into cvx.risk.core subpackage
- Relocate core subpackage to cvx.core (sibling of cvx.risk)
- Move Model and Bounds into cvx.core; revise docs to be domain-neutral
- Fix stale Variable import in README after move to cvx.core
- Merge pull request #437 from cvxgrp/core
- Move rand_cov into cvx.risk.linalg; remove cvx.risk.rand subpackage
- Promote linalg to cvx.linalg, a sibling of cvx.risk and cvx.core
- Update README import from cvx.risk.linalg to cvx.linalg
- Move solver logic into models, reorganize tests, flatten cvx.core imports
- Add solve_minrisk to Model base class to satisfy type checker
- Merge pull request #438 from cvxgrp/dependabot/github_actions/github-actions-8abaa2cbc6
- Merge pull request #439 from cvxgrp/dependabot/uv/python-dependencies-0ca5743cbe
- Initial plan
- Delete tests/linalg directory
- Merge pull request #441 from cvxgrp/copilot/bring-in-cvx-linalg-package
- Update pyproject.toml
- Potential fix for pull request finding
- Merge pull request #442 from cvxgrp/linalg_update
- Merge branch 'main' into tschm-patch-1
- Merge pull request #443 from cvxgrp/tschm-patch-1
- Remove solve_minrisk from Model base class
- Move cvx-linalg to dev, drop solve_minrisk from Model, add SolvableModel protocol
- Add cvx-linalg to marimo notebook inline dependencies
- Merge pull request #444 from cvxgrp/core3
- Replace pandas with polars/numpy in marimo notebooks
- Replace cvxsimulator with jquantstats in demo notebook
- Add pyarrow to demo.py inline dependencies
- Bump jquantstats minimum version to >=0.8.2
- Drop pandas/pyarrow from demo.py using jquantstats.exponential_cov
- Replace pandas with polars in test_factor.py fixture
- Merge pull request #445 from cvxgrp/jquantstats
- Merge branch 'main' into jquantstats
- Remove stale pyarrow/pandas/cvxsimulator references
- Add docstrings to _SolvableModel protocol for docs coverage
- Merge pull request #446 from cvxgrp/jquantstats
- Bump version 1.4.14 → 1.5.0

## [1.4.14] - 2026-03-01

### Other Changes
- Disable PyPI publishing in rhiza_release workflow
- Bump version 1.4.13 → 1.4.14

## [1.4.13] - 2026-02-28

### Bug Fixes
- Resolve ruff TRY003 and shellcheck SC2059 linting errors
- Add deptry package_module_name_map to suppress warnings
- Add missing __init__.py to enable doctest discovery
- Resolve mypy type errors for src layout and indexed assignments
- Resolve mypy strict type checking errors
- Support namespace packages in doctest discovery
- Rename random subpackage to rand to avoid stdlib shadowing

### Documentation
- Add detailed doctests across all risk model modules

### Dependencies
- *(deps)* Lock file maintenance (#318)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.14 (#321)
- *(deps)* Update softprops/action-gh-release action to v2.5.0
- *(deps)* Lock file maintenance (#323)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.16 (#325)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.8
- *(deps)* Lock file maintenance (#330)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.0 (#332)
- *(deps)* Lock file maintenance (#336)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.10 (#337)
- *(deps)* Lock file maintenance (#339)
- *(deps)* Update actions/checkout action to v6
- *(deps)* Update dependency astral-sh/uv to v0.9.20 (#342)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.20 (#341)
- *(deps)* Lock file maintenance (#344)
- *(deps)* Lock file maintenance (#346)
- *(deps)* Lock file maintenance (#349)
- *(deps)* Update dependency astral-sh/uv to v0.9.26 (#351)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.13
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.26 (#352)
- *(deps)* Lock file maintenance (#354)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.1 (#356)
- *(deps)* Update dependency astral-sh/uv to v0.9.27
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.27
- *(deps)* Lock file maintenance (#359)
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.25
- *(deps)* Update github/codeql-action action to v4.32.1
- *(deps)* Lock file maintenance (#364)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.30
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.11
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#373)
- *(deps)* Update dependency astral-sh/uv to v0.10.3 (#375)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.11
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.0
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.3
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2
- *(deps)* Update actions/download-artifact action to v7
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.4
- *(deps)* Update dependency astral-sh/uv to v0.10.4
- *(deps)* Update actions/download-artifact action to v7
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2
- *(deps)* Update github/codeql-action action to v4.32.4 (#385)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.2 (#386)
- *(deps)* Lock file maintenance (#387)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.3
- *(deps)* Update dependency astral-sh/uv to v0.10.6 (#390)

### Maintenance
- Sync template files
- Sync template files
- Import rhiza templates
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Update via rhiza
- Remove dotenv dependency from doctest test file
- Chore(deps)(deps): bump the github-actions group with 2 updates
- Update via rhiza

### Other Changes
- Merge pull request #319 from cvxgrp/template-updates
- Merge pull request #322 from cvxgrp/renovate/softprops-action-gh-release-2.x
- Merge pull request #326 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Add pytest.ini to template.yml
- Delete tests/test_docstrings.py
- Merge pull request #327 from cvxgrp/tschm-patch-1
- Delete tests/test_makefile.py
- Merge pull request #328 from cvxgrp/tschm-patch-2
- Delete tests/test_readme.py
- Merge pull request #329 from cvxgrp/tschm-patch-3
- Update template repository in template.yml
- Merge pull request #331 from cvxgrp/template-updates
- Update LICENSE
- Update template.yml
- Delete .github/scripts/build-extras.sh
- Delete .github/workflows/_devcontainer.yml
- Rhiza
- Delete tests/test_config_templates directory
- Delete .github/scripts/sync.sh
- Initial plan
- Add PEP 723 script headers to all marimo notebooks
- Add clarabel==0.11.1 to all script headers
- Update factormodel.py
- Merge pull request #334 from cvxgrp/copilot/create-script-headers-notebooks
- Rhiza
- Add CodeQL analysis workflow configuration
- Migrate
- Merge pull request #338 from cvxgrp/renovate/actions-checkout-6.x
- Remove dependabot
- Delete .github/workflows/codeql.yml
- Merge pull request #343 from cvxgrp/tschm-patch-1
- Merge pull request #340 from cvxgrp/rhiza/20561720679
- Rhiza.env
- Dependencies
- Merge branch 'main' into rhiza/20701519134
- Missing plotly dependency
- Merge pull request #345 from cvxgrp/rhiza/20701519134
- Merge pull request #347 from cvxgrp/rhiza/20904450919
- Delete .rhiza.env
- Merge pull request #348 from cvxgrp/tschm-patch-1
- Merge pull request #353 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Github.mk in
- Delete src/cvx/__init__.py
- Update SOURCE_FOLDER path in .env file
- Add .rhiza/.env to exclusion list in template.yml
- Merge pull request #355 from cvxgrp/rhiza/21342296039
- Merge pull request #357 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #358 from cvxgrp/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #360 from cvxgrp/rhiza/21573251026
- Merge pull request #361 from cvxgrp/rhiza/21573251026
- Merge pull request #363 from cvxgrp/renovate/abravalheri-validate-pyproject-0.x
- Merge pull request #362 from cvxgrp/renovate/github-codeql-action-4.x
- Remove 'tests' from included files in template.yml
- Merge pull request #366 from cvxgrp/tschm-patch-1
- Delete .github/workflows/rhiza_benchmarks.yml
- Merge pull request #367 from cvxgrp/tschm-patch-2
- Merge pull request #368 from cvxgrp/dependabot/github_actions/github-actions-b535e03c44
- Merge pull request #369 from cvxgrp/renovate/ghcr.io-astral-sh-uv-0.x
- Update template branch and include new templates
- Sync
- Fmt
- Merge branch 'main' into renovate/rhysd-actionlint-1.x
- Merge branch 'main' into renovate/lock-file-maintenance
- Merge pull request #372 from cvxgrp/renovate/lock-file-maintenance
- Merge branch 'main' into renovate/rhysd-actionlint-1.x
- Merge pull request #371 from cvxgrp/renovate/rhysd-actionlint-1.x
- Merge pull request #374 from cvxgrp/rhiza/22046094274
- Merge pull request #378 from cvxgrp/renovate/rhysd-actionlint-1.x
- Merge branch 'main' into renovate/astral-sh-uv-pre-commit-0.x
- Merge branch 'main' into renovate/python-jsonschema-check-jsonschema-0.x
- Merge pull request #377 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge branch 'main' into renovate/astral-sh-uv-pre-commit-0.x
- Merge branch 'main' into renovate/major-github-artifact-actions
- Merge pull request #380 from cvxgrp/renovate/major-github-artifact-actions
- Merge branch 'main' into renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #376 from cvxgrp/renovate/astral-sh-uv-pre-commit-0.x
- Merge branch 'main' into renovate/jebel-quant-rhiza-0.x
- Sync
- Fix type checking issues
- Merge pull request #379 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Merge pull request #382 from cvxgrp/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #381 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #383 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge branch 'main' into renovate/major-github-artifact-actions
- Merge pull request #384 from cvxgrp/renovate/major-github-artifact-actions
- Conftest
- Sync
- Release
- Merge pull request #391 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Merge branch 'main' into renovate/jebel-quant-rhiza-0.x
- Bump version to 1.4.12 in pyproject.toml
- Sync
- Merge pull request #392 from cvxgrp/tschm-patch-1
- Merge branch 'main' into renovate/jebel-quant-rhiza-0.x
- Merge pull request #393 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Bump version 1.4.12 → 1.4.13

## [1.4.12] - 2025-11-30

### Bug Fixes
- *(deps)* Update dependency python-dotenv to v1.2.1
- Fixing notebook imports
- Fixing notebook imports
- Fixing notebook imports
- Fix deptry

### Dependencies
- *(deps)* Lock file maintenance (#310)

### Maintenance
- Sync config files from .config-templates (#38) (#292)
- Sync config files from .config-templates (#295)
- Sync template files (#300)
- Sync template files (#305)
- Sync template files (#306)
- Sync template files (#307)
- Sync template files
- Sync template files
- Sync template files

### Other Changes
- Renovate (#288)
- Using the .config-templates (#289)
- Remove all init files from tests (#291)
- Update from tschm/cvxrisk (#293)
- Sync (#294)
- Sync (#296)
- Updates from tschm (#299)
- Update from tschm (#301)
- 68 remove spicy and scikit learn (#302)
- 68 remove spicy and scikit learn (#303)
- Delete tests/test_docs.py
- Update from tschm (#308)
- Merge pull request #309 from cvxgrp/template-updates
- Fix README
- Install clarabel as dev dependency
- Explicit solver in README
- Merge pull request #311 from cvxgrp/template-updates
- Merge pull request #312 from cvxgrp/renovate/python-dotenv-1.x
- Merge pull request #313 from cvxgrp/template-updates
- Remove traces of task
- Move to src folder and cvx/risk strucuture
- Merge pull request #314 from cvxgrp/sep
- Initial plan
- Add Google-style docstrings with code examples and doctest
- Merge pull request #316 from cvxgrp/copilot/add-google-style-docstrings
- Potential fix for code scanning alert no. 11: Workflow does not contain permissions
- Merge pull request #317 from cvxgrp/alert-autofix-11
- Update template.yml
- Remove devcontainer
- Remove docker.yml
- Update template.yml
- Remove copyright
- Mock _devcontainer.yml
- _devcontainer back
- Remove GitHub Codespaces badge

## [1.4.11] - 2025-07-03

### Other Changes
- Ignore __marimo__
- Update book.yml
- Update marimo.yml (#286)
- Remove marimo workshop
- Makefile
- Merge fork back into (#287)

## [1.4.10] - 2025-06-10

### Maintenance
- Test all Marimo notebooks
- Test versions (#277)
- Build cvxrisk folder

### Other Changes
- Tilting with to_numpy
- Assert in factor
- Assert in factor
- Update release.yml
- Update .pre-commit-config.yaml
- Update pyproject.toml (#278)
- Uvx for fmt
- Cvxr (#282)
- Updates (#283)
- 284 revisit notebooks (#285)

## [1.4.9] - 2025-06-07

### Maintenance
- Testing the README
- Testing the core functionality
- Test all Marimo notebooks
- Test all Marimo notebooks
- Test all Marimo notebooks

### Other Changes
- Micropip and python >=3.12
- Remove tests for 3.11 and 3.10

## [1.4.8] - 2025-06-07

### Other Changes
- Versions in all submodules
- Versions in all submodules
- Micropip installs
- Micropip installs
- Polars dependency
- All notebooks run

## [1.4.7] - 2025-06-06

### Bug Fixes
- Fixing test in book (#275)

### Other Changes
- Update __init__.py (#271)
- Update book.yml (#272)
- Update book.yml
- Workflows
- Fmt
- Update .pre-commit-config.yaml (#273)
- Update README.md
- Wasm
- Wasm
- Wasm

## [1.4.6] - 2025-05-26

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.11 (#262)
- *(deps)* Update pre-commit hook asottile/pyupgrade to v3.20.0 (#268)
- *(deps)* Lock file maintenance (#270)

### Maintenance
- Test portfolio in cvx risk

### Other Changes
- Update README.md
- Potential fix for code scanning alert no. 2: Workflow does not contain permissions (#261)
- Update README.md
- Potential fix for code scanning alert no. 1: Workflow does not contain permissions (#263)
- Fmt
- Extra (#264)
- Install clarabel and marimo
- Merge pull request #265
- Devcon (#266)
- Init (#267)
- Makebranch (#269)

## [1.4.5] - 2025-05-22

### Other Changes
- Tilting formatting
- Google style for pdoc
- Numpy style for pdoc
- Numpy style for pdoc
- Numpy style for pdoc
- Numpy style for pdoc
- Hints (#260)
- Typehints
- Fmt
- Fmt
- Fmt
- Fmt
- Introduction

## [1.4.4] - 2025-05-22

### Dependencies
- *(deps)* Lock file maintenance (#259)

### Other Changes
- Tilting (#258)
- Update README.md
- Tilting move
- Move tilting

## [1.4.3] - 2025-05-22

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.10 (#251)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.32.0 (#252)
- *(deps)* Lock file maintenance (#254)
- *(deps)* Update pre-commit hook igorshubovych/markdownlint-cli to v0.45.0 (#253)

### Other Changes
- 255 add constraints to min variance portfolio (#256)

## [1.4.2] - 2025-05-21

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.2 (#237)
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.24.1 (#236)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.32.1 (#239)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.31.0 (#238)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.31.1 (#240)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.3 (#241)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.4 (#242)
- *(deps)* Lock file maintenance (#243)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.5 (#244)
- *(deps)* Lock file maintenance (#245)
- *(deps)* Lock file maintenance (#247)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.0 (#248)

### Maintenance
- *(config)* Migrate config .github/renovate.json (#246)

### Other Changes
- Update book.yml
- Update ci.yml
- Update pre-commit.yml
- Update release.yml
- Update book.yml
- Update pre-commit hooks
- Update dependabot.yml
- Update pre-commit hooks
- Fmt
- Update devcontainer.json
- Explicit checkouts in workflows
- Explicit workflow version 2.0.0
- Update pre-commit.yml
- Bump cvxgrp/.github from 2.0.0 to 2.0.3
- [pre-commit.ci] pre-commit autoupdate
- Update .pre-commit-config.yaml (#173)
- Bump cvxgrp/.github from 2.0.3 to 2.0.6 (#174)
- [pre-commit.ci] pre-commit autoupdate (#175)
- Update release.yml (#176)
- Update ci.yml
- Bump cvxgrp/.github from 2.0.6 to 2.0.8 (#177)
- [pre-commit.ci] pre-commit autoupdate (#178)
- Update book.yml (#180)
- Configure Renovate (#179)
- Update pre-commit.yml (#181)
- Update mcr.microsoft.com/devcontainers/python Docker tag to v3.13 (#184)
- Update cvxgrp/.github action to v2.0.13 (#183)
- Update renovate.json (#186)
- Update release.yml (#182)
- Update pre-commit hook crate-ci/typos to v1 (#189)
- Update cvxgrp/.github action to v2.0.13 (#187)
- Update pre-commit hooks (#188)
- Update .pre-commit-config.yaml (#191)
- Lock file maintenance (#190)
- Update cvxgrp/.github action to v2.0.17 (#192)
- Introduce Makefile (#194)
- Update ci.yml (#196)
- Update release.yml (#195)
- Update book construction
- Update cvxgrp/.github action to v2.1.1 (#193)
- Update release.yml (#197)
- Update cvxgrp/.github action to v2.1.2 (#198)
- Change uv/environment to environment/uv
- Update pre-commit.yml (#200)
- Update release.yml (#201)
- Update ci.yml (#202)
- Update book.yml (#203)
- Update cvxgrp/.github action to v2.2.1 (#199)
- Lock file maintenance (#208)
- Lock file maintenance (#209)
- Lock file maintenance (#210)
- Update cvxgrp/.github action to v2.2.3 (#211)
- Update ci.yml (#212)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.5 (#213)
- Lock file maintenance (#214)
- Update cvxgrp/.github action to v2.2.4 (#215)
- Lock file maintenance (#216)
- Update cvxgrp/.github action to v2.2.5 (#217)
- Lock file maintenance (#218)
- Update pre-commit hooks (#219)
- Lock file maintenance (#220)
- Update cvxgrp/.github action to v2.2.6 (#221)
- Lock file maintenance (#223)
- Update pre-commit hooks (#222)
- Update cvxgrp/.github action to v2.2.7 (#224)
- Lock file maintenance (#225)
- Lock file maintenance (#226)
- Update cvxgrp/.github action to v2.2.8 (#227)
- Lock file maintenance (#228)
- Lock file maintenance (#229)
- Lock file maintenance (#230)
- Lock file maintenance (#231)
- Lock file maintenance (#232)
- Update pyproject.toml (#234)
- Update renovate.json (#235)
- Update renovate.json
- Update README.md
- Make file point to tests rather than src/tests (#250)
- Release

## [1.4.1] - 2024-12-18

### Other Changes
- Expose pdoc documentation
- Construct pdoc
- Comments for pdoc?
- Address update of exposure in exposure matrix
- Better comments

## [1.4.0] - 2024-12-17

### Other Changes
- Contributing
- Index replaced by link to README
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book
- Maths in book

## [1.3.4] - 2024-12-17

### Other Changes
- Project urls

## [1.3.3] - 2024-12-17

### Other Changes
- Devcontainer
- Devcontainer
- Update pyproject.toml
- Update .pre-commit-config.yaml
- Label to codespaces

## [1.3.2] - 2024-12-16

### Maintenance
- Build
- Test and coverage
- Test and coverage

### Other Changes
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Pre-commit simplified with central action
- Update book.yml
- Towards Marimo
- Marimo
- Demo.html
- Export to html
- Export to html
- Export to html
- Export to html
- Marimo_html
- Export to html
- Export to html
- Export to html
- Copy into artifacts/marimo
- Copy into artifacts/marimo
- Copy into artifacts/marimo
- Copy into artifacts/marimo
- Copy into artifacts/marimo
- Copy into artifacts/marimo
- Setup environment for marimo
- Install marimo
- Install marimo
- Delete hard html files
- Remove Jupyter
- Correct marimo scripts
- Move marimo folder into book
- Move marimo folder into book
- Export also md files
- Sphinx and deptry
- Jupyter ci/cd
- Update index.md
- README for book
- README for book
- README for book
- README for book
- README for book
- README for book
- README for book
- README for book
- README for book
- README for book
- Remove uv.lock
- Remove uv.lock
- Remove uv.lock
- Remove uv.lock
- Remove uv.lock
- Remove uv.lock
- Marimo extern
- Marimo extern
- Marimo extern
- Lock file
- Remove requirements.txt
- Marimo extern
- Update Makefile
- Marimo extern
- Marimo extern
- Remove stock prices
- Create taskfile.yml
- Env for remote taskfiles
- Taskfile
- Makefile2
- Remove Makefile
- Marimo extern

## [1.3.1] - 2024-11-27

### Other Changes
- Update README for uv
- Makefile installing uv
- Update pyproject.toml
- Update README.md
- Update book.yml
- Update release.yml

## [1.3.0] - 2024-11-27

### Maintenance
- Testing
- Ci
- Ci

### Other Changes
- Update pyproject.toml
- Python-version as required by uv
- Lock files
- Dependencies
- Towards local Makefile
- Towards local Makefile
- Target py version
- Dev dependencies
- Update __init__.py
- Update __init__.py
- Update __init__.py
- Update __init__.py
- Update __init__.py
- Update __init__.py
- Makefile with uv coverage
- Cvx simulator in
- Update pyproject.toml
- Deptry with uv
- Ignore cvxpy missing
- Ignore cvxpy missing
- Remove jupyter dep
- Remove jupyter
- Remove jupyter
- Remove jupyter
- Remove jupyter
- Remove jupyter
- Remove jupyter
- Remove jupyter
- Remove jupyter
- Release with uv

## [1.2.1] - 2024-11-27

### Bug Fixes
- Fixing notebooks

### Other Changes
- Bump cvxsimulator from 1.1.1 to 1.2.0
- Bump cvxsimulator from 1.2.0 to 1.2.1

## [1.2.0] - 2024-11-13

### Other Changes
- Bump cvxpy-base from 1.5.3 to 1.6.0
- Bump cvxsimulator from 1.0.1 to 1.1.1
- Update ci.yml
- Update pyproject.toml
- Update ci.yml
- Lock

## [1.1.1] - 2024-11-09

### Bug Fixes
- Fix test

### Other Changes
- Bump pytest-cov from 5.0.0 to 6.0.0
- Remove logo?

## [1.1.0] - 2024-10-27

### Other Changes
- Bump cvxsimulator from 0.9.19 to 1.0.0
- Bump pre-commit from 4.0.0 to 4.0.1
- Update pre-commit.yml
- Update __init__.py
- Update pyproject.toml
- Update cvar.py
- Numpy explicit dependency
- Remove clarabel
- Update test_valid.py
- Update sample.py
- Clarabel as test dependency
- Update factor.py
- Update bounds.py
- Moving linalg into risk to reduce level of relative imports
- Move tests
- Move tests
- Fmt

## [1.0.0] - 2024-10-07

### Maintenance
- Build and publish instead of release

### Other Changes
- Update Makefile
- Update Makefile (#44)
- Update Makefile (#45)
- Update Makefile
- Bump cvxsimulator from 0.7.3 to 0.7.5 (#46)
- Coverage with install prior
- Artifacts
- Checking poetry
- Update .pre-commit-config.yaml
- Update in lock
- Update ci.yml
- Update pre-commit.yml
- Revisit workflows
- Verbose pre-commit
- Jupyter in readme
- Update .pre-commit-config.yaml
- Update .pre-commit-config.yaml
- Fmt
- Deactive lock file check
- License
- Lock file update
- Update pre-commit.yml
- Bump pytest from 7.4.0 to 7.4.1 (#47)
- Lock fmt
- Fmt
- Make fmt update
- Update .pre-commit-config.yaml
- Bump pytest from 7.4.1 to 7.4.2 (#48)
- Update README.md
- Conduct
- Bump loguru from 0.7.1 to 0.7.2 (#50)
- Bump plotly from 5.16.1 to 5.17.0 (#49)
- Update pyproject.toml
- Update Makefile
- Update conf.py
- Bump pandas from 2.1.0 to 2.1.1 (#53)
- Update conf.py
- Bump scikit-learn from 1.3.0 to 1.3.1 (#52)
- Bump cvxcovariance from 0.1.1 to 0.1.2 (#51)
- Book construction
- Bump urllib3 from 2.0.4 to 2.0.6 (#54)
- Bump pillow from 10.0.0 to 10.0.1 (#56)
- Bump cvxpy from 1.3.2 to 1.4.1 (#57)
- Bump urllib3 from 2.0.6 to 2.0.7 (#58)
- Bump plotly from 5.17.0 to 5.18.0 (#60)
- Bump pandas from 2.1.1 to 2.1.2 (#61)
- Bump pytest from 7.4.2 to 7.4.3 (#62)
- Bump scikit-learn from 1.3.1 to 1.3.2 (#59)
- Bump pandas from 2.1.2 to 2.1.3
- Bump cvxsimulator from 0.7.6 to 0.7.8
- Bump cvxsimulator from 0.7.8 to 0.8.11
- Bump pandas from 2.1.3 to 2.1.4
- Bump cvxsimulator from 0.8.11 to 0.8.17
- Bump cvxsimulator from 0.8.17 to 0.9.1
- Bump pytest from 7.4.3 to 7.4.4
- Updates
- Bump cvxsimulator from 0.5.0 to 0.9.3
- Bump cvxsimulator from 0.9.3 to 0.9.10
- Bump cvxpy from 1.4.1 to 1.4.2
- Bump pandas from 2.1.4 to 2.2.0
- Bump scikit-learn from 1.3.2 to 1.4.0
- Bump cvxsimulator from 0.9.10 to 0.9.11
- Bump pytest from 7.4.4 to 8.0.0
- Bump cvxsimulator from 0.9.11 to 0.9.13
- Bump pre-commit from 3.6.0 to 3.6.1
- Update dependabot.yml
- Bump pre-commit/action from 3.0.0 to 3.0.1
- Bump pre-commit from 3.6.1 to 3.6.2
- Bump pytest from 8.0.0 to 8.0.1
- Bump plotly from 5.18.0 to 5.19.0
- Bump cvxsimulator from 0.9.13 to 0.9.14
- Bump scikit-learn from 1.4.0 to 1.4.1.post1
- Bump pytest from 8.0.1 to 8.0.2
- Bump pandas from 2.2.0 to 2.2.1
- Bump cvxsimulator from 0.9.14 to 0.9.15
- Bump cvxsimulator from 0.9.15 to 0.9.16
- Bump pytest from 8.0.2 to 8.1.0
- Bump pytest from 8.1.0 to 8.1.1
- Bump plotly from 5.19.0 to 5.20.0
- Bump pre-commit from 3.6.2 to 3.7.0
- Bump pytest-cov from 4.1.0 to 5.0.0
- Bump cvxsimulator from 0.9.16 to 0.9.17
- Update pyproject.toml
- Bump cvxcovariance from 0.1.1 to 0.1.4
- Bump pandas from 2.2.1 to 2.2.2
- Bump scikit-learn from 1.4.1.post1 to 1.4.2
- Bump plotly from 5.20.0 to 5.21.0
- Bump cvxpy from 1.4.2 to 1.4.3
- Bump pytest from 8.1.1 to 8.2.0
- Bump plotly from 5.21.0 to 5.22.0
- Bump pre-commit from 3.7.0 to 3.7.1
- Bump cvxsimulator from 0.9.17 to 0.9.18
- Bump cvxpy from 1.4.3 to 1.5.1
- Bump pytest from 8.2.0 to 8.2.1
- Bump scikit-learn from 1.4.2 to 1.5.0
- Bump pytest from 8.2.1 to 8.2.2
- Update pre-commit.yml
- Update book.yml
- Dependencies
- Remove cvxcovariance
- Update ci.yml
- Bump cvxpy-base from 1.5.1 to 1.5.2
- Update README.md
- Bump cvxsimulator from 0.9.18 to 0.9.19
- Bump scikit-learn from 1.5.0 to 1.5.1
- Bump pytest from 8.2.2 to 8.3.1
- Bump pre-commit from 3.7.1 to 3.8.0
- Bump plotly from 5.22.0 to 5.23.0
- Bump pytest from 8.3.1 to 8.3.2
- Bump cvxpy-base from 1.5.2 to 1.5.3
- Bump plotly from 5.23.0 to 5.24.0
- Bump pandas from 2.2.2 to 2.2.3
- Bump scikit-learn from 1.5.1 to 1.5.2
- Bump pytest from 8.3.2 to 8.3.3
- Bump plotly from 5.24.0 to 5.24.1
- Bump pre-commit from 3.8.0 to 4.0.0

## [0.1.4] - 2023-08-18

### Other Changes
- Update release.yml

## [0.1.3] - 2023-08-17

### Other Changes
- Update release.yml

## [0.0.9] - 2023-08-17

### Other Changes
- Update release.yml

## [0.0.8] - 2023-08-17

### Other Changes
- Remove obsolete minvar implementation
- Makefile
- Update book.yml
- Linting (#37)
- Cleaning LICENSE file
- Update index.md
- No isort needed
- Poetry updates
- Update Makefile
- Create dependabot.yml
- Bump tornado from 6.3.2 to 6.3.3 (#38)
- Bump plotly from 5.15.0 to 5.16.0 (#42)
- Deps updates
- Deps updates
- Update deps
- Remove book

## [0.0.7] - 2023-06-21

### Other Changes
- Remove duplicated test resource
- 28 revisit comments on changing dimension (#30)
- Bounds isolated (#31)
- Bounds isolated (#32)
- Introduce bounds for factor weights (#33)
- 34 fix doc index (#35)

## [0.0.6] - 2023-06-21

### Other Changes
- Rearranging stuff
- Rename RiskModel to Model

## [0.0.5] - 2023-06-21

### Maintenance
- Refactor RiskModel
- Refactor RiskModel

### Other Changes
- Notebooks

## [0.0.4] - 2023-06-20

### Bug Fixes
- Fixing broken dgp test

### Maintenance
- Test for spd-ness
- Test linalg
- Testing factors
- Test volatility sample
- Test volatility factor

### Other Changes
- Update README.md
- Correct README
- Demo notebook
- Demo notebook
- Obsolete utils removed
- Cholesky
- Cvar
- Cvar
- Cvar
- Sample test
- Random cov matrix
- Introduce valid for covariance matrices
- Only one SampleCovariance
- Using scipy cholesky
- Notebooks
- Revisiting factor models
- Factor investment dpp
- Notebook
- Update_data approach
- Notebooks@
- Cosmetics
- Factor model
- Using ghost assets
- Notebooks

## [0.0.3] - 2023-06-14

### Maintenance
- Test raising error for non psd cov matrix (#22)

### Other Changes
- Update README.md
- Make the RiskModel importable
- Obsolete utils removed
- Tschm patch 1 (#23)

## [0.0.2] - 2023-06-14

### Other Changes
- Update ci.yml
- Update book.yml
- Update reports.md
- Update api.md
- Pre-commit
- Poetry
- Update pyproject.toml
- Pre-commit
- Update _config.yml
- Using parameter correctly
- Towards a README (#21)

## [0.0.1] - 2023-06-09

### Bug Fixes
- Fixing notebooks
- Fixing notebooks (#17)

### Other Changes
- Initial commit
- Update create_kernel.sh
- Dependencies
- README.md
- Towards testing
- Towards testing
- Import scikit-learn
- Towards tests
- Towards tests
- Merge pull request #2 from cvxgrp/1-bring-over-code
- Update _config.yml
- Update conf.py
- Correct index
- Kernel for book
- Remove obsolete notebooks
- Notebooks
- Update README.md
- Headers in notebooks
- Make notebook work
- Pre-commit hooks
- Merge pull request #4 from cvxgrp/3-bring-in-pre-commit-hooks
- Coverage
- Coverage
- Coverage
- Update and rename coverage.sh to coverage.s
- Rename coverage.s to coverage.sh
- Using parameter correctly
- Updates
- Move random to notebooks
- Merge pull request #6 from cvxgrp/5-use-kwargs-in-abstract-base-class
- Merge pull request #8 from cvxgrp/7-fix-broken-book
- Precommit
- Update ci.yml
- Revisiting
- Cvxcovariance in
- Merge pull request #13 from cvxgrp/11-link-to-cvxcovariance
- Remove variance matrix
- Move product out
- Merge pull request #14 from cvxgrp/9-remove-the-product-versions
- 12 include cvar (#16)
- Update factormodel.ipynb
- Dependencies (#19)

<!-- generated by git-cliff -->
