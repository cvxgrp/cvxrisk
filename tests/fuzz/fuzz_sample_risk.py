"""Fuzz the cvxrisk sample-covariance risk model against arbitrary matrices.

``SampleCovariance`` ingests a covariance matrix via ``update`` and computes the
portfolio risk (``||R @ w||``) via ``estimate``. Neither should crash with an
unexpected exception on adversarial input — degenerate, non-finite or
non-positive-definite covariance matrices should produce a result or raise a
documented error (cvx.linalg raises ValueError/TypeError subclasses; numpy may
raise ``LinAlgError``). This harness exercises that contract with coverage-guided
input.

Run locally:
    pip install atheris numpy scipy clarabel cvx-linalg
    python tests/fuzz/fuzz_sample_risk.py -atheris_runs=20000

Run in ClusterFuzzLite: this file is built by .clusterfuzzlite/build.sh.
"""

from __future__ import annotations

import contextlib
import sys

import atheris

# Pre-import the heavy native dependencies OUTSIDE the instrumentation block.
# Atheris's import hook miscompiles parts of Rust/C extensions' Python machinery
# (e.g. clarabel), so we let them load uninstrumented and instrument only the
# first-party package under test.
import clarabel  # noqa: F401  # pre-imported uninstrumented
import numpy as np
import scipy.sparse  # noqa: F401  # pre-imported uninstrumented

with atheris.instrument_imports():
    from cvx.risk.sample import SampleCovariance

_ALLOWED = (ValueError, TypeError, np.linalg.LinAlgError)


def test_one_input(data: bytes) -> None:
    """Build, update and estimate a SampleCovariance model from fuzzed data."""
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(1, 5)
    cov = np.array([fdp.ConsumeFloat() for _ in range(n * n)], dtype=np.float64).reshape(n, n)
    weights = np.array([fdp.ConsumeFloat() for _ in range(n)], dtype=np.float64)

    model = SampleCovariance(num=n)
    with contextlib.suppress(_ALLOWED):
        model.update(cov=cov, lower_assets=np.zeros(n), upper_assets=np.ones(n))
        model.estimate(weights)


def main() -> None:
    """Run the Atheris fuzz loop."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
