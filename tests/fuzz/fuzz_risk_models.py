"""Atheris fuzz harness exercising the cvxrisk risk models on arbitrary numeric input.

The harness builds small, fuzzer-derived covariance / returns / exposure matrices
and drives each risk model end-to-end (construct, update, estimate, and — for the
sample model — solve a minimum-risk problem). Input that is merely malformed or
ill-conditioned is rejected by cvxrisk with a documented exception and is treated
as expected; anything else propagates so the fuzzer can flag it.

Run locally (requires atheris, which builds on Linux):
    uv run --with atheris python tests/fuzz/fuzz_risk_models.py -atheris_runs=50000

Built into a libFuzzer binary for ClusterFuzzLite by ``.clusterfuzzlite/build.sh``.
This file is named ``fuzz_*.py`` (not ``test_*.py``) and is excluded from pytest
collection via ``--ignore=tests/fuzz`` so the project test run never imports atheris.
"""

from __future__ import annotations

import sys

import atheris
import numpy as np

from cvx.core import Variable
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.sample import SampleCovariance

# Exceptions that represent a legitimate rejection of malformed or
# ill-conditioned input rather than a defect in cvxrisk.
_EXPECTED = (
    ValueError,
    IndexError,
    ZeroDivisionError,
    FloatingPointError,
    OverflowError,
    np.linalg.LinAlgError,
)


def _floats(fdp: atheris.FuzzedDataProvider, count: int) -> np.ndarray:
    """Return an array of ``count`` finite floats drawn from the fuzzer."""
    return np.array([fdp.ConsumeFloatInRange(-1.0e3, 1.0e3) for _ in range(count)], dtype=float)


def test_one_input(data: bytes) -> None:
    """Drive one of the three risk models end-to-end on fuzzer-derived numeric input."""
    fdp = atheris.FuzzedDataProvider(data)
    n = fdp.ConsumeIntInRange(1, 6)
    choice = fdp.ConsumeIntInRange(0, 2)
    try:
        if choice == 0:
            model = SampleCovariance(num=n)
            model.update(
                cov=_floats(fdp, n * n).reshape(n, n),
                lower_assets=_floats(fdp, n),
                upper_assets=_floats(fdp, n),
            )
            model.estimate(_floats(fdp, n))
            minrisk_problem(model, Variable(n)).solve()
        elif choice == 1:
            scenarios = fdp.ConsumeIntInRange(1, 25)
            model = CVar(alpha=fdp.ConsumeProbability(), n=scenarios, m=n)
            model.update(
                returns=_floats(fdp, scenarios * n).reshape(scenarios, n),
                lower_assets=_floats(fdp, n),
                upper_assets=_floats(fdp, n),
            )
            model.estimate(_floats(fdp, n))
        else:
            k = fdp.ConsumeIntInRange(1, n)
            model = FactorModel(assets=n, k=k)
            model.update(
                exposure=_floats(fdp, k * n).reshape(k, n),
                cov=_floats(fdp, k * k).reshape(k, k),
                idiosyncratic_risk=_floats(fdp, n),
                lower_assets=_floats(fdp, n),
                upper_assets=_floats(fdp, n),
                lower_factors=_floats(fdp, k),
                upper_factors=_floats(fdp, k),
            )
            model.estimate(_floats(fdp, n))
    except _EXPECTED:
        return


def main() -> None:
    """Register the harness with Atheris and start fuzzing."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
