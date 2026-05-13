"""Benchmarks comparing direct Clarabel vs cvxpy approaches.

Measures performance of the new Clarabel-direct portfolio optimisation
(no cvxpy overhead) against the equivalent cvxpy formulation for both
SampleCovariance and FactorModel risk models.

Run with::

    pytest tests/benchmarks/ --benchmark-sort=name

The cvxpy benchmarks are skipped automatically when cvxpy-base is not
installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from cvx.linalg import cholesky

from cvx.core.variable import Variable
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.sample import SampleCovariance

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cov(n: int, seed: int = 42) -> np.ndarray:
    """Return a random positive-definite n×n covariance matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))  # noqa: N806
    return (A @ A.T) / n + np.eye(n) * 0.1


def _make_factor_data(assets: int, k: int, seed: int = 0):
    """Return (exposure, factor_cov, idio_risk) for a factor model."""
    rng = np.random.default_rng(seed)
    exposure = rng.standard_normal((k, assets))
    factor_cov = _make_cov(k, seed=seed + 1)
    idio_risk = np.abs(rng.standard_normal(assets)) * 0.1 + 0.05
    return exposure, factor_cov, idio_risk


# ---------------------------------------------------------------------------
# Direct Clarabel benchmarks (new approach)
# ---------------------------------------------------------------------------


class TestClarabelDirect:
    """Benchmark min-risk problems solved directly with Clarabel (no cvxpy)."""

    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    def test_sample_covariance(self, benchmark, n):
        """Time each full solve (model setup already done outside the loop)."""
        cov = _make_cov(n)
        model = SampleCovariance(num=n)
        model.update(
            cov=cov,
            lower_assets=np.zeros(n),
            upper_assets=np.ones(n),
        )
        weights = Variable(n)

        def solve():
            problem = minrisk_problem(model, weights)
            problem.solve()
            return weights.value

        result = benchmark(solve)
        assert result is not None

    @pytest.mark.parametrize(
        ("assets", "k"),
        [(20, 5), (100, 10), (200, 20)],
    )
    def test_factor_model(self, benchmark, assets, k):
        """Benchmark FactorModel optimisation via direct Clarabel."""
        exposure, factor_cov, idio_risk = _make_factor_data(assets, k)
        model = FactorModel(assets=assets, k=k)
        model.update(
            exposure=exposure,
            cov=factor_cov,
            idiosyncratic_risk=idio_risk,
            lower_assets=np.zeros(assets),
            upper_assets=np.ones(assets),
            lower_factors=-np.ones(k),
            upper_factors=np.ones(k),
        )
        weights = Variable(assets)

        def solve():
            problem = minrisk_problem(model, weights)
            problem.solve()
            return weights.value

        result = benchmark(solve)
        assert result is not None


# ---------------------------------------------------------------------------
# cvxpy benchmarks (old approach) - skipped when cvxpy-base is absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy-base not installed")
class TestCvxpy:
    """Equivalent benchmarks using cvxpy + Clarabel backend (old approach)."""

    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    def test_sample_covariance(self, benchmark, n):
        """Time each cvxpy problem construction + solve."""
        cov = _make_cov(n)
        L = cholesky(cov)  # upper-triangular Cholesky factor  # noqa: N806

        def solve():
            w = cp.Variable(n)
            risk = cp.norm(L @ w, 2)
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= 1,
            ]
            prob = cp.Problem(cp.Minimize(risk), constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            return w.value

        result = benchmark(solve)
        assert result is not None

    @pytest.mark.parametrize(
        ("assets", "k"),
        [(20, 5), (100, 10), (200, 20)],
    )
    def test_factor_model(self, benchmark, assets, k):
        """Benchmark FactorModel equivalent using cvxpy."""
        exposure, factor_cov, idio_risk = _make_factor_data(assets, k)
        L = cholesky(factor_cov)  # (k, k) upper-triangular Cholesky factor  # noqa: N806

        def solve():
            w = cp.Variable(assets)
            y = cp.Variable(k)
            systematic_risk = cp.norm(L @ y, 2)
            idio_term = cp.norm(cp.multiply(idio_risk, w), 2)
            total_risk = cp.norm(cp.vstack([systematic_risk, idio_term]), 2)
            constraints = [
                cp.sum(w) == 1,
                y == exposure @ w,
                w >= 0,
                w <= 1,
                y >= -1,
                y <= 1,
            ]
            prob = cp.Problem(cp.Minimize(total_risk), constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            return w.value

        result = benchmark(solve)
        assert result is not None
