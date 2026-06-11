"""Property-based tests for the risk models using Hypothesis.

Each test states a mathematical invariant the models must satisfy for *any*
valid input, and Hypothesis searches for counterexamples:

- SampleCovariance.estimate equals the quadratic form sqrt(w' cov w).
- FactorModel.estimate equals the sample-covariance risk of the reconstructed
  covariance matrix beta' cov_f beta + diag(idio^2).
- CVar.estimate equals the negated mean of the k worst scenario returns.
- Minimum-risk solutions are feasible and no feasible portfolio beats them.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from cvx.core import Variable
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.sample import SampleCovariance

pytestmark = pytest.mark.property


@st.composite
def covariance_matrices(draw, n: int) -> np.ndarray:
    """Draw a strictly positive definite n x n covariance matrix."""
    g = draw(hnp.arrays(np.float64, (n, n), elements=st.floats(-2.0, 2.0)))
    return g.T @ g + 0.5 * np.eye(n)


@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_sample_estimate_is_quadratic_form(data):
    """SampleCovariance.estimate(w) must equal sqrt(w' cov w) for any w."""
    n = data.draw(st.integers(2, 8), label="n")
    cov = data.draw(covariance_matrices(n), label="cov")
    w = data.draw(hnp.arrays(np.float64, (n,), elements=st.floats(-1.0, 1.0)), label="w")

    model = SampleCovariance(num=n)
    model.update(cov=cov, lower_assets=np.zeros(n), upper_assets=np.ones(n))

    expected = np.sqrt(w @ cov @ w)
    assert np.isclose(model.estimate(w), expected, rtol=1e-8, atol=1e-10)


@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_factor_estimate_matches_reconstructed_covariance(data):
    """FactorModel risk must equal the full-covariance risk of beta' cov_f beta + diag(idio^2)."""
    n = data.draw(st.integers(2, 8), label="assets")
    k = data.draw(st.integers(1, 4), label="factors")
    cov_f = data.draw(covariance_matrices(k), label="factor cov")
    exposure = data.draw(hnp.arrays(np.float64, (k, n), elements=st.floats(-1.0, 1.0)), label="exposure")
    idio = data.draw(hnp.arrays(np.float64, (n,), elements=st.floats(0.01, 1.0)), label="idio")
    w = data.draw(hnp.arrays(np.float64, (n,), elements=st.floats(-1.0, 1.0)), label="w")

    model = FactorModel(assets=n, k=k)
    model.update(
        exposure=exposure,
        cov=cov_f,
        idiosyncratic_risk=idio,
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
        lower_factors=-np.ones(k),
        upper_factors=np.ones(k),
    )

    reconstructed = exposure.T @ cov_f @ exposure + np.diag(idio**2)
    expected = np.sqrt(w @ reconstructed @ w)
    assert np.isclose(model.estimate(w), expected, rtol=1e-8, atol=1e-10)


@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_cvar_estimate_is_tail_mean(data):
    """CVar.estimate(w) must equal the negated mean of the k smallest scenario returns."""
    n_scenarios = data.draw(st.integers(20, 60), label="scenarios")
    m_max = data.draw(st.integers(2, 6), label="max assets")
    # Use fewer assets than the model capacity to exercise the padding logic.
    m_used = data.draw(st.integers(2, m_max), label="used assets")
    alpha = data.draw(st.sampled_from([0.8, 0.9, 0.95]), label="alpha")
    returns = data.draw(
        hnp.arrays(np.float64, (n_scenarios, m_used), elements=st.floats(-1.0, 1.0)),
        label="returns",
    )
    w = data.draw(hnp.arrays(np.float64, (m_used,), elements=st.floats(0.0, 1.0)), label="w")

    model = CVar(alpha=alpha, n=n_scenarios, m=m_max)
    model.update(returns=returns, lower_assets=np.zeros(m_used), upper_assets=np.ones(m_used))

    k = int(n_scenarios * (1 - alpha))
    padded_w = np.zeros(m_max)
    padded_w[:m_used] = w
    expected = -np.mean(np.sort(returns @ w)[:k])
    assert np.isclose(model.estimate(padded_w), expected, rtol=1e-8, atol=1e-10)


@settings(max_examples=25, deadline=None)
@given(data=st.data())
def test_minrisk_solution_is_feasible_and_optimal(data):
    """The minimum-risk portfolio must be feasible, and no feasible portfolio may have lower risk."""
    n = data.draw(st.integers(2, 6), label="n")
    cov = data.draw(covariance_matrices(n), label="cov")
    seed = data.draw(st.integers(0, 2**32 - 1), label="seed")

    model = SampleCovariance(num=n)
    model.update(cov=cov, lower_assets=np.zeros(n), upper_assets=np.ones(n))

    weights = Variable(n)
    problem = minrisk_problem(model, weights)
    problem.solve()

    assert "Solved" in problem.status
    w_opt = np.array(weights.value)

    # Feasibility (allowing for solver tolerance).
    assert np.isclose(np.sum(w_opt), 1.0, atol=1e-6)
    assert np.all(w_opt >= -1e-6)
    assert np.all(w_opt <= 1.0 + 1e-6)

    # Optimality: random points on the simplex must not have lower risk.
    rng = np.random.default_rng(seed)
    candidates = rng.dirichlet(np.ones(n), size=50)
    optimal_risk = model.estimate(w_opt)
    for candidate in candidates:
        assert model.estimate(candidate) >= optimal_risk - 1e-6
