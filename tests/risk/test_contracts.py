"""Tests pinning API contracts surfaced by mutation testing.

Each test here kills a mutant that survived the initial mutation-testing run:
constructor defaults, the documented ``y`` keyword of ``FactorModel.estimate``,
dimension-violation error messages, and the (objective, risk, status) return
contract of ``solve_minrisk``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core import Bounds, Model, Variable
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio.min_risk import MinRiskProblem
from cvx.risk.sample import SampleCovariance


def test_constructor_defaults():
    """The documented dataclass defaults are part of the public API."""
    assert SampleCovariance().num == 0
    assert FactorModel().assets == 0
    assert FactorModel().k == 0
    cvar = CVar()
    assert cvar.alpha == 0.95
    assert cvar.n == 0
    assert cvar.m == 0
    bounds = Bounds()
    assert bounds.m == 0
    assert bounds.name == ""


def test_model_is_abstract():
    """Both estimate and update must be abstract: implementing only one is not enough."""
    with pytest.raises(TypeError):
        Model()

    class OnlyEstimate(Model):
        """Subclass that implements only estimate, leaving update abstract."""

        def estimate(self, weights, **kwargs):
            """Return a constant risk estimate, ignoring the inputs."""
            return 0.0

    class OnlyUpdate(Model):
        """Subclass that implements only update, leaving estimate abstract."""

        def update(self, **kwargs):
            """Do nothing; provided solely to satisfy the abstract update method."""

    with pytest.raises(TypeError):
        OnlyEstimate()
    with pytest.raises(TypeError):
        OnlyUpdate()


def test_minrisk_problem_pads_short_base():
    """A base portfolio shorter than the weights vector must be zero-padded."""
    n = 3
    model = SampleCovariance(num=n)
    cov = np.array([[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 2.0]])
    model.update(cov=cov, lower_assets=np.zeros(n), upper_assets=np.ones(n))

    weights_short = Variable(n)
    problem_short = MinRiskProblem(riskmodel=model, weights=weights_short, base=np.array([0.5, 0.5]))
    problem_short.solve()
    assert "Solved" in problem_short.status

    weights_padded = Variable(n)
    problem_padded = MinRiskProblem(riskmodel=model, weights=weights_padded, base=np.array([0.5, 0.5, 0.0]))
    problem_padded.solve()
    assert "Solved" in problem_padded.status

    assert np.isclose(problem_short.value, problem_padded.value, rtol=1e-9)
    assert np.allclose(np.array(weights_short.value), np.array(weights_padded.value), atol=1e-7)


def test_minrisk_problem_direct_construction():
    """MinRiskProblem must be usable directly with its field defaults (no base, no constraints)."""
    n = 2
    model = SampleCovariance(num=n)
    model.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
    )
    weights = Variable(n)
    problem = MinRiskProblem(riskmodel=model, weights=weights)
    problem.solve()

    assert "Solved" in problem.status
    assert np.isclose(np.sum(np.array(weights.value)), 1.0, atol=1e-6)


def test_factor_estimate_uses_explicit_y():
    """An explicitly passed ``y`` must be used instead of exposure @ weights."""
    n, k = 3, 2
    model = FactorModel(assets=n, k=k)
    exposure = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    model.update(
        exposure=exposure,
        cov=np.eye(k),
        idiosyncratic_risk=0.1 * np.ones(n),
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
        lower_factors=-np.ones(k),
        upper_factors=np.ones(k),
    )
    w = np.array([0.4, 0.3, 0.3])
    custom_y = np.array([10.0, -10.0])

    default_risk = model.estimate(w)
    custom_risk = model.estimate(w, y=custom_y)
    expected_custom = np.sqrt(custom_y @ custom_y + np.sum((0.1 * w) ** 2))

    assert not np.isclose(default_risk, custom_risk)
    assert np.isclose(custom_risk, expected_custom, rtol=1e-8)


def test_factor_update_rejects_too_many_assets():
    """Exceeding the asset capacity must raise with the documented message."""
    model = FactorModel(assets=2, k=2)
    with pytest.raises(ValueError, match=r"^Too many assets$"):
        model.update(
            exposure=np.ones((2, 5)),  # 5 assets > capacity of 2
            cov=np.eye(2),
            idiosyncratic_risk=np.ones(5),
            lower_assets=np.zeros(5),
            upper_assets=np.ones(5),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )


@pytest.mark.parametrize("model_name", ["sample", "factor"])
def test_solve_minrisk_returns_achieved_risk(model_name):
    """The risk element of (objective, risk, status) must equal the optimal objective.

    For the minimum-risk problems the objective *is* the portfolio risk, so the
    second element must match the first (and not, say, a stray solution entry).
    """
    n, k = 4, 2
    rng = np.random.default_rng(42)
    if model_name == "sample":
        g = rng.standard_normal((n, n))
        model = SampleCovariance(num=n)
        model.update(
            cov=g.T @ g / n + 0.1 * np.eye(n),
            lower_assets=np.zeros(n),
            upper_assets=np.ones(n),
        )
    else:
        model = FactorModel(assets=n, k=k)
        model.update(
            exposure=rng.standard_normal((k, n)),
            cov=np.eye(k),
            idiosyncratic_risk=rng.uniform(0.05, 0.2, n),
            lower_assets=np.zeros(n),
            upper_assets=np.ones(n),
            lower_factors=-100 * np.ones(k),
            upper_factors=100 * np.ones(k),
        )

    weights = Variable(n)
    objective, risk, status = model.solve_minrisk(weights, np.zeros(n), [])

    assert "Solved" in status
    assert np.isclose(risk, objective, rtol=1e-9)
    assert np.isclose(risk, model.estimate(np.array(weights.value)), rtol=1e-6)
