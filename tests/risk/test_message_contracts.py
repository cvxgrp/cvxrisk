"""Error-message and parameter-name contracts, pinned by mutation testing.

Each test kills a mutant that survived the initial mutation-testing run by
asserting the *exact* validation messages (and parameter names) of the three
risk models. Exact equality (rather than a substring/regex search) is essential:
it also kills mutants that merely wrap the message in sentinel text, which a
``pytest.raises(match=...)`` search would still accept.

The messages are part of the user-facing contract: they tell a caller precisely
which input was wrong and why.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core import Variable
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.sample import SampleCovariance


# --------------------------------------------------------------------------- #
# SampleCovariance
# --------------------------------------------------------------------------- #
def test_sample_chol_parameter_name():
    """The Cholesky parameter is labelled "cholesky of covariance"."""
    assert SampleCovariance(num=2).parameter["chol"].name == "cholesky of covariance"


def test_sample_update_requires_cov():
    """Omitting ``cov`` names the missing argument exactly."""
    model = SampleCovariance(num=2)
    with pytest.raises(ValueError) as exc:
        model.update(lower_assets=np.zeros(2), upper_assets=np.ones(2))
    assert str(exc.value) == "update() requires a 'cov' argument"


def test_sample_update_rejects_non_square_cov():
    """A non-square covariance is rejected with its shape, exercising the squareness guard."""
    model = SampleCovariance(num=3)
    with pytest.raises(ValueError) as exc:
        model.update(cov=np.zeros((2, 3)), lower_assets=np.zeros(2), upper_assets=np.ones(2))
    assert str(exc.value) == "cov must be a square matrix, got shape (2, 3)"


def test_sample_update_rejects_oversized_cov():
    """A covariance larger than capacity reports its size and the model capacity, verbatim."""
    model = SampleCovariance(num=2)
    with pytest.raises(ValueError) as exc:
        model.update(cov=np.eye(3), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    assert str(exc.value) == "Too many assets: cov is 3x3 but the model capacity is num=2"


def test_sample_solve_minrisk_rejects_wrong_weight_dimension():
    """``solve_minrisk`` reports a weight/capacity mismatch with both dimensions, verbatim."""
    model = SampleCovariance(num=2)
    with pytest.raises(ValueError) as exc:
        model.solve_minrisk(Variable(3), np.zeros(3), [])
    assert str(exc.value) == "weights has dimension 3 but the model capacity is num=2"


# --------------------------------------------------------------------------- #
# CVar
# --------------------------------------------------------------------------- #
def test_cvar_returns_parameter_name():
    """The returns matrix parameter is labelled "returns"."""
    assert CVar(alpha=0.95, n=10, m=5).parameter["R"].name == "returns"


def test_cvar_update_requires_returns():
    """Omitting ``returns`` names the missing argument exactly."""
    model = CVar(alpha=0.95, n=10, m=5)
    with pytest.raises(ValueError) as exc:
        model.update(lower_assets=np.zeros(3), upper_assets=np.ones(3))
    assert str(exc.value) == "update() requires a 'returns' argument"


def test_cvar_update_rejects_non_2d_returns():
    """A non-2D returns array is rejected with its shape, verbatim."""
    model = CVar(alpha=0.95, n=10, m=5)
    with pytest.raises(ValueError) as exc:
        model.update(returns=np.zeros(10), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    assert str(exc.value) == "returns must be a 2d matrix of shape (n, num_assets), got shape (10,)"


def test_cvar_update_rejects_wrong_scenario_count():
    """A wrong scenario count reports the actual row count (shape[0]), not the column count."""
    model = CVar(alpha=0.95, n=10, m=5)
    with pytest.raises(ValueError) as exc:
        model.update(returns=np.zeros((7, 3)), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    assert str(exc.value) == "returns has 7 scenarios but the model expects n=10"


def test_cvar_update_rejects_too_many_assets():
    """Too many return columns reports the column count and the model capacity, verbatim."""
    model = CVar(alpha=0.95, n=10, m=5)
    with pytest.raises(ValueError) as exc:
        model.update(returns=np.zeros((10, 6)), lower_assets=np.zeros(6), upper_assets=np.ones(6))
    assert str(exc.value) == "Too many assets: returns has 6 columns but the model capacity is m=5"


def test_cvar_solve_minrisk_rejects_wrong_weight_dimension():
    """``solve_minrisk`` reports a weight/capacity mismatch with both dimensions, verbatim."""
    model = CVar(alpha=0.95, n=10, m=5)
    with pytest.raises(ValueError) as exc:
        model.solve_minrisk(Variable(6), np.zeros(6), [])
    assert str(exc.value) == "weights has dimension 6 but the model capacity is m=5"


# --------------------------------------------------------------------------- #
# FactorModel
# --------------------------------------------------------------------------- #
def test_factor_parameter_names():
    """The exposure, idiosyncratic-risk, and Cholesky parameters carry their names."""
    model = FactorModel(assets=3, k=2)
    assert model.parameter["exposure"].name == "exposure"
    assert model.parameter["idiosyncratic_risk"].name == "idiosyncratic risk"
    assert model.parameter["chol"].name == "cholesky of covariance"


def test_factor_update_lists_all_missing_arguments():
    """Missing arguments are reported as an exact comma-separated list (separator pinned)."""
    model = FactorModel(assets=3, k=2)
    with pytest.raises(ValueError) as exc:
        model.update()
    assert str(exc.value) == "update() missing required arguments: exposure, cov, idiosyncratic_risk"


def test_factor_update_rejects_cov_shape_mismatch():
    """A covariance whose shape does not match the exposure factor count is rejected, verbatim."""
    model = FactorModel(assets=3, k=2)
    with pytest.raises(ValueError) as exc:
        model.update(
            exposure=np.zeros((2, 3)),
            cov=np.eye(3),
            idiosyncratic_risk=np.ones(3),
            lower_assets=np.zeros(3),
            upper_assets=np.ones(3),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )
    assert str(exc.value) == "cov must have shape (2, 2) to match exposure, got (3, 3)"


def test_factor_update_rejects_idiosyncratic_risk_shape_mismatch():
    """An idiosyncratic-risk vector whose shape does not match the asset count is rejected, verbatim."""
    model = FactorModel(assets=3, k=2)
    with pytest.raises(ValueError) as exc:
        model.update(
            exposure=np.zeros((2, 3)),
            cov=np.eye(2),
            idiosyncratic_risk=np.ones(2),
            lower_assets=np.zeros(3),
            upper_assets=np.ones(3),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )
    assert str(exc.value) == "idiosyncratic_risk must have shape (3,) to match exposure, got (2,)"


def test_factor_solve_minrisk_rejects_wrong_weight_dimension():
    """``solve_minrisk`` reports a weight/capacity mismatch with both dimensions, verbatim."""
    model = FactorModel(assets=3, k=2)
    with pytest.raises(ValueError) as exc:
        model.solve_minrisk(Variable(5), np.zeros(5), [])
    assert str(exc.value) == "weights has dimension 5 but the model capacity is assets=3"
