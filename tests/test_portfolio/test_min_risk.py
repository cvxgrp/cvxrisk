"""Testing MinVar."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from cvx.portfolio.min_risk import minrisk_problem
from cvx.risk.sample import SampleCovariance


def test_minrisk_problem_basic():
    """Test the functionality and correctness of the minrisk_problem setup and solution.

    This function verifies that a simple risk optimization problem can be created, solved
    successfully, and yields expected solutions under given conditions. It checks for
    validity, optimality, and adherence to portfolio constraints using basic assertions.

    Raises:
        AssertionError: If any of the validation checks fail, such as problem validity,
        DCP compliance, solution optimality, portfolio constraints (e.g., weights summing
        to 1), or specific expected behavior derived from the covariance matrix.

    """
    # Create a simple risk model
    riskmodel = SampleCovariance(num=2)
    riskmodel.update(cov=np.array([[1.0, 0.5], [0.5, 2.0]]), lower_assets=np.zeros(2), upper_assets=np.ones(2))

    # Define portfolio weights variable
    weights = cp.Variable(2)

    # Create the optimization problem
    problem = minrisk_problem(riskmodel, weights)

    # Check that the problem is valid
    assert isinstance(problem, cp.Problem)
    assert problem.is_dcp()

    # Solve the problem
    problem.solve(solver="CLARABEL")

    w = np.array(weights.value)

    # Check that the problem was solved successfully
    assert problem.status == cp.OPTIMAL

    # Check that the weights sum to 1
    assert np.isclose(np.sum(w), 1.0)

    # Check that the weights are non-negative
    assert np.all(w >= 0)

    # For this specific covariance matrix, we expect more weight on the first asset
    # since it has lower variance
    assert float(w[0]) > float(w[1])


def test_minrisk_problem_with_base():
    """Test that minrisk_problem works with a base portfolio."""
    # Create a simple risk model
    riskmodel = SampleCovariance(num=2)
    riskmodel.update(cov=np.array([[1.0, 0.5], [0.5, 2.0]]), lower_assets=np.zeros(2), upper_assets=np.ones(2))

    # Define portfolio weights variable
    weights = cp.Variable(2)

    # Define a base portfolio (e.g., for tracking error minimization)
    base = np.array([0.5, 0.5])

    # Create the optimization problem
    problem = minrisk_problem(riskmodel, weights, base=base)

    # Check that the problem is valid
    assert isinstance(problem, cp.Problem)
    assert problem.is_dcp()

    # Solve the problem
    problem.solve(solver="CLARABEL")

    # Check that the problem was solved successfully
    assert problem.status == cp.OPTIMAL

    # Check that the weights sum to 1
    assert np.isclose(np.sum(weights.value), 1.0)

    # Check that the weights are non-negative
    assert np.all(np.array(weights.value) >= 0)


def test_minrisk_problem_with_additional_constraints():
    """Test that minrisk_problem works with additional constraints."""
    # Create a simple risk model
    riskmodel = SampleCovariance(num=2)
    riskmodel.update(cov=np.array([[1.0, 0.5], [0.5, 2.0]]), lower_assets=np.zeros(2), upper_assets=np.ones(2))

    # Define portfolio weights variable
    weights = cp.Variable(2)

    # Define additional constraints
    additional_constraints = [weights[0] >= 0.3]  # At least 30% in the first asset

    # Create the optimization problem
    problem = minrisk_problem(riskmodel, weights, constraints=additional_constraints)

    # Check that the problem is valid
    assert isinstance(problem, cp.Problem)
    assert problem.is_dcp()

    # Solve the problem
    problem.solve(solver="CLARABEL")

    w = np.array(weights.value)

    # Check that the problem was solved successfully
    assert problem.status == cp.OPTIMAL

    # Check that the weights sum to 1
    assert np.isclose(np.sum(w), 1.0)

    # Check that the weights are non-negative
    assert np.all(w >= 0)

    # Check that the additional constraint is satisfied
    assert w[0] >= 0.3
