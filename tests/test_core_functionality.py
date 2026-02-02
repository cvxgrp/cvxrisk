"""Tests demonstrating the core functionality of the cvxrisk package.

This test file provides comprehensive examples of using the main components
of the cvxrisk package, including:

1. Creating and solving minimum risk portfolio optimization problems
2. Using different risk models:
   - Sample Covariance
   - Factor Model
   - Conditional Value at Risk (CVaR)
3. Generating random data for testing and simulation

These tests serve both as verification of the package's functionality
and as examples of how to use the package in practice.
"""

import cvxpy as cp
import numpy as np
import pandas as pd

from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.linalg import pca
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.rand import rand_cov
from cvx.risk.sample import SampleCovariance


def test_sample_covariance_optimization():
    """Test portfolio optimization using the Sample Covariance risk model.

    This test demonstrates:
    1. Creating a Sample Covariance risk model
    2. Updating the model with a covariance matrix
    3. Creating and solving a minimum risk portfolio optimization problem
    4. Verifying the solution satisfies basic constraints
    """
    # Create a risk model
    n_assets = 10
    riskmodel = SampleCovariance(num=n_assets)

    # Generate a random covariance matrix
    cov_matrix = rand_cov(n_assets)

    # Update the model with data
    riskmodel.update(cov=cov_matrix, lower_assets=np.zeros(n_assets), upper_assets=np.ones(n_assets))

    # Define portfolio weights variable
    weights = cp.Variable(n_assets)

    # Create and solve the optimization problem
    problem = minrisk_problem(riskmodel, weights)
    problem.solve(solver="CLARABEL")

    # Verify the solution
    assert problem.status == "optimal"
    assert np.isclose(np.sum(weights.value), 1.0)
    assert np.all(weights.value >= -1e-6)  # Allow for small numerical errors

    # Print the optimal weights and risk
    print(f"Optimal weights: {weights.value}")
    print(f"Portfolio risk: {riskmodel.estimate(weights).value}")


def test_factor_model_optimization():
    """Test portfolio optimization using the Factor Model risk model.

    This test demonstrates:
    1. Creating a Factor Model risk model
    2. Generating random returns data and computing principal components
    3. Updating the model with factor data
    4. Creating and solving a minimum risk portfolio optimization problem
    5. Verifying the solution satisfies basic constraints
    """
    # Create a random number generator
    rng = np.random.default_rng(42)

    # Create a factor model
    n_assets = 20
    n_factors = 5
    model = FactorModel(assets=n_assets, k=n_factors)

    # Generate random returns data
    n_periods = 100
    returns = pd.DataFrame(rng.standard_normal((n_periods, n_assets)))

    # Compute principal components
    factors = pca(returns, n_components=n_factors)

    # Update the factor model
    model.update(
        cov=factors.cov.values,
        exposure=factors.exposure.values,
        idiosyncratic_risk=factors.idiosyncratic.std().values,
        lower_assets=np.zeros(n_assets),
        upper_assets=np.ones(n_assets),
        lower_factors=-0.1 * np.ones(n_factors),
        upper_factors=0.1 * np.ones(n_factors),
    )

    # Define portfolio weights and factor exposures variables
    w = cp.Variable(n_assets)
    y = cp.Variable(n_factors)

    # Create and solve the optimization problem
    problem = minrisk_problem(model, w, y=y)
    problem.solve(solver="CLARABEL")

    # Verify the solution
    assert problem.status == "optimal"
    assert np.isclose(np.sum(w.value), 1.0)
    assert np.all(w.value >= -1e-6)  # Allow for small numerical errors

    # Print the optimal weights, factor exposures, and risk
    print(f"Optimal weights: {w.value}")
    print(f"Factor exposures: {y.value}")
    print(f"Portfolio risk: {model.estimate(w, y=y).value}")


def test_cvar_optimization():
    """Test portfolio optimization using the Conditional Value at Risk (CVaR) model.

    This test demonstrates:
    1. Creating a CVaR risk model
    2. Generating random historical returns
    3. Updating the model with returns data
    4. Creating and solving a minimum risk portfolio optimization problem
    5. Verifying the solution satisfies basic constraints
    """
    # Create a random number generator
    rng = np.random.default_rng(42)

    # Create a CVaR model
    n_assets = 15
    n_periods = 200
    alpha = 0.95
    model = CVar(alpha=alpha, n=n_periods, m=n_assets)

    # Generate random historical returns
    historical_returns = rng.standard_normal((n_periods, n_assets))

    # Update the CVaR model
    model.update(returns=historical_returns, lower_assets=np.zeros(n_assets), upper_assets=np.ones(n_assets))

    # Define portfolio weights variable
    weights = cp.Variable(n_assets)

    # Create and solve the optimization problem
    problem = minrisk_problem(model, weights)
    problem.solve(solver="CLARABEL")

    # Verify the solution
    assert problem.status == "optimal"
    assert np.isclose(np.sum(weights.value), 1.0)
    assert np.all(weights.value >= -1e-6)  # Allow for small numerical errors

    # Print the optimal weights and CVaR
    print(f"Optimal weights: {weights.value}")
    print(f"Portfolio CVaR: {model.estimate(weights).value}")


def test_combined_optimization():
    """Test portfolio optimization comparing different risk models on the same data.

    This test demonstrates:
    1. Creating multiple risk models
    2. Generating common data for all models
    3. Solving optimization problems with each model
    4. Comparing the results
    """
    # Create a random number generator
    rng = np.random.default_rng(42)

    # Set up common parameters
    n_assets = 8
    n_periods = 150

    # Generate random returns data
    returns = rng.standard_normal((n_periods, n_assets))
    cov_matrix = np.cov(returns.T)

    # Create risk models
    sample_model = SampleCovariance(num=n_assets)
    cvar_model = CVar(alpha=0.9, n=n_periods, m=n_assets)

    # Update the models with data
    sample_model.update(cov=cov_matrix, lower_assets=np.zeros(n_assets), upper_assets=np.ones(n_assets))

    cvar_model.update(returns=returns, lower_assets=np.zeros(n_assets), upper_assets=np.ones(n_assets))

    # Define portfolio weights variable
    weights_sample = cp.Variable(n_assets)
    weights_cvar = cp.Variable(n_assets)

    # Create and solve the optimization problems
    problem_sample = minrisk_problem(sample_model, weights_sample)
    problem_sample.solve(solver="CLARABEL")

    problem_cvar = minrisk_problem(cvar_model, weights_cvar)
    problem_cvar.solve(solver="CLARABEL")

    # Verify the solutions
    assert problem_sample.status == "optimal"
    assert problem_cvar.status == "optimal"

    # Compare the results
    print("Sample Covariance optimal weights:")
    print(weights_sample.value)
    print(f"Sample Covariance risk: {sample_model.estimate(weights_sample).value}")

    print("\nCVaR optimal weights:")
    print(weights_cvar.value)
    print(f"CVaR risk: {cvar_model.estimate(weights_cvar).value}")

    # Calculate the risk of each portfolio using the other risk model
    print(f"\nSample portfolio evaluated with CVaR: {cvar_model.estimate(weights_sample).value}")
    print(f"CVaR portfolio evaluated with Sample Covariance: {sample_model.estimate(weights_cvar).value}")


def test_custom_constraints():
    """Test portfolio optimization with custom constraints.

    This test demonstrates:
    1. Creating a risk model
    2. Adding custom constraints to the optimization problem
    3. Solving the constrained optimization problem
    4. Verifying the solution satisfies all constraints
    """
    # Create a risk model
    n_assets = 10
    riskmodel = SampleCovariance(num=n_assets)

    # Generate a random covariance matrix
    cov_matrix = rand_cov(n_assets)

    # Update the model with data
    riskmodel.update(cov=cov_matrix, lower_assets=np.zeros(n_assets), upper_assets=np.ones(n_assets))

    # Define portfolio weights variable
    weights = cp.Variable(n_assets)

    # Create custom constraints

    # 1. Sector constraints: assume first 3 assets are in sector 1, next 3 in sector 2, rest in sector 3
    sector1_weights = weights[:3]
    sector2_weights = weights[3:6]
    sector3_weights = weights[6:]

    sector_constraints = [
        cp.sum(sector1_weights) <= 0.4,  # Sector 1 <= 40%
        cp.sum(sector2_weights) >= 0.2,  # Sector 2 >= 20%
        cp.sum(sector3_weights) <= 0.5,  # Sector 3 <= 50%
    ]

    # 2. Tracking error constraint: assume we have a benchmark
    benchmark = np.ones(n_assets) / n_assets  # Equal weight benchmark
    tracking_constraints = [
        cp.sum(cp.abs(weights - benchmark)) <= 0.5  # Limit total deviation from benchmark
    ]

    # Combine all custom constraints
    custom_constraints = sector_constraints + tracking_constraints

    # Create and solve the optimization problem with custom constraints
    problem = minrisk_problem(riskmodel, weights, constraints=custom_constraints)
    problem.solve(solver="CLARABEL")

    # Verify the solution
    assert problem.status == "optimal"
    assert np.isclose(np.sum(weights.value), 1.0)
    assert np.all(weights.value >= -1e-6)  # Allow for small numerical errors

    # Verify sector constraints
    assert np.sum(weights.value[:3]) <= 0.4 + 1e-6
    assert np.sum(weights.value[3:6]) >= 0.2 - 1e-6
    assert np.sum(weights.value[6:]) <= 0.5 + 1e-6

    # Verify tracking error constraint
    assert np.sum(np.abs(weights.value - benchmark)) <= 0.5 + 1e-6

    # Print the optimal weights and risk
    print(f"Optimal weights: {weights.value}")
    print(f"Portfolio risk: {riskmodel.estimate(weights).value}")
    print(f"Sector 1 weight: {np.sum(weights.value[:3])}")
    print(f"Sector 2 weight: {np.sum(weights.value[3:6])}")
    print(f"Sector 3 weight: {np.sum(weights.value[6:])}")
    print(f"Tracking error: {np.sum(np.abs(weights.value - benchmark))}")
