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

import numpy as np
import pandas as pd

from cvx.core.variable import Variable
from cvx.linalg import pca, rand_cov
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem
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
    weights = Variable(n_assets)

    # Create and solve the optimization problem
    problem = minrisk_problem(riskmodel, weights)
    problem.solve()

    # Verify the solution
    assert "Solved" in problem.status
    assert np.isclose(np.sum(weights.value), 1.0)
    assert np.all(weights.value >= -1e-6)  # Allow for small numerical errors

    # Print the optimal weights and risk
    print(f"Optimal weights: {weights.value}")
    print(f"Portfolio risk: {riskmodel.estimate(weights.value)}")


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
    w = Variable(n_assets)
    y = Variable(n_factors)

    # Create and solve the optimization problem
    problem = minrisk_problem(model, w, y=y)
    problem.solve()

    # Verify the solution
    assert "Solved" in problem.status
    assert np.isclose(np.sum(w.value), 1.0)
    assert np.all(w.value >= -1e-6)  # Allow for small numerical errors

    # Print the optimal weights, factor exposures, and risk
    print(f"Optimal weights: {w.value}")
    print(f"Factor exposures: {y.value}")
    print(f"Portfolio risk: {model.estimate(w.value)}")


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
    weights = Variable(n_assets)

    # Create and solve the optimization problem
    problem = minrisk_problem(model, weights)
    problem.solve()

    # Verify the solution
    assert "Solved" in problem.status
    assert np.isclose(np.sum(weights.value), 1.0)
    assert np.all(weights.value >= -1e-6)  # Allow for small numerical errors

    # Print the optimal weights and CVaR
    print(f"Optimal weights: {weights.value}")
    print(f"Portfolio CVaR: {model.estimate(weights.value)}")


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
    weights_sample = Variable(n_assets)
    weights_cvar = Variable(n_assets)

    # Create and solve the optimization problems
    problem_sample = minrisk_problem(sample_model, weights_sample)
    problem_sample.solve()

    problem_cvar = minrisk_problem(cvar_model, weights_cvar)
    problem_cvar.solve()

    # Verify the solutions
    assert "Solved" in problem_sample.status
    assert "Solved" in problem_cvar.status

    # Compare the results
    print("Sample Covariance optimal weights:")
    print(weights_sample.value)
    print(f"Sample Covariance risk: {sample_model.estimate(weights_sample.value)}")

    print("\nCVaR optimal weights:")
    print(weights_cvar.value)
    print(f"CVaR risk: {cvar_model.estimate(weights_cvar.value)}")

    # Calculate the risk of each portfolio using the other risk model
    print(f"\nSample portfolio evaluated with CVaR: {cvar_model.estimate(weights_sample.value)}")
    print(f"CVaR portfolio evaluated with Sample Covariance: {sample_model.estimate(weights_cvar.value)}")


def test_custom_constraints():
    """Test portfolio optimization with custom linear constraints.

    This test demonstrates:
    1. Creating a risk model
    2. Adding linear constraints to the optimization problem
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
    weights = Variable(n_assets)

    # Create custom constraints as (a, lb, ub) tuples

    # 1. Sector constraints: assume first 3 assets are in sector 1, next 3 in sector 2, rest in sector 3
    sector1 = np.zeros(n_assets)
    sector1[:3] = 1.0
    sector2 = np.zeros(n_assets)
    sector2[3:6] = 1.0
    sector3 = np.zeros(n_assets)
    sector3[6:] = 1.0

    custom_constraints = [
        (sector1, None, 0.4),  # Sector 1 <= 40%
        (sector2, 0.2, None),  # Sector 2 >= 20%
        (sector3, None, 0.5),  # Sector 3 <= 50%
    ]

    # Create and solve the optimization problem with custom constraints
    problem = minrisk_problem(riskmodel, weights, constraints=custom_constraints)
    problem.solve()

    # Verify the solution
    assert "Solved" in problem.status
    assert np.isclose(np.sum(weights.value), 1.0)
    assert np.all(weights.value >= -1e-6)  # Allow for small numerical errors

    # Verify sector constraints
    assert np.sum(weights.value[:3]) <= 0.4 + 1e-6
    assert np.sum(weights.value[3:6]) >= 0.2 - 1e-6
    assert np.sum(weights.value[6:]) <= 0.5 + 1e-6

    # Print the optimal weights and risk
    print(f"Optimal weights: {weights.value}")
    print(f"Portfolio risk: {riskmodel.estimate(weights.value)}")
    print(f"Sector 1 weight: {np.sum(weights.value[:3])}")
    print(f"Sector 2 weight: {np.sum(weights.value[3:6])}")
    print(f"Sector 3 weight: {np.sum(weights.value[6:])}")
