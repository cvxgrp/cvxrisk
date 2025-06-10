# Introduction to cvxrisk

cvxrisk is a Python library for portfolio risk management using convex optimization. It provides a flexible framework for implementing various risk models that can be used with [CVXPY](https://github.com/cvxpy/cvxpy) to solve portfolio optimization problems.

## What is cvxrisk?

cvxrisk is designed to make it easy to formulate and solve portfolio optimization problems with different risk models. The library is built around an abstract `Model` class that standardizes the interface for different risk models, making it easy to swap between them in your optimization problems.

Key features of cvxrisk include:

- A unified interface for different risk models
- Support for sample covariance, factor models, and Conditional Value at Risk (CVaR)
- Integration with CVXPY for convex optimization
- Utilities for portfolio optimization and risk analysis
- Tools for generating random data for testing and simulation

## Installation

You can install cvxrisk using pip:

```bash
pip install cvxrisk
```

For development installation:

```bash
git clone https://github.com/cvxgrp/cvxrisk.git
cd cvxrisk
make install
```

## Key Concepts

### Risk Models

cvxrisk provides several risk models that all implement the abstract `Model` interface:

1. **Sample Covariance**: The simplest risk model based on the sample covariance matrix
2. **Factor Models**: Reduce dimensionality by projecting asset returns onto a smaller set of factors
3. **Conditional Value at Risk (CVaR)**: Measures the expected loss in the worst-case scenarios

All risk models implement three key methods:

- `estimate(weights, **kwargs)`: Computes the risk estimate for a given portfolio
- `update(**kwargs)`: Updates the model with new data
- `constraints(weights, **kwargs)`: Returns constraints for the optimization problem

### Portfolio Optimization

cvxrisk makes it easy to formulate and solve portfolio optimization problems using the risk models. The most common problem is the minimum risk problem, which minimizes the portfolio risk subject to constraints.

## Getting Started

### Sample Covariance Model

The simplest risk model in cvxrisk is the sample covariance model. Here's how to use it:

```python
import cvxpy as cp
import numpy as np
from cvxrisk.sample import SampleCovariance
from cvxrisk.portfolio.min_risk import minrisk_problem

# Create a risk model
riskmodel = SampleCovariance(num=2)

# Update the model with data
riskmodel.update(
    cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
    lower_assets=np.zeros(2),
    upper_assets=np.ones(2)
)

# Define portfolio weights variable
weights = cp.Variable(2)

# Create and solve the optimization problem
problem = minrisk_problem(riskmodel, weights)
problem.solve()

# Print the optimal weights
print(weights.value)  # [0.75, 0.25]
```

### Factor Risk Model

Factor models reduce dimensionality by projecting asset returns onto a smaller set of factors:

```python
import numpy as np
import pandas as pd
from cvxrisk.factor import FactorModel
from cvxrisk.linalg import pca
from cvxrisk.portfolio.min_risk import minrisk_problem
import cvxpy as cp

# Load returns data
returns = pd.DataFrame(np.random.randn(100, 20))

# Compute principal components
factors = pca(returns, n_components=10)

# Create and update the factor model
model = FactorModel(assets=20, k=10)
model.update(
    cov=factors.cov.values,
    exposure=factors.exposure.values,
    idiosyncratic_risk=factors.idiosyncratic.std().values,
    lower_assets=np.zeros(20),
    upper_assets=np.ones(20),
    lower_factors=np.zeros(10),
    upper_factors=np.ones(10)
)

# Define portfolio weights variable
weights = cp.Variable(20)
y = cp.Variable(10)  # Factor exposures

# Create and solve the optimization problem
problem = minrisk_problem(model, weights, y=y)
problem.solve()

# Print the optimal weights
print(weights.value)
```

### Conditional Value at Risk (CVaR)

CVaR measures the expected loss in the worst-case scenarios:

```python
import numpy as np
import cvxpy as cp
from cvxrisk.cvar import CVar
from cvxrisk.portfolio.min_risk import minrisk_problem

# Create a CVaR model
model = CVar(alpha=0.95, n=50, m=10)

# Generate random returns data
returns = np.random.randn(50, 10)

# Update the model with data
model.update(
    returns=returns,
    lower_assets=np.zeros(10),
    upper_assets=np.ones(10)
)

# Define portfolio weights variable
weights = cp.Variable(10)

# Create and solve the optimization problem
problem = minrisk_problem(model, weights)
problem.solve()

# Print the optimal weights
print(weights.value)
```

## Advanced Usage

### Custom Constraints

You can add custom constraints to the optimization problem:

```python
import cvxpy as cp
import numpy as np
from cvxrisk.sample import SampleCovariance
from cvxrisk.portfolio.min_risk import minrisk_problem

# Create a risk model
riskmodel = SampleCovariance(num=4)
riskmodel.update(
    cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
    lower_assets=np.zeros(2),
    upper_assets=np.ones(2)
)

# Define portfolio weights variable
weights = cp.Variable(4)

# Add custom constraints
custom_constraints = [weights[0] == 0.0]  # Force the first asset weight to be zero

# Create and solve the optimization problem with custom constraints
problem = minrisk_problem(riskmodel, weights, constraints=custom_constraints)
problem.solve()

# Print the optimal weights
print(weights.value)  # [0.0, 1.0, 0.0, 0.0]
```

### Tracking Error Minimization

You can minimize the tracking error relative to a benchmark portfolio:

```python
import cvxpy as cp
import numpy as np
from cvxrisk.sample import SampleCovariance
from cvxrisk.portfolio.min_risk import minrisk_problem

# Create a risk model
riskmodel = SampleCovariance(num=3)
riskmodel.update(
    cov=np.array([[1.0, 0.5, 0.3], [0.5, 2.0, 0.4], [0.3, 0.4, 1.5]]),
    lower_assets=np.zeros(3),
    upper_assets=np.ones(3)
)

# Define portfolio weights variable
weights = cp.Variable(3)

# Define benchmark portfolio
benchmark = np.array([0.3, 0.4, 0.3])

# Create and solve the tracking error minimization problem
problem = minrisk_problem(riskmodel, weights, base=benchmark)
problem.solve()

# Print the optimal weights
print(weights.value)
```

### Random Covariance Matrix Generation

cvxrisk provides utilities for generating random covariance matrices for testing:

```python
import numpy as np
from cvxrisk.random import rand_cov

# Generate a random 5x5 covariance matrix
cov = rand_cov(5)
print(cov)

# Verify it's positive definite
eigenvalues = np.linalg.eigvals(cov)
print("All eigenvalues positive:", np.all(eigenvalues > 0))
```

## Mathematical Background

### Factor Risk Models

Factor risk models use the projection of the weight vector into a lower dimensional subspace, where each asset is the linear combination of k factors:

$$r_i = \sum_{j=1}^k f_j \beta_{ji} + \epsilon_i$$

The factor time series are $f_1, \ldots, f_k$. The loadings are the coefficients $\beta_{ji}$. The residual returns $\epsilon_i$ are assumed to be uncorrelated with the factors.

Any position $w$ in weight space projects to a position $y = \beta^T w$ in factor space. The variance for a position $w$ is the sum of the variance of the systematic returns explained by the factors and the variance of the idiosyncratic returns:

$$Var(r) = Var(\beta^T w) + Var(\epsilon w)$$

We assume the residual returns are uncorrelated and hence:

$$Var(r) = y^T \Sigma_f y + \sum_i w_i^2 Var(\epsilon_i)$$

where $\Sigma_f$ is the covariance matrix of the factors and $Var(\epsilon_i)$ is the variance of the idiosyncratic returns.

## Further Reading

For more detailed documentation and examples, visit the [cvxrisk documentation site](http://www.cvxgrp.org/cvxrisk/book).

To learn more about convex optimization and its applications in finance, check out:

- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) by Stephen Boyd and Lieven Vandenberghe
- [CVXPY](https://www.cvxpy.org/) documentation
