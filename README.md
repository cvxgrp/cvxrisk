# [cvxrisk](https://www.cvxgrp.org/cvxrisk/book): Convex Optimization for Portfolio Risk Management

[![PyPI version](https://badge.fury.io/py/cvxrisk.svg)](https://badge.fury.io/py/cvxrisk)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxrisk/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxrisk?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxrisk)
[![Renovate enabled](https://img.shields.io/badge/renovate-enabled-brightgreen.svg)](https://github.com/renovatebot/renovate)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxrisk)

## 📋 Overview

cvxrisk is a Python library for portfolio risk management using convex optimization.
It provides a flexible framework for implementing various risk models that
can be used with [CVXPY](https://github.com/cvxpy/cvxpy) to solve portfolio
optimization problems.

The library is built around an abstract `Model` class that standardizes
the interface for different risk models, making it easy to swap between
them in your optimization problems.

## 🚀 Installation

```bash
# Install from PyPI (without any convex solver)
pip install cvxrisk

# For development installation
git clone https://github.com/cvxgrp/cvxrisk.git
cd cvxrisk
make install

# For experimenting with the notebooks (after cloning)
make marimo
```

⚠️ **Warning!** The package does not install a convex solver if not explicitly desired.
It relies on cvxpy-base. If you use cvxrisk as a dependency
in your projects you may want to install [clarabel](https://github.com/oxfordcontrol/Clarabel.rs)
using `pip install cvxrisk[clarabel]` or [mosek](https://www.mosek.com)
using `pip install cvxrisk[mosek]`.

## 🔧 Quick Start

cvxrisk makes it easy to formulate and solve portfolio optimization problems:

```python
import cvxpy as cp
import numpy as np
from cvxrisk.sample import SampleCovariance
from cvxrisk.portfolio import minrisk_problem

# Create a risk model
riskmodel = SampleCovariance(num=2)

# Update the model with data
riskmodel.update(
    cov = np.array([[1.0, 0.5], [0.5, 2.0]]),
    lower_assets = np.zeros(2),
    upper_assets = np.ones(2)
)

# Define portfolio weights variable
weights = cp.Variable(2)

# Create and solve the optimization problem
problem = minrisk_problem(riskmodel, weights)
problem.solve(solver=cp.CLARABEL)

# Print the optimal weights with deterministic spacing
print(np.array2string(np.round(weights.value, 2), separator=" "))
```

```result
[0.75 0.25]
```

## 📊 Features

cvxrisk provides several risk models:

### Sample Covariance

The simplest risk model based on the sample covariance matrix:

```python
from cvxrisk.sample import SampleCovariance
import numpy as np

riskmodel = SampleCovariance(num=2)
riskmodel.update(
    cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
    lower_assets=np.zeros(2),
    upper_assets=np.ones(2),
)
# Reconstruct covariance from Cholesky factor and print
cov_est = riskmodel.parameter["chol"].value.T @ riskmodel.parameter["chol"].value
print(np.array2string(cov_est, precision=1))
```
```result
[[1.  0.5]
 [0.5 2. ]]
```

### Factor Risk Models

Factor models reduce dimensionality by projecting asset returns onto a
smaller set of factors:

```python
import numpy as np
from cvxrisk.factor import FactorModel
from cvxrisk.linalg import pca
import pandas as pd

# Create some sample returns data
a = 100
m = 25
returns = pd.DataFrame(np.random.randn(a, m))

# Compute principal components (deterministic using a fixed seed for reproducibility)
np.random.seed(0)
factors = pca(returns, n_components=10)

# Create and update the factor model
model = FactorModel(assets=m, k=10)
model.update(
    cov=factors.cov.values,
    exposure=factors.exposure.values,
    idiosyncratic_risk=factors.idiosyncratic.std().values,
    lower_assets=np.zeros(m),
    upper_assets=np.ones(m),
    lower_factors=-0.1*np.ones(10),
    upper_factors=0.1*np.ones(10),
)

# Verify the model has the correct dimensions
print(model.parameter["exposure"].value.shape)
```
```result
(10, 25)
```

Factor risk models use the projection of the weight vector into a lower
dimensional subspace, e.g. each asset is the linear combination of $k$ factors.

$$r_i = \sum_{j=1}^k f_j \beta_{ji} + \epsilon_i$$

The factor time series are $f_1, \ldots, f_k$. The loadings are the coefficients
$\beta_{ji}$.
The residual returns $\epsilon_i$ are assumed to be uncorrelated with the factors.

Any position $w$ in weight space projects to a position $y = \beta^T w$ in
factor space. The variance for a position $w$ is the sum of the variance of the
systematic returns explained by the factors and the variance of the
idiosyncratic returns.

$$Var(r) = Var(\beta^T w) + Var(\epsilon w)$$

We assume the residual returns are uncorrelated and hence

$$Var(r) = y^T \Sigma_f y + \sum_i w_i^2 Var(\epsilon_i)$$

where $\Sigma_f$ is the covariance matrix of the factors and $Var(\epsilon_i)$
is the variance of the idiosyncratic returns.

### Conditional Value at Risk (CVaR)

CVaR measures the expected loss in the worst-case scenarios:

```python
import numpy as np
from cvxrisk.cvar import CVar

# Create some sample historical returns (deterministic)
np.random.seed(0)
historical_returns = np.random.randn(50, 14)

# Create and update the CVaR model
model = CVar(alpha=0.95, n=50, m=14)
model.update(
    returns=historical_returns,
    lower_assets=np.zeros(14),
    upper_assets=np.ones(14),
)

# Verify the model parameters
print(model.alpha)
print(model.parameter["R"].value.shape)
```

```result
0.95
(50, 14)
```
## 📚 Documentation

For more detailed documentation and examples, visit our [documentation site](http://www.cvxgrp.org/cvxrisk/book).

## 🛠️ Development

cvxrisk uses modern Python development tools:

```bash
# Install development dependencies
make install

# Run tests
make test

# Format code
make fmt

# Start interactive notebooks
make marimo
```

## 📄 License

cvxrisk is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md).
