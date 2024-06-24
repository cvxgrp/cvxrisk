# [cvxrisk](http://www.cvxgrp.org/cvxrisk/book)

[![PyPI version](https://badge.fury.io/py/cvxrisk.svg)](https://badge.fury.io/py/cvxrisk)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/simulator/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxrisk?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxrisk)

We provide an abstract `Model` class.
The class is designed to be used in conjunction with [cvxpy](https://github.com/cvxpy/cvxpy).
Using this class, we can formulate a function computing a standard minimum
risk portfolio as

```python
import cvxpy as cp

from cvx.risk import Model


def minimum_risk(w: cp.Variable, risk_model: Model, **kwargs) -> cp.Problem:
    """Constructs a minimum variance portfolio.

    Args:
        w: cp.Variable representing the portfolio weights.
        risk_model: A risk model.

    Returns:
        A convex optimization problem.
    """
    return cp.Problem(
        cp.Minimize(risk_model.estimate(w, **kwargs)),
        [cp.sum(w) == 1, w >= 0] + risk_model.constraints(w, **kwargs)
    )
```

The risk model is injected into the function.
The function is not aware of the precise risk model used.
All risk models are required to implement the `estimate` method.

Note that factor risk models work with weights for the assets but also with
weights for the factors.
To stay flexible we are applying the `**kwargs` pattern to the function above.

## A first example

A first example is a risk model based on the sample covariance matrix.
We construct the risk model as follows

```python
import numpy as np
import cvxpy as cp

from cvx.risk.sample import SampleCovariance

riskmodel = SampleCovariance(num=2)
w = cp.Variable(2)
problem = minimum_risk(w, riskmodel)

riskmodel.update(cov=np.array([[1.0, 0.5], [0.5, 2.0]]))
problem.solve()
print(w.value)
```

The risk model and the actual optimization problem are decoupled.
This is good practice and keeps the code clean and maintainable.

In a backtest we don't have to reconstruct the problem in every iteration.
We can simply update the risk model with the new data and solve the problem
again. The implementation of the risk models is flexible enough to deal with
changing dimensions of the underlying weight space.

## Risk models

### Sample covariance

We offer a `SampleCovariance` class as seen above.

### Factor risk models

Factor risk models use the projection of the weight vector into a lower
dimensional subspace, e.g. each asset is the linear combination of $k$ factors.

```math
r_i = \sum_{j=1}^k f_j \beta_{ji} + \epsilon_i
```

The factor time series are $f_1, \ldots, f_k$. The loadings are the coefficients
$\beta_{ji}$.
The residual returns $\epsilon_i$ are assumed to be uncorrelated with the f
actors.

Any position $w$ in weight space projects to a position $y = \beta^T w$ in
factor space. The variance for a position $w$ is the sum of the variance of the
systematic returns explained by the factors and the variance of the
idiosyncratic returns.

```math
Var(r) = Var(\beta^T w) + Var(\epsilon w)
```

We assume the residual returns are uncorrelated and hence

```math
Var(r) = y^T \Sigma_f y + \sum_i w_i^2 Var(\epsilon_i)
```

where $\Sigma_f$ is the covariance matrix of the factors and $Var(\epsilon_i)$
is the variance of the idiosyncratic returns.

Factor risk models are widely used in practice. Usually two scenarios are
distinguished. A first route is to rely on estimates for the factor covariance
matrix $\Sigma_f$, the loadings $\beta$ and the volatilities of the
idiosyncratic returns $\epsilon_i$. Usually those quantities are provided by
external parties, e.g. Barra or Axioma.

An alternative would be to start with the estimation of factor time series
$f_1, \ldots, f_k$.
Usually they are estimated via a principal component analysis (PCA) of the
asset returns.  It is then a simple linear regression to compute the loadings
$\beta$. The volatilities of the idiosyncratic returns $\epsilon_i$ are computed
as the standard deviation of the observed residuals.
The factor covariance matrix $\Sigma_f$ may even be diagonal in this case as the
factors are orthogonal.

We expose a method to compute the first $k$ principal components.

### cvar

We currently also support the conditional value at risk (CVaR) as a risk
measure.

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the jupyter lab.
