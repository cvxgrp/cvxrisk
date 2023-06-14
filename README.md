# [cvxrisk](http://www.cvxgrp.org/cvxrisk/)

[![PyPI version](https://badge.fury.io/py/cvxrisk.svg)](https://badge.fury.io/py/cvxrisk)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/simulator/blob/master/LICENSE)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxrisk.svg)](https://pypi.python.org/pypi/cvxrisk/)

We provide an abstract `RiskModel` class. The class is designed to be used in conjunction with [cvxpy](https://github.com/cvxpy/cvxpy).
Using this class, we can formulate a function computing a standard minimum variance portfolio as

```python
import cvxpy as cp

from cvx.risk import RiskModel

def minimum_variance(w: cp.Variable, risk_model: RiskModel) -> cp.Problem:
    """Constructs a minimum variance portfolio.

    Args:
        w: cp.Variable representing the portfolio weights.
        risk_model: A risk model.

    Returns:
        A convex optimization problem.
    """
    return cp.Problem(
        cp.Minimize(risk_model.estimate_risk(w)),
        [cp.sum(w) == 1, w >= 0]
    )
```

The risk model is injected into the function.
The function is not aware of the precise risk model used.
All risk models are required to implement the `estimate_risk` method.

## A first example

A first example is a risk model based on the sample covariance matrix.
We construct the risk model as follows

```python
import numpy as np
import cvxpy as cp

from cvx.risk.sample import SampleCovariance

riskmodel = SampleCovariance(num=2)
riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 2.0]])

w = cp.Variable(2)

minimum_variance(w, riskmodel).solve()
print(w.value)
```

The risk model and the actual optimization problem are decoupled.
This is good practice and keeps the code clean and maintainable.

## Risk models

### Sample covariance

We offer two variants of the sample covariance risk model.
The first variant is the `SampleCovariance` class.
It relies on cxxpy's `quad_form` function to compute the variance
```math
w^T \Sigma w.
```
The second variant is the `SampleCovariance_product` class.
It relies on cxxpy's `sum_of_squares` function to compute the variance using
```math
\| \Sigma^{1/2} w \|_2.
```


### Factor risk models

Factor risk models use the projection of the weight vector into a lower
dimensional subspace, e.g. each asset is the linear combination of $k$ factors.
```math
r_i = \sum_{j=1}^k \beta_{ij} f_j + \epsilon_i
```


## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org). Once you have installed poetry you can perform

```bash
poetry install
```

to replicate the virtual environment we have defined in pyproject.toml.

## Kernel

We install [JupyterLab](https://jupyter.org) within your new virtual environment. Executing

```bash
./create_kernel.sh
```

constructs a dedicated [Kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) for the project.
