---
title: Sample
marimo-version: 0.9.27
---

# Sample covariance

```{.python.marimo}
import cvxpy as cp
import numpy as np

from cvx.portfolio.min_risk import minrisk_problem
from cvx.random import rand_cov
from cvx.risk.sample import SampleCovariance

_n = 6
```

```{.python.marimo}
def problem(n):
    weights = cp.Variable(n)
    _riskmodel = SampleCovariance(num=n)
    _problem = minrisk_problem(_riskmodel, weights)

    return _problem, _riskmodel
```

```{.python.marimo}
_n = 50
_a = rand_cov(_n - 2)
_problem, _riskmodel = problem(_n)
for _i in range(100):
    _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
    _problem.solve()
```

```{.python.marimo}
_n = 50
_a = rand_cov(_n - 2)
for _i in range(100):
    _problem, _riskmodel = problem(_a.shape[0])
    _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
    _problem.solve()
```

```{.python.marimo}
import marimo as mo
```