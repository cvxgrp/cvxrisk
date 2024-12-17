---
title: Large
marimo-version: 0.10.4
---

# Large problem with 1000 assets and 100 factors

```{.python.marimo}
import cvxpy as cvx
import numpy as np
import pandas as pd
from util.random import random_assets, random_beta, random_factors, random_noise

from cvx.portfolio.min_risk import minrisk_problem
from cvx.risk.factor import FactorModel
```

```{.python.marimo}
T = 2000
factors = random_factors(T=T, N=100, const_factor=False)
```

```{.python.marimo}
beta = random_beta(assets=random_assets(1000), factors=factors)
```

```{.python.marimo}
ret = factors @ beta + 0.01 * random_noise(factors @ beta)
```

```{.python.marimo}
triangle = FactorModel(assets=len(ret.columns), k=100)
```

```{.python.marimo}
w = cvx.Variable(1000)
y = cvx.Variable(100)
_problem = minrisk_problem(triangle, w, y=y)
triangle.update(
    exposure=beta.values,
    cov=factors.cov().values,
    idiosyncratic_risk=pd.DataFrame(
        data=ret - factors @ beta, index=ret.index, columns=ret.columns
    )
    .std()
    .values,
    lower_assets=np.zeros(1000),
    upper_assets=np.ones(1000),
    lower_factors=-0.1 * np.ones(100),
    upper_factors=0.1 * np.ones(100),
)
```

```{.python.marimo}
for i in range(1):
    _problem = minrisk_problem(triangle, w, y=y)
    triangle.update(
        exposure=beta.values,
        cov=factors.cov().values,
        idiosyncratic_risk=pd.DataFrame(
            data=ret - factors @ beta, index=ret.index, columns=ret.columns
        )
        .std()
        .values,
        lower_assets=np.zeros(1000),
        upper_assets=np.ones(1000),
        lower_factors=-0.1 * np.ones(100),
        upper_factors=0.1 * np.ones(100),
    )
    _problem.solve(ignore_dpp=True)
```

```{.python.marimo}
import marimo as mo
```