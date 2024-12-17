---
title: Demo
marimo-version: 0.10.4
width: medium
---

```{.python.marimo}
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from cvx.portfolio.min_risk import minrisk_problem
from cvx.risk.sample import SampleCovariance
from cvx.simulator import Builder

pd.options.plotting.backend = "plotly"

path = Path(__file__).parent
```

```{.python.marimo}
# Load some historic stock prices
prices = pd.read_csv(
    path / "data" / "stock_prices.csv", index_col=0, parse_dates=True, header=0
)

# Estimate a series of historic covariance matrices
returns = prices.pct_change().dropna(axis=0, how="all")
```

```{.python.marimo}
cov = returns.ewm(com=60, min_periods=100).cov().dropna(axis=0, how="all")
start = cov.index[0][0]

# Establish a risk model
_risk_model = SampleCovariance(num=20)

# Perform the backtest
_builder = Builder(prices=prices.truncate(before=start), initial_aum=1e6)

_w = cp.Variable(len(_builder.prices.columns))
_problem = minrisk_problem(_risk_model, _w)

for _t, _state in _builder:
    _risk_model.update(
        cov=cov.loc[_t[-1]].values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
    )

    # don't reconstruct the problem in every iteration!
    _problem.solve()

    _builder.weights = _w.value
    _builder.aum = _state.aum

_portfolio = _builder.build()

_portfolio.nav.plot()
```

```{.python.marimo}
from cvx.risk.cvar import CVar

_risk_model = CVar(alpha=0.80, n=40, m=20)

# Perform the backtest
_builder = Builder(prices=prices.truncate(before=start), initial_aum=1e6)

_w = cp.Variable(len(_builder.prices.columns))
_problem = minrisk_problem(_risk_model, _w)

for _t, _state in _builder:
    _risk_model.update(
        returns=returns.truncate(after=_t[-1]).tail(40).values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
    )

    # don't reconstruct the problem in every iteration!
    _problem.solve()

    _builder.weights = _w.value
    _builder.aum = _state.aum

_portfolio = _builder.build()
_portfolio.nav.plot()
```