"""Demo of the risk and simulator packages."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import cvxpy as cp
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    from cvxsimulator import Builder

    from cvx.risk.portfolio import minrisk_problem
    from cvx.risk.sample import SampleCovariance

    pd.options.plotting.backend = "plotly"

    from cvxsimulator import __version__ as simulator_version

    from cvx.risk import __version__ as risk_version

    print(f"Risk version: {risk_version}")
    print(f"Simulator version: {simulator_version}")


@app.cell
def _():
    # Load some historic stock prices
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"), try_parse_dates=True)

    prices = prices.to_pandas().set_index("date")

    # Estimate a series of historic covariance matrices
    returns = prices.pct_change().dropna(axis=0, how="all")
    return prices, returns


@app.cell
def _(prices, returns):
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
        _problem.solve(solver="CLARABEL")

        _builder.weights = _w.value
        _builder.aum = _state.aum

    _portfolio = _builder.build()

    _portfolio.nav.plot()
    return (start,)


@app.cell
def _(prices, returns, start):
    from cvxrisk.cvar import CVar

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
        _problem.solve(solver="CLARABEL")

        _builder.weights = _w.value
        _builder.aum = _state.aum

    _portfolio = _builder.build()
    _portfolio.nav.plot()
    return


if __name__ == "__main__":
    app.run()
