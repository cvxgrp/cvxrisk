import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    from pathlib import Path

    import cvxpy as cp
    import numpy as np
    import pandas as pd

    from cvx.portfolio.min_risk import minrisk_problem
    from cvx.risk.sample import SampleCovariance
    from cvx.simulator import Builder

    pd.options.plotting.backend = "plotly"

    path = Path(__file__).parent
    return (
        Builder,
        Path,
        SampleCovariance,
        cp,
        minrisk_problem,
        np,
        path,
        pd,
    )


@app.cell
def __(path, pd):
    # Load some historic stock prices
    prices = pd.read_csv(path / "data" / "stock_prices.csv", index_col=0, parse_dates=True, header=0)

    # Estimate a series of historic covariance matrices
    returns = prices.pct_change().dropna(axis=0, how="all")
    return prices, returns


@app.cell
def __(
    Builder,
    SampleCovariance,
    cp,
    minrisk_problem,
    np,
    prices,
    returns,
):
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
    return cov, start


@app.cell
def __(Builder, cp, minrisk_problem, np, prices, returns, start):
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
    return (CVar,)


if __name__ == "__main__":
    app.run()
