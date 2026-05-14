# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "pandas",
#     "polars",
#     "jquantstats",
#     "cvxrisk"
# ]
#
# [tool.uv.sources]
# cvxrisk = { path = "../..", editable=true }
# ///

"""Demo of the risk packages with jquantstats performance reporting."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import jquantstats as jqs
    import marimo as mo
    import numpy as np
    import polars as pl

    from cvx.core.variable import Variable
    from cvx.risk import __version__ as risk_version
    from cvx.risk.portfolio import minrisk_problem
    from cvx.risk.sample import SampleCovariance

    print(f"Risk version: {risk_version}")
    print(f"jquantstats version: {jqs.__version__}")


@app.cell
def _():
    # Load some historic stock prices
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"), try_parse_dates=True)

    prices = prices.to_pandas().set_index("date")

    # Compute returns and EWM covariance (requires pandas)
    returns = prices.pct_change().dropna(axis=0, how="all")
    cov = returns.ewm(com=60, min_periods=100).cov().dropna(axis=0, how="all")
    start = cov.index[0][0]
    return prices, returns, cov, start


@app.cell
def _(cov, returns, start):
    # Minimum-risk backtest using SampleCovariance
    cov_dates = set(cov.index.get_level_values(0))
    returns_bt = returns.loc[start:]

    _risk_model = SampleCovariance(num=20)
    _w = Variable(20)
    _problem = minrisk_problem(_risk_model, _w)
    _current_w = None

    _port_rets = []
    for _date in returns_bt.index:
        if _date in cov_dates:
            _risk_model.update(
                cov=cov.loc[_date].values,
                lower_assets=np.zeros(20),
                upper_assets=np.ones(20),
            )
            _problem.solve()
            _current_w = _w.value.copy()

        if _current_w is not None:
            _port_rets.append(float(returns_bt.loc[_date].values @ _current_w))

    _returns_pl = pl.DataFrame(
        {
            "Date": returns_bt.index[-len(_port_rets) :].to_list(),
            "SampleCovariance": _port_rets,
        }
    )
    data_sc = jqs.Data.from_returns(_returns_pl, date_col="Date")
    print(data_sc.stats.sharpe())
    return (data_sc,)


@app.cell
def _(data_sc):
    data_sc.plots.snapshot()


@app.cell
def _(returns, start):
    from cvx.risk.cvar import CVar

    _returns_bt = returns.loc[start:]

    _risk_model = CVar(alpha=0.80, n=40, m=20)
    _w = Variable(20)
    _problem = minrisk_problem(_risk_model, _w)
    _current_w = None

    _port_rets = []
    for _i, _date in enumerate(_returns_bt.index):
        _hist = _returns_bt.iloc[: _i + 1].tail(40)
        if len(_hist) == 40:
            _risk_model.update(
                returns=_hist.values,
                lower_assets=np.zeros(20),
                upper_assets=np.ones(20),
            )
            _problem.solve()
            _current_w = _w.value.copy()

        if _current_w is not None:
            _port_rets.append(float(_returns_bt.iloc[_i].values @ _current_w))

    _returns_pl = pl.DataFrame(
        {
            "Date": _returns_bt.index[-len(_port_rets) :].to_list(),
            "CVar": _port_rets,
        }
    )
    data_cvar = jqs.Data.from_returns(_returns_pl, date_col="Date")
    print(data_cvar.stats.sharpe())
    return (data_cvar,)


@app.cell
def _(data_cvar):
    data_cvar.plots.snapshot()


if __name__ == "__main__":
    app.run()
