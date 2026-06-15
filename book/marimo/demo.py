# /// script
# dependencies = [
#     "marimo==0.18.4",
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
    from typing import Any

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
def _() -> tuple[list[str], pl.DataFrame, Any, Any]:
    # Load prices and compute returns entirely in polars
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"), try_parse_dates=True)

    asset_cols = [c for c in prices.columns if c != "date"]
    returns = prices.select([pl.col("date"), pl.col(asset_cols).pct_change()]).drop_nulls()

    # EWM covariance via jquantstats (com=60 → span=121, warmup=100)
    data = jqs.Data.from_returns(returns, date_col="date")
    cov = data.utils.exponential_cov(window=121, warmup=100)
    start = next(iter(cov))
    return asset_cols, returns, cov, start


@app.cell
def _(asset_cols: list[str], cov: Any, returns: pl.DataFrame, start: Any) -> tuple[Any]:
    # Minimum-risk backtest using SampleCovariance
    _returns_bt = returns.filter(pl.col("date") >= start)
    _dates = _returns_bt["date"].to_list()
    _ret_values = _returns_bt.select(asset_cols).to_numpy()

    _risk_model = SampleCovariance(num=20)
    _w = Variable(20)
    _problem = minrisk_problem(_risk_model, _w)
    _current_w = None

    _port_rets = []
    for _i, _date in enumerate(_dates):
        if _date in cov:
            _risk_model.update(
                cov=cov[_date],
                lower_assets=np.zeros(20),
                upper_assets=np.ones(20),
            )
            _problem.solve()
            _current_w = _w.value.copy()

        if _current_w is not None:
            _port_rets.append(float(_ret_values[_i] @ _current_w))

    _returns_pl = pl.DataFrame({"date": _dates[-len(_port_rets) :], "SampleCovariance": _port_rets})
    data_sc = jqs.Data.from_returns(_returns_pl, date_col="date")
    print(data_sc.stats.sharpe())
    return (data_sc,)


@app.cell
def _(data_sc: Any) -> None:
    data_sc.plots.snapshot()


@app.cell
def _(returns: pl.DataFrame, start: Any) -> tuple[Any]:
    from cvx.risk.cvar import CVar

    _returns_bt = returns.filter(pl.col("date") >= start)
    _dates = _returns_bt["date"].to_list()
    _ret_values = _returns_bt.select([c for c in _returns_bt.columns if c != "date"]).to_numpy()

    _risk_model = CVar(alpha=0.80, n=40, m=20)
    _w = Variable(20)
    _problem = minrisk_problem(_risk_model, _w)
    _current_w = None

    _port_rets = []
    for _i in range(len(_dates)):
        _hist = _ret_values[max(0, _i - 39) : _i + 1]
        if len(_hist) == 40:
            _risk_model.update(
                returns=_hist,
                lower_assets=np.zeros(20),
                upper_assets=np.ones(20),
            )
            _problem.solve()
            _current_w = _w.value.copy()

        if _current_w is not None:
            _port_rets.append(float(_ret_values[_i] @ _current_w))

    _returns_pl = pl.DataFrame({"date": _dates[-len(_port_rets) :], "CVar": _port_rets})
    data_cvar = jqs.Data.from_returns(_returns_pl, date_col="date")
    print(data_cvar.stats.sharpe())
    return (data_cvar,)


@app.cell
def _(data_cvar: Any) -> None:
    data_cvar.plots.snapshot()


if __name__ == "__main__":
    app.run()
