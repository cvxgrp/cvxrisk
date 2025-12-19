# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "cvxpy-base",
#     "numpy",
#     "pandas",
#     "polars",
#     "clarabel==0.11.1",
#     "cvxrisk",
#     "pyarrow"
# ]
#
# [tool.uv.sources]
# cvxrisk = { path = "../..", editable=true }
# ///

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

    from cvx.risk.factor import FactorModel
    from cvx.risk.linalg import pca
    from cvx.risk.portfolio import minrisk_problem


@app.cell
def _():
    # Load some historic stock prices
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"), try_parse_dates=True)

    prices = prices.to_pandas().set_index("date")

    # Estimate a series of historic covariance matrices
    returns = prices.pct_change().dropna(axis=0, how="all")
    return prices, returns


@app.cell
def _(returns):
    factors = pca(returns=returns, n_components=10)
    return (factors,)


@app.cell
def _(factors, returns):
    model = FactorModel(assets=len(returns.columns), k=10)

    # update the model parameters
    model.update(
        cov=factors.cov,
        exposure=factors.exposure.values,
        idiosyncratic_risk=factors.idiosyncratic.std().values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=-0.1 * np.ones(10),
        upper_factors=0.1 * np.ones(10),
    )

    # test the risk model with uniform weights
    weights = 0.05 * np.ones(20)
    risk = model.estimate(weights).value
    print(risk)
    return (model,)


@app.cell
def _(model, prices):
    w = cp.Variable(20)
    y = cp.Variable(10)

    problem = minrisk_problem(model, w, y=y)
    problem.solve(solver="CLARABEL")

    print(pd.Series(data=w.value, index=prices.columns))
    print(model.estimate(w, y=y).value)

    # check the solution
    print(f"Check sum of weights: {np.isclose(w.value.sum(), 1.0)}")
    print(f"Check all weights non-negative: {np.all(w.value > -0.01)}")
    print(y.value)
    return


if __name__ == "__main__":
    app.run()
