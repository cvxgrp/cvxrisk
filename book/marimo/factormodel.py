# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "numpy",
#     "polars",
#     "cvxrisk",
#     "cvx-linalg"
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
    import marimo as mo
    import numpy as np
    import polars as pl
    from cvx.linalg import pca

    from cvx.core.variable import Variable
    from cvx.risk.factor import FactorModel
    from cvx.risk.portfolio import minrisk_problem


@app.cell
def _():
    # Load some historic stock prices
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"), try_parse_dates=True)

    asset_cols = [c for c in prices.columns if c != "date"]

    # Compute percentage returns in polars
    returns = prices.select(pl.col(asset_cols).pct_change()).drop_nulls()
    return prices, returns, asset_cols


@app.cell
def _(returns):
    factors = pca(returns=returns.to_numpy(), n_components=10)
    return (factors,)


@app.cell
def _(factors, returns):
    model = FactorModel(assets=len(returns.columns), k=10)

    # update the model parameters
    model.update(
        cov=factors.cov,
        exposure=factors.exposure,
        idiosyncratic_risk=factors.idiosyncratic.std(axis=0, ddof=1),
        lower_assets=np.zeros(model.assets),
        upper_assets=np.ones(model.assets),
        lower_factors=-0.1 * np.ones(model.k),
        upper_factors=0.1 * np.ones(model.k),
    )

    # test the risk model with uniform weights
    weights = 0.05 * np.ones(model.assets)
    risk = model.estimate(weights)
    print(risk)
    return (model,)


@app.cell
def _(asset_cols, model) -> None:
    w = Variable(model.assets)
    y = Variable(model.k)

    problem = minrisk_problem(model, w, y=y)
    problem.solve()

    print(pl.DataFrame({"asset": asset_cols, "weight": w.value}))
    print(model.estimate(w.value))

    # check the solution
    print(f"Check sum of weights: {np.isclose(w.value.sum(), 1.0)}")
    print(f"Check all weights non-negative: {np.all(w.value > -0.01)}")
    print(y.value)
    return


if __name__ == "__main__":
    app.run()
