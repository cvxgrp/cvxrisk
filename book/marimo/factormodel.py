"""Demo of the risk and simulator packages."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
async def _():
    # Check if running in WebAssembly environment
    try:
        import sys

        if "pyodide" in sys.modules:
            import micropip

            await micropip.install("cvxrisk")
            await micropip.install("cvxsimulator")
    except ImportError:
        pass

    import cvxpy as cvx
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl

    pd.options.plotting.backend = "plotly"

    from cvx.portfolio import minrisk_problem
    from cvx.risk.factor import FactorModel
    from cvx.risk.linalg import pca

    return FactorModel, cvx, minrisk_problem, mo, cvx, np, pca, pl


@app.cell
def _(mo, pl):
    # Load some historic stock prices
    prices = pl.read_csv(str(mo.notebook_location() / "public" / "stock_prices.csv"), try_parse_dates=True)

    prices = prices.to_pandas().set_index("date")

    # Estimate a series of historic covariance matrices
    returns = prices.pct_change().dropna(axis=0, how="all")
    return prices, returns


@app.cell
def _(pca, returns):
    factors = pca(returns=returns, n_components=10)
    return (factors,)


@app.cell
def _(FactorModel, factors, np, returns):
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
def _(cvx, minrisk_problem, model, np, pd, prices):
    w = cvx.Variable(20)
    y = cvx.Variable(10)

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
