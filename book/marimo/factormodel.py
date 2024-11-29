import marimo

__generated_with = "0.9.27"
app = marimo.App(width="medium")


@app.cell
def __():
    from pathlib import Path

    import cvxpy as cvx
    import numpy as np
    import pandas as pd

    from cvx.portfolio.min_risk import minrisk_problem
    from cvx.risk.factor import FactorModel
    from cvx.risk.linalg import pca

    path = Path(__file__).parent
    return FactorModel, cvx, minrisk_problem, np, pca, path, pd


@app.cell
def __(path, pd):
    # Load some historic stock prices
    prices = pd.read_csv(
        path / "data" / "stock_prices.csv", index_col=0, parse_dates=True, header=0
    )

    # Estimate a series of historic covariance matrices
    returns = prices.pct_change().dropna(axis=0, how="all")
    return prices, returns


@app.cell
def __(pca, returns):
    factors = pca(returns=returns, n_components=10)
    return (factors,)


@app.cell
def __(FactorModel, factors, np, returns):
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
    model.estimate(weights).value
    return model, weights


@app.cell
def __(cvx, minrisk_problem, model, np, pd, prices):
    w = cvx.Variable(20)
    y = cvx.Variable(10)

    problem = minrisk_problem(model, w, y=y)
    problem.solve()

    print(pd.Series(data=w.value, index=prices.columns))
    print(model.estimate(w, y=y).value)

    # check the solution
    assert np.isclose(w.value.sum(), 1.0)
    assert np.all(w.value > -0.01)
    print(y.value)
    return problem, w, y


if __name__ == "__main__":
    app.run()
