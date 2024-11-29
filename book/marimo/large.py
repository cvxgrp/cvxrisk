import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""# Large problem with 1000 assets and 100 factors""")
    return


@app.cell
def __():
    import cvxpy as cvx
    import numpy as np
    import pandas as pd
    from util.random import random_assets, random_beta, random_factors, random_noise

    from cvx.portfolio.min_risk import minrisk_problem
    from cvx.risk.factor import FactorModel

    return (
        FactorModel,
        cvx,
        minrisk_problem,
        np,
        pd,
        random_assets,
        random_beta,
        random_factors,
        random_noise,
    )


@app.cell
def __(random_factors):
    T = 2000
    factors = random_factors(T=T, N=100, const_factor=False)
    return T, factors


@app.cell
def __(factors, random_assets, random_beta):
    beta = random_beta(assets=random_assets(1000), factors=factors)
    return (beta,)


@app.cell
def __(beta, factors, random_noise):
    ret = factors @ beta + 0.01 * random_noise(factors @ beta)
    return (ret,)


@app.cell
def __(FactorModel, ret):
    triangle = FactorModel(assets=len(ret.columns), k=100)
    return (triangle,)


@app.cell
def __(beta, cvx, factors, minrisk_problem, np, pd, ret, triangle):
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
    return w, y


@app.cell
def __(beta, factors, minrisk_problem, np, pd, ret, triangle, w, y):
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
    return (i,)


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
