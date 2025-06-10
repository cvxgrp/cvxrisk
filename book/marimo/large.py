"""Large problem with 1000 assets and 100 factors."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Large problem with 1000 assets and 100 factors""")
    return


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

    return (mo, cvx, np, pd)


@app.cell
def _random(np, pd):
    import uuid

    # Create a single random number generator instance
    rng = np.random.default_rng(42)

    def random_weights(assets):
        """Construct a vector of non-negative random weights. Their sum shall be 1."""
        # Get some random weights
        weights = pd.Series(index=assets, data=rng.random(len(assets)))
        return weights / weights.sum()

    def random_factors(t, n=2, const_factor=True):
        """Construct N random factor time series for T timestamps."""
        factors = pd.DataFrame(
            index=range(1, t + 1),
            columns=[f"F{i}" for i in range(n)],
            data=rng.standard_normal((t, n)),
        )
        # add the constant factor
        if const_factor:
            factors["const"] = 1
        return factors

    def random_beta(assets, factors):
        """Construct a random exposure matrix."""
        data = rng.standard_normal((factors.shape[1], len(assets)))
        return pd.DataFrame(columns=assets, index=factors.columns, data=data)

    def random_noise(frame):
        """Construct a frame of random noise with exactly the same dimensions as the input frame."""
        return pd.DataFrame(
            columns=frame.columns,
            index=frame.index,
            data=rng.standard_normal((frame.shape[0], frame.shape[1])),
        )

    def random_assets(num):
        """Construct a vector of random assets."""
        return [str(uuid.uuid4())[:7] for _ in range(num)]

    return random_weights, random_assets, random_noise, random_factors, random_beta


@app.cell
def _():
    from cvxrisk.factor import FactorModel
    from cvxrisk.portfolio import minrisk_problem

    return (
        FactorModel,
        minrisk_problem,
    )


@app.cell
def _(random_factors):
    t = 2000
    factors = random_factors(t=t, n=100, const_factor=False)
    return (factors,)


@app.cell
def _(factors, random_assets, random_beta):
    beta = random_beta(assets=random_assets(1000), factors=factors)
    return (beta,)


@app.cell
def _(beta, factors, random_noise):
    ret = factors @ beta + 0.01 * random_noise(factors @ beta)
    return (ret,)


@app.cell
def _(FactorModel, ret):
    triangle = FactorModel(assets=len(ret.columns), k=100)
    return (triangle,)


@app.cell
def _(beta, cvx, factors, minrisk_problem, np, pd, ret, triangle):
    w = cvx.Variable(1000)
    y = cvx.Variable(100)
    _problem = minrisk_problem(triangle, w, y=y)
    triangle.update(
        exposure=beta.values,
        cov=factors.cov().values,
        idiosyncratic_risk=pd.DataFrame(data=ret - factors @ beta, index=ret.index, columns=ret.columns).std().values,
        lower_assets=np.zeros(1000),
        upper_assets=np.ones(1000),
        lower_factors=-0.1 * np.ones(100),
        upper_factors=0.1 * np.ones(100),
    )
    return w, y


@app.cell
def _(beta, factors, minrisk_problem, np, pd, ret, triangle, w, y):
    for _i in range(1):
        _problem = minrisk_problem(triangle, w, y=y)
        triangle.update(
            exposure=beta.values,
            cov=factors.cov().values,
            idiosyncratic_risk=pd.DataFrame(data=ret - factors @ beta, index=ret.index, columns=ret.columns)
            .std()
            .values,
            lower_assets=np.zeros(1000),
            upper_assets=np.ones(1000),
            lower_factors=-0.1 * np.ones(100),
            upper_factors=0.1 * np.ones(100),
        )
        _problem.solve(ignore_dpp=True, solver="CLARABEL")
    return


if __name__ == "__main__":
    app.run()
