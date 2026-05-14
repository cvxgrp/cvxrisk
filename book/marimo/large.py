# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "numpy",
#     "cvxrisk"
# ]
#
# [tool.uv.sources]
# cvxrisk = { path = "../..", editable=true }
# ///

"""Large problem with 1000 assets and 100 factors."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import uuid

    import marimo as mo
    import numpy as np

    from cvx.core.variable import Variable
    from cvx.risk.factor import FactorModel
    from cvx.risk.portfolio import minrisk_problem


@app.cell
def _():
    mo.md(r"""# Large problem with 1000 assets and 100 factors""")
    return


@app.cell
def _random():
    # Create a single random number generator instance
    rng = np.random.default_rng(42)

    def random_weights(assets):
        """Construct a vector of non-negative random weights. Their sum shall be 1."""
        w = rng.random(len(assets))
        return w / w.sum()

    def random_factors(t, n=2, const_factor=True):
        """Construct N random factor time series for T timestamps."""
        data = rng.standard_normal((t, n))
        if const_factor:
            data = np.column_stack([data, np.ones(t)])
        return data

    def random_beta(assets, factors):
        """Construct a random exposure matrix."""
        return rng.standard_normal((factors.shape[1], len(assets)))

    def random_noise(frame):
        """Construct a frame of random noise with exactly the same dimensions as the input frame."""
        return rng.standard_normal(frame.shape)

    def random_assets(num):
        """Construct a vector of random assets."""
        return [str(uuid.uuid4())[:7] for _ in range(num)]

    return random_weights, random_assets, random_noise, random_factors, random_beta


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
def _(ret):
    triangle = FactorModel(assets=ret.shape[1], k=100)
    return (triangle,)


@app.cell
def _(beta, factors, ret, triangle):
    w = Variable(1000)
    y = Variable(100)
    _problem = minrisk_problem(triangle, w, y=y)
    triangle.update(
        exposure=beta,
        cov=np.cov(factors.T),
        idiosyncratic_risk=(ret - factors @ beta).std(axis=0),
        lower_assets=np.zeros(1000),
        upper_assets=np.ones(1000),
        lower_factors=-0.1 * np.ones(100),
        upper_factors=0.1 * np.ones(100),
    )
    return w, y


@app.cell
def _(beta, factors, ret, triangle, w, y):
    for _i in range(1):
        _problem = minrisk_problem(triangle, w, y=y)
        triangle.update(
            exposure=beta,
            cov=np.cov(factors.T),
            idiosyncratic_risk=(ret - factors @ beta).std(axis=0),
            lower_assets=np.zeros(1000),
            upper_assets=np.ones(1000),
            lower_factors=-0.1 * np.ones(100),
            upper_factors=0.1 * np.ones(100),
        )
        _problem.solve()
    return


if __name__ == "__main__":
    app.run()
