# /// script
# dependencies = [
#     "marimo==0.18.4",
#     "numpy",
#     "polars",
#     "cvx-linalg",
#     "cvxrisk"
# ]
#
# [tool.uv.sources]
# cvxrisk = { path = "../..", editable=true }
# ///

"""Tilting problem."""

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl
    from cvx.linalg import rand_cov

    from cvx.core.variable import Variable
    from cvx.risk.portfolio import minrisk_problem
    from cvx.risk.sample import SampleCovariance


@app.cell
def _() -> None:
    mo.md(
        r"""
    This notebook is based on an idea by Nicholas Gunther

    We welcome such ideas very much as they help to improve our tools
    """
    )
    return


@app.cell
def _() -> None:
    mo.md(
        r"""
    Can you give some guidance on how to include constraints in cvxrisk?
    It doesn't seem to appear in the short documentation.
    For specificity, here's a question related to price-earnings "tilts".
    I illustrate it with a toy example.

    Let S be a sample covariance matrix, let w_SP be the (market cap) weights of stocks in the S&P 500 index
    and let V be a vector of earning-price normalized "z-scores".  I want to solve the following for w:

    Minimize (w - w_SP).transpose @ S @ (w - w_SP)

    subject to:

    w >= 0

    sum(w) = 1

    w.dot(V) = 0.5

    The exercise is to find a portfolio that is "close" to the S&P 500 index but
    has lower price-earnings ratios, a "value" tilt.  I can work through this in cvxpy,
    somewhat laboriously, but can cvxrisk accommodate these constraints easily?
    """
    )
    return


@app.cell
def _() -> None:
    mo.md(
        r"""
    We shall first observe that if we drop the constraint w.dot(V)
    the optimal vector is just w = w_SP assuming that w_SP >= 0 and sum(w_SP) = 1.
    """
    )
    return


@app.cell
def _() -> None:
    mo.md(
        r"""
    For this problem we have enhanced the builtin minrisk function.
    One can now inject linear constraints as (a, lb, ub) tuples.
    """
    )
    return


@app.cell
def _():
    # Let's start without the tilting constraint
    assets = ["A", "B", "C", "D", "E"]
    s = pl.DataFrame(dict(zip(assets, rand_cov(len(assets)).T, strict=False)))

    # those are the market weights for our 5 markets
    w_sp = pl.Series(values=[0.1, 0.2, 0.3, 0.1, 0.3])

    # the Variable for the weights
    weights = Variable(len(assets))

    # Let's define a sample covariance riskmodel
    riskmodel = SampleCovariance(num=len(assets))
    riskmodel.update(cov=s.to_numpy(), lower_assets=np.zeros(len(assets)), upper_assets=np.ones(len(assets)))

    # the tilting vector
    v = pl.Series(values=[0.1, 0.1, 0.5, 0.0, 0.5])
    return assets, riskmodel, v, w_sp, weights


@app.cell
def _(assets, riskmodel, w_sp, weights) -> None:
    # without the tilting constraint. We reproduce (as predicted) w_sp.
    problem = minrisk_problem(riskmodel=riskmodel, weights=weights, base=w_sp.to_numpy(), constraints=[])

    problem.solve()

    solution = pl.DataFrame({"asset": assets, "weight": weights.value})
    print(solution)
    return


@app.cell
def _(assets, riskmodel, v, w_sp, weights) -> None:
    # Now we specify the tilting constraint: v @ w == 0.5 as (a, lb, ub) with lb == ub
    constraints = [(v.to_numpy(), 0.5, 0.5)]
    # We inject the constraints
    problem_tilt = minrisk_problem(riskmodel=riskmodel, weights=weights, base=w_sp.to_numpy(), constraints=constraints)

    problem_tilt.solve()

    # The solution is different from the previous problem
    solution_tilt = pl.DataFrame({"asset": assets, "weight": weights.value})
    print(solution_tilt)
    # We check whether the tilting constraint is respected
    print("Tilting value. Should be close to 0.5:")
    print(weights.value @ v.to_numpy())
    return


if __name__ == "__main__":
    app.run()
