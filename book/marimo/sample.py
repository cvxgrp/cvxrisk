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

"""Sample covariance."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()

with app.setup:
    import marimo as mo
    import numpy as np

    from cvx.core.variable import Variable
    from cvx.risk.portfolio import minrisk_problem
    from cvx.risk.rand import rand_cov
    from cvx.risk.sample import SampleCovariance


@app.cell
def _():
    mo.md(r"""# Sample covariance""")
    return


@app.cell
def _():
    def problem(n):
        weights = Variable(n)
        _riskmodel = SampleCovariance(num=n)
        _problem = minrisk_problem(_riskmodel, weights)

        return _problem, _riskmodel

    return (problem,)


@app.cell
def _(problem):
    _n = 50
    _a = rand_cov(_n - 2)
    _problem, _riskmodel = problem(_n)
    for _i in range(100):
        _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
        _problem.solve()
    return


@app.cell
def _(problem):
    _n = 50
    _a = rand_cov(_n - 2)
    for _i in range(100):
        _problem, _riskmodel = problem(_a.shape[0])
        _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
        _problem.solve()
    return


if __name__ == "__main__":
    app.run()
