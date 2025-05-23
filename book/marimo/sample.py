import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(r"""# Sample covariance""")
    return


@app.cell
def __():
    import cvxpy as cp
    import numpy as np

    from cvx.portfolio.min_risk import minrisk_problem
    from cvx.random import rand_cov
    from cvx.risk.sample import SampleCovariance

    _n = 6
    return SampleCovariance, cp, minrisk_problem, np, rand_cov


@app.cell
def __(SampleCovariance, cp, minrisk_problem):
    def problem(n):
        weights = cp.Variable(n)
        _riskmodel = SampleCovariance(num=n)
        _problem = minrisk_problem(_riskmodel, weights)

        return _problem, _riskmodel

    return (problem,)


@app.cell
def __(np, problem, rand_cov):
    _n = 50
    _a = rand_cov(_n - 2)
    _problem, _riskmodel = problem(_n)
    for _i in range(100):
        _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
        _problem.solve(solver="CLARABEL")
    return


@app.cell
def __(np, problem, rand_cov):
    _n = 50
    _a = rand_cov(_n - 2)
    for _i in range(100):
        _problem, _riskmodel = problem(_a.shape[0])
        _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
        _problem.solve(solver="CLARABEL")
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
