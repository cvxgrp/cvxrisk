"""Sample covariance."""

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Sample covariance""")
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

    import cvxpy as cp
    import marimo as mo
    import numpy as np

    from cvxrisk.portfolio import minrisk_problem
    from cvxrisk.random import rand_cov
    from cvxrisk.sample import SampleCovariance

    _n = 6
    return SampleCovariance, cp, minrisk_problem, mo, np, rand_cov


@app.cell
def _(SampleCovariance, cp, minrisk_problem):
    def problem(n):
        weights = cp.Variable(n)
        _riskmodel = SampleCovariance(num=n)
        _problem = minrisk_problem(_riskmodel, weights)

        return _problem, _riskmodel

    return (problem,)


@app.cell
def _(np, problem, rand_cov):
    _n = 50
    _a = rand_cov(_n - 2)
    _problem, _riskmodel = problem(_n)
    for _i in range(100):
        _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
        _problem.solve(solver="CLARABEL")
    return


@app.cell
def _(np, problem, rand_cov):
    _n = 50
    _a = rand_cov(_n - 2)
    for _i in range(100):
        _problem, _riskmodel = problem(_a.shape[0])
        _riskmodel.update(cov=_a, lower_assets=np.zeros(48), upper_assets=np.ones(48))
        _problem.solve(solver="CLARABEL")
    return


if __name__ == "__main__":
    app.run()
