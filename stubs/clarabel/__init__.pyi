from typing import Any

import numpy as np
import scipy.sparse as sp

class DefaultSettings:
    verbose: bool
    max_iter: int
    time_limit: float
    tol_gap_abs: float
    tol_gap_rel: float
    tol_feas: float
    tol_infeas_abs: float
    tol_infeas_rel: float
    tol_ktratio: float
    def __init__(self) -> None: ...

class SecondOrderConeT:
    def __init__(self, dim: int) -> None: ...

class ZeroConeT:
    def __init__(self, dim: int) -> None: ...

class NonnegativeConeT:
    def __init__(self, dim: int) -> None: ...

class PowerConeT:
    def __init__(self, alpha: float) -> None: ...

class ExponentialConeT:
    def __init__(self) -> None: ...

class PSDTriangleConeT:
    def __init__(self, dim: int) -> None: ...

class SolverStatus:
    Solved: SolverStatus
    AlmostSolved: SolverStatus
    PrimalInfeasible: SolverStatus
    DualInfeasible: SolverStatus
    AlmostPrimalInfeasible: SolverStatus
    AlmostDualInfeasible: SolverStatus
    MaxIterations: SolverStatus
    MaxTime: SolverStatus
    NumericalError: SolverStatus
    InsufficientProgress: SolverStatus
    CallbackTerminated: SolverStatus
    Unsolved: SolverStatus

class DefaultSolution:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    s: np.ndarray
    status: SolverStatus
    obj_val: float
    obj_val_dual: float
    solve_time: float
    iterations: int
    r_prim: float
    r_dual: float

class DefaultSolver:
    def __init__(
        self,
        P: sp.csc_matrix,
        q: np.ndarray,
        A: sp.csc_matrix,
        b: np.ndarray,
        cones: list[Any],
        settings: DefaultSettings,
    ) -> None: ...
    def solve(self) -> DefaultSolution: ...
