#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
"""Shared helper for building and solving Clarabel conic programs.

This module provides the :class:`ConeProgramBuilder`, a small utility that
accumulates the constraint blocks and cones of a conic program

    minimize    q' x
    subject to  A x + s = b,  s in K

and solves it with Clarabel. The risk models use it to express their
minimum-risk problems without repeating the boilerplate of stacking
constraint blocks, handling user-supplied linear constraints, and
invoking the solver.

Example:
    Minimize x subject to x >= 1:

    >>> import clarabel
    >>> import numpy as np
    >>> from cvx.core import ConeProgramBuilder
    >>> builder = ConeProgramBuilder(n_vars=1)
    >>> builder.add(np.array([[-1.0]]), np.array([-1.0]), clarabel.NonnegativeConeT(1))
    >>> solution, status = builder.solve(q=np.array([1.0]))
    >>> status
    'Solved'
    >>> bool(np.isclose(solution.x[0], 1.0))
    True

"""

from __future__ import annotations

from typing import Any

import clarabel
import numpy as np
from scipy import sparse

# User-supplied linear constraint: (a, lb, ub) meaning lb <= a @ w <= ub.
# Use None for one-sided bounds; lb == ub yields an equality constraint.
LinearConstraint = tuple[np.ndarray, float | None, float | None]


class ConeProgramBuilder:
    """Accumulate constraint blocks for a Clarabel conic program and solve it.

    Constraints are added as blocks of the stacked system ``A x + s = b`` with
    slack ``s`` in the corresponding cone. The builder assembles the blocks
    into sparse matrices and runs the Clarabel solver.

    Attributes:
        n_vars: Total number of decision variables in the program.

    """

    def __init__(self, n_vars: int) -> None:
        """Create an empty builder for a program with ``n_vars`` variables.

        Args:
            n_vars: Total number of decision variables.

        """
        self.n_vars = n_vars
        self._a_blocks: list[sparse.csr_matrix] = []
        self._b_blocks: list[np.ndarray] = []
        self._cones: list[Any] = []

    def block(self, rows: int) -> np.ndarray:
        """Return a zero matrix with ``rows`` rows and one column per variable.

        Args:
            rows: Number of constraint rows in the block.

        Returns:
            A zero matrix of shape ``(rows, n_vars)`` ready to be filled in.

        """
        return np.zeros((rows, self.n_vars))

    def add(self, a_block: np.ndarray | sparse.spmatrix, b_block: np.ndarray, cone: Any) -> None:
        """Append the constraint block ``a_block @ x + s = b_block`` with ``s`` in ``cone``.

        Args:
            a_block: Constraint matrix block of shape ``(rows, n_vars)``,
                either dense or scipy sparse. Blocks are stored in sparse
                form so large structured constraints stay memory-efficient.
            b_block: Right-hand side vector of length ``rows``.
            cone: Clarabel cone for the slack of this block.

        """
        self._a_blocks.append(sparse.csr_matrix(a_block))
        self._b_blocks.append(b_block)
        self._cones.append(cone)

    def add_sum_constraint(self, cols: slice, total: float = 1.0) -> None:
        """Constrain the variables selected by ``cols`` to sum to ``total``.

        Args:
            cols: Column slice selecting the variables.
            total: Required sum of the selected variables.

        """
        a = self.block(1)
        a[0, cols] = 1.0
        self.add(a, np.array([total]), clarabel.ZeroConeT(1))

    def add_variable_bounds(self, cols: slice, lower: np.ndarray, upper: np.ndarray) -> None:
        """Add elementwise bounds ``lower <= x[cols] <= upper``.

        The identity blocks are built directly in sparse form, so the cost is
        O(m) rather than O(m * n_vars).

        Args:
            cols: Column slice selecting the variables to bound.
            lower: Lower bound for each selected variable.
            upper: Upper bound for each selected variable.

        """
        m = len(lower)
        eye = sparse.identity(m, format="csr")
        left = sparse.csr_matrix((m, cols.start))
        right = sparse.csr_matrix((m, self.n_vars - cols.stop))
        self.add(sparse.hstack([left, -eye, right]), -lower, clarabel.NonnegativeConeT(m))
        self.add(sparse.hstack([left, eye, right]), upper, clarabel.NonnegativeConeT(m))

    def add_linear_constraints(self, constraints: list[LinearConstraint], cols: slice) -> None:
        """Add user-supplied linear constraints ``lb <= a @ x[cols] <= ub``.

        Args:
            constraints: List of ``(a, lb, ub)`` tuples. Use ``None`` to drop
                one side; ``lb == ub`` produces an equality constraint.
            cols: Column slice the coefficient vectors refer to (typically the
                portfolio weights).

        """
        for coeffs, lower, upper in constraints:
            a = np.asarray(coeffs)
            if lower is not None and upper is not None and lower == upper:
                row = self.block(1)
                row[0, cols] = a
                self.add(row, np.array([lower]), clarabel.ZeroConeT(1))
                continue
            if lower is not None:
                row = self.block(1)
                row[0, cols] = -a
                self.add(row, np.array([-lower]), clarabel.NonnegativeConeT(1))
            if upper is not None:
                row = self.block(1)
                row[0, cols] = a
                self.add(row, np.array([upper]), clarabel.NonnegativeConeT(1))

    def solve(self, q: np.ndarray) -> tuple[Any, str]:
        """Assemble the accumulated blocks, solve with Clarabel, and return the result.

        Args:
            q: Linear objective vector of length ``n_vars``.

        Returns:
            Tuple ``(solution, status)`` where ``solution`` is the Clarabel
            solution object and ``status`` its status as a string.

        """
        p = sparse.csc_matrix((self.n_vars, self.n_vars))
        a = sparse.vstack(self._a_blocks, format="csc")
        b = np.concatenate(self._b_blocks)

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        solution = clarabel.DefaultSolver(p, q, a, b, self._cones, settings).solve()
        return solution, str(solution.status)
