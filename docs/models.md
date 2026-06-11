# Mathematical formulations

cvxrisk solves minimum-risk portfolio problems directly with the
[Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) conic solver.
This page documents the exact conic program each risk model builds in its
`solve_minrisk` method, so the implementations can be audited against the
mathematics.

All problems share the budget constraint and weight bounds

$$\sum_i w_i = 1, \qquad \ell_w \le w \le u_w,$$

plus any user-supplied linear constraints $\ell \le a^T w \le u$
(equalities when $\ell = u$). A base portfolio $b$ (default $0$) turns the
problem into tracking-error minimization over the active position $w - b$.

## Sample covariance

With $R$ the upper-triangular Cholesky factor of the covariance matrix
($R^T R = \Sigma$), the problem is the second-order cone program

$$\min_{t,\,w}\ t \quad \text{s.t.} \quad \lVert R (w - b) \rVert_2 \le t$$

over the variables $x = [t, w]$. The optimal $t$ is the portfolio
volatility $\sqrt{(w-b)^T \Sigma (w-b)}$.

## Factor model

With factor exposure $\beta \in \mathbb{R}^{k \times n}$, factor covariance
Cholesky factor $R_f$ ($R_f^T R_f = \Sigma_f$), and idiosyncratic
volatilities $\sigma$, the problem introduces the factor position
$y = \beta w$ as an explicit variable:

$$\min_{t,\,w,\,y}\ t \quad \text{s.t.} \quad
\left\lVert \begin{bmatrix} R_f\,(y - \beta b) \\ \operatorname{diag}(\sigma)\,(w - b) \end{bmatrix} \right\rVert_2 \le t,
\qquad y = \beta w, \qquad \ell_y \le y \le u_y$$

over $x = [t, w, y]$. The optimal $t$ equals the total tracking error
$\sqrt{(w-b)^T (\beta^T \Sigma_f \beta + \operatorname{diag}(\sigma^2)) (w-b)}$,
combining systematic and idiosyncratic risk of the active position.

## Conditional Value at Risk

Given scenario returns $R \in \mathbb{R}^{T \times n}$ and confidence level
$\alpha$, the model minimizes the expected loss in the worst
$k = \lfloor T (1-\alpha) \rfloor$ scenarios using the
Rockafellar–Uryasev linear program over $x = [w, \gamma, u]$:

$$\min_{w,\,\gamma,\,u}\ \gamma + \frac{1}{k} \sum_{t=1}^{T} u_t
\quad \text{s.t.} \quad u \ge -R\,(w - b) - \gamma, \qquad u \ge 0,$$

where $\gamma$ plays the role of the Value at Risk and $u$ the scenario
losses beyond it.

## Implementation

The shared assembly of these programs — stacking constraint blocks, mapping
user constraints to cones, and invoking Clarabel — lives in
`cvx.core.ConeProgramBuilder`. Each model contributes only its
model-specific blocks (the SOC above for the covariance models, the
scenario constraints for CVaR); identity-like blocks are built directly in
sparse form so large universes and scenario sets stay memory-efficient.
