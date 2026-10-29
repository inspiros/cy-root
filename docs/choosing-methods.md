# Choose A Method

Method choice depends on the information available and the cost of evaluating
the residual. More derivative orders do not automatically make a solve faster.
Use iteration histories and callback counts to understand the work on your problem.

## Scalar Equations

| What You Have | A Starting Point | Why |
| --- | --- | --- |
| A continuous function and a sign-changing bracket | `bisect` or `brent` | Retains a scalar bracket; bisection has a simple contraction rule. |
| One estimate and a derivative | `newton` | A full Newton step with analytic or numerical first derivatives. |
| A Newton step that overshoots | `damped_newton` | Armijo backtracking accepts a smaller improving step. |
| First and second derivatives | `halley` or `chebyshev` | Uses curvature; derivative cost and conditioning still matter. |
| Several derivative orders | `householder` | A native runtime-order recurrence without symbolic formulas. |
| Two starting estimates | `secant` | Derivative-free open iteration; no bracket safeguard. |
| An interval enclosure callback | `hybisect` | Searches subintervals and can return multiple roots with search status. |

[Scalar Migration](scalar-native-migration.md) lists the full bracketing and
interpolation coverage. Open methods can diverge or encounter a zero denominator.
Bracketing relies on its mathematical assumptions, including continuity.

## Vector Systems

| Method | Square | Tall Or Wide | Main Information |
| --- | --- | --- | --- |
| Generalized Newton / damped Newton | Yes | Yes | Dense Jacobian or numerical plan. |
| Halley / Super-Halley / Chebyshev / tangent hyperbolas | Yes | Yes | Dense Jacobian and residual Hessian, supplied or approximated. |
| Newton–Krylov | GMRES | LSQR | `Jv`; rectangular supplied products also need `J.Tw`. |
| Broyden / Barnes / Klement | Yes | No | Evolving matrix initialized explicitly or numerically. |
| Robinson / Traub–Steffensen | Yes | No | Residual probes that construct a secant or difference matrix. |
| Wolfe–Bittner | Yes | Yes | A pool of points and an affine least-squares step. |
| Levenberg–Marquardt / BFGS / L-BFGS / dogleg | Yes | Yes | Residual objective with a dense Jacobian or numerical plan. |
| Vrahatis | Yes | Yes, subject to sign-polygon limits | An admissible vertex sign pattern; multidimensional width alone is not root success. |

A Jacobian has shape `(m,n)` for `m` residuals and `n` unknowns. The dense Newton
and higher-order methods use rank-aware least-squares corrections for rectangular
systems. Square Newton retains a direct factorization policy: a singular square
system is not silently replaced by a pseudoinverse. Consult each method's guide
for initialization, rank cutoffs, and failure semantics.

## Matrix Storage

A dense Jacobian needs `m*n` doubles. A residual Hessian needs `m*n*n`; higher
mixed partials grow as `m*n**order`. Native code removes Python operations from
the numerical kernel, not these storage or sampling costs.

[Newton–Krylov](native-newton-krylov.md) avoids a dense Jacobian. Supplied native
products are its efficient path. Its rectangular numerical fallback recomputes
all coordinate columns per product to keep the transpose consistent, so a
storage saving is not automatically a speed improvement.

L-BFGS bounds the history of optimizer correction pairs, but the present
least-squares objective model still forms a dense Jacobian. It is not a
matrix-free replacement for Newton–Krylov.

## Optimization And Root Success

The optimization family minimizes \(\frac12\|F(x)\|_2^2\). A positive local
minimum is possible even when another point is a root. These solvers check the
original residual before declaring success. Gradient stationarity, an accepted
linear solve, and a tiny step are separate diagnostics.

Use [Optimization](optimization.md) for objective derivatives and external
optimizers. Use [Stopping And Results](stopping.md) for the shared tolerances.
There is no automatic method selection or guarantee for arbitrary starting points.
