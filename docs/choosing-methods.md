# Choose A Method

The right method depends on what you know about the function and on how
expensive the residual is to evaluate. Higher derivative orders do not by
themselves make a solve faster. Compare iteration histories and callback counts
on your own problem.

## Scalar Equations

| What You Have | Try | Why |
| --- | --- | --- |
| A continuous function and a sign-changing bracket | `bisect` or `brent` | Keeps a sign-changing bracket; bisection halves it every step. |
| One estimate and a derivative | `newton` | Full Newton steps with an analytic or numerical first derivative. |
| A Newton step that overshoots | `damped_newton` | Armijo backtracking accepts a smaller improving step. |
| First and second derivatives | `halley` or `chebyshev` | Uses curvature, at the cost of a second derivative. |
| Several derivative orders | `householder` | Any order, through a native recurrence instead of symbolic formulas. |
| Two starting estimates | `secant` | Needs no derivative, but has no bracket to fall back on. |
| An interval enclosure callback | `hybisect` | Splits the interval and can return several roots with a search status. |

[Scalar Migration](scalar-native-migration.md) lists every bracketing and
interpolation method. Open methods can diverge or hit a zero denominator.
Bracketing methods are only as reliable as their assumptions, such as
continuity.

## Vector Systems

| Method | Square | Tall Or Wide | Main Information |
| --- | --- | --- | --- |
| Generalized Newton / damped Newton | Yes | Yes | Dense Jacobian or numerical plan. |
| Halley / Super-Halley / Chebyshev / tangent hyperbolas | Yes | Yes | Dense Jacobian and residual Hessian, supplied or numerical. |
| Newton–Krylov | GMRES | LSQR | Jacobian-vector products `Jv`; rectangular systems also need `J^T w`. |
| Broyden / Barnes / Klement | Yes | No | A Jacobian approximation updated each step, initialized explicitly or numerically. |
| Robinson / Traub–Steffensen | Yes | No | Extra residual evaluations that build a secant or difference matrix. |
| Wolfe–Bittner | Yes | Yes | A pool of points and an affine least-squares step. |
| Levenberg–Marquardt / BFGS / L-BFGS / dogleg | Yes | Yes | Residual objective with a dense Jacobian or numerical plan. |
| Vrahatis | Yes | Yes, subject to sign-polygon limits | A sign polygon, one vertex per residual sign pattern; a small polygon alone is not a root. |

A Jacobian has shape `(m, n)` for `m` residuals and `n` unknowns. The dense
Newton and higher-order methods take rank-aware least-squares steps on
rectangular systems. Square systems use a direct factorization, and a singular
square Jacobian is reported as a breakdown rather than silently replaced by a
pseudoinverse. Each method's page covers initialization, rank cutoffs and
failure reasons.

## Matrix Storage

A dense Jacobian needs `m*n` doubles, a residual Hessian `m*n*n`, and higher
mixed partials `m*n**order`. Native code removes Python overhead from the
numerical kernel; it does not reduce these storage or sampling costs.

[Newton–Krylov](native-newton-krylov.md) never forms a dense Jacobian. It is
fastest with supplied native products. Without them, rectangular systems
recompute every Jacobian column for each product, so that `Jv` and `J^T w`
come from the same matrix. Saving storage then does not mean saving time.

L-BFGS keeps a bounded number of correction pairs, but its least-squares
objective still forms a dense Jacobian, so it is not a matrix-free alternative
to Newton–Krylov.

## Optimization And Root Success

The optimization family minimizes \(\frac12\|F(x)\|_2^2\). It can stop at a
local minimum with a nonzero residual even when a root exists elsewhere, so
these solvers check the original residual before reporting success. A
stationary gradient, an accepted linear solve and a tiny step are reported
separately; none of them is a root.

See [Optimization](optimization.md) for objective derivatives and external
optimizers, and [Stopping And Results](stopping.md) for the shared tolerances.
CyRoot does not choose a method for you, and no method is guaranteed to
converge from an arbitrary starting point.
