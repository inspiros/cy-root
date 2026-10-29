# Native Square Broyden

`cyroot.native.BroydenGood` and `BroydenBad` are resumable solvers for square
systems that share one native state engine. Good Broyden starts from a Jacobian
matrix, a native `JacobianPlan`, or `None` for an automatic central plan. The
`BroydenBad` constructor instead takes an **inverse** Jacobian approximation and
does not invert what you pass; to start from a Jacobian matrix, plan or `None`,
use `BroydenBad.from_jacobian(...)`, which builds the initial inverse. The
public `cyroot.broyden` runs the same engine to completion. Its `J_x0` is a
**Jacobian** guess for both variants; for bad Broyden it goes through
`from_jacobian`.

```python
from cyroot.native import BroydenGood, BroydenBad

def residual(point):
    x, y = point
    return (x*x + y - 2., x + 2*y*y - 3.)

jacobian0 = ((2.8, 1.), (1., 2.8))
good = BroydenGood(residual, jacobian0, (1.4, .7), history=32)
first = good.step()
result = good.run()
assert result.success
assert max(abs(v-1.) for v in result.root) < 1e-9
assert good.history[-1].root == result.root

determinant = 2.8*2.8 - 1.
inverse0 = ((2.8/determinant, -1./determinant),
            (-1./determinant, 2.8/determinant))
bad = BroydenBad(residual, inverse0, (1.4, .7))
assert bad.run().success
```

## Public Entry Point

The public signature is:

```text
broyden(F, x0, F_x0=None, J_x0=None, *, algo=2,
        etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
        history=0, history_every=1, history_bytes=8*1024*1024,
        workspace_bytes=64*1024*1024)
```

Here `eps` is the binary64 machine epsilon. `algo` is the built-in integer `1`
(good) or `2` (bad), or the string `'good'` or `'bad'`. Booleans, subclasses and
other objects that merely compare equal or convert to integers are rejected.
The default is bad Broyden.

```python
from cyroot import broyden
from cyroot.native import JacobianPlan

result = broyden(residual, (1.4, .7), J_x0=jacobian0, algo='bad')
assert result.success
assert result.approximation_kind == 'inverse_jacobian'

planned_result = broyden(residual, (1.4, .7),
                         J_x0=JacobianPlan(2, step=1e-4), algo='good')
assert planned_result.success
```

`J_x0` is a matrix, a native `JacobianPlan` of matching size, or `None` for an
automatic central plan. Here it is never an inverse; to start from an inverse
guess, use `BroydenBad` directly. A callable Jacobian is not accepted.

Compared with 1.x:

- `algo`, the tolerances and the iteration and resource options are
  keyword-only after `J_x0`. The `h` option is gone; use
  `JacobianPlan(..., step=...)`.
- Python residuals receive immutable tuples instead of NumPy arrays. Replace
  expressions such as `point - 1.` with indexed arithmetic, or convert inside
  the callback (in Python, under the GIL).
- Points, residuals and approximation matrices in the result are copied tuples.
  The result is a `BroydenResult` with the fields described below, so code that
  unpacked the old tuple or used ndarray methods needs updating. `f_calls` is an
  integer.
- Only square, nonempty systems are accepted, even when a cached initial
  residual is already zero. Shape errors raise; a singular solve reports
  `breakdown`, with no pseudoinverse or least-squares fallback.
- Native plans build the initial Jacobian. A central plan costs `2*n` residual
  samples plus one evaluation at the starting point unless it is cached;
  forward and backward plans cost `n` plus that evaluation. `F_x0` supplies it
  at no cost. The samples are skipped when the starting point is a root and are
  never repeated after setup.

`cyroot.vector_quasi_newton.broyden` and `find_vector_root('broyden', ...)` are
the same function. It returns only the final result; although it accepts the
history options, use a `BroydenGood` or `BroydenBad.from_jacobian` solver with
`step()`, `run()` and `.history` to inspect intermediate records.
[Klement](native-klement.md) shares this state engine with its own
row-weighted update, and [Barnes](native-barnes.md) shares the direct solve and
initialization with a projected update over retained secants.

## Distinct Updates

Write \(s=x_{k+1}-x_k\), using the actual rounded accepted points, and
\(y=F(x_{k+1})-F(x_k)\). Direct good Broyden solves \(B_k d=-F(x_k)\),
then updates

\[
B_{k+1}=B_k+\frac{(y-B_k s)s^T}{s^Ts},\qquad B_{k+1}s=y.
\]

Inverse bad Broyden takes \(d=-H_kF(x_k)\), then updates

\[
H_{k+1}=H_k+\frac{(s-H_k y)y^T}{y^Ty},\qquad H_{k+1}y=s.
\]

These are different minimum-change secant constraints. Neither matrix is an
analytic derivative at the current point. The direct variant uses packed
pivoted LU and checks factor finiteness even when LAPACK reports success.
The inverse variant uses scaled native matrix-vector contraction.

The rank-one update scales binary exponents and uses compensated sums across
the denominator, residual and final matrix addition. It avoids some premature
product, denominator and correction overflow/underflow. It does not increase
precision or certify a small secant error; cancellation, conditioning and final
binary64 range still limit accuracy. In one dimension the update uses the
secant quotient directly, avoiding subtraction of the previous approximation. There is no damping, line search, restart,
pseudoinverse or rectangular fallback. Local convergence requires suitable
smoothness, initial approximations and a nonsingular root Jacobian.

## Interface And Boundaries

`BroydenGood(function, jacobian0, x0, f_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`.
`BroydenBad` has the same options, replacing `jacobian0` with `inverse0`.
`broyden_good(function, jacobian0, x0, **options)` and
`broyden_bad(function, inverse0, x0, **options)` construct and run one solver.
The default `max_iter` is 200; zero or `None` is unlimited. The total budget is shared by
`step()` and `run()`. Iterating over a solver yields its remaining steps.

Points and residuals must have the same nonzero length \(n\), and initial
matrices the shape \((n,n)\). Sized sequences and buffers, including strided
NumPy arrays, are copied. Supplied matrix shapes and plan dimensions are
checked even when the cached starting point is already a root. An unused
nonfinite explicit matrix does not prevent residual convergence, but is
reported as invalid. A callable initial Jacobian is not accepted; pass a matrix
or a plan.

Python residual callbacks receive an immutable tuple and return a numeric
sequence or a supported buffer. A residual `VectorCallback` created in Python,
by the Cython factory or from a C capsule can run without the GIL, and its
context is kept alive. All point, matrix, cache and option conversions happen
while explicit adapters are reserved. Setup and result and history export hold
the GIL.

## Native Initial Jacobian And Inverse Setup

```python
from cyroot.native import JacobianPlan

planned = BroydenGood(residual, JacobianPlan(2, step=.001), (1.4, .7))
assert planned.result.initialization_calls == 1
assert planned.result.approximation_valid
assert planned.run().success

inverse_planned = BroydenBad.from_jacobian(residual, None, (1.4, .7))
assert inverse_planned.run().success
```

`None` selects the default central plan; an explicit `JacobianPlan` sets the
scheme and steps. The initial residual, including one passed as `f_x0`, is
reused as the plan's base point. `initialization_calls` counts attempts to build
the initial Jacobian numerically (zero or one, failures included), and
`f_calls` includes every residual sample. An explicit matrix counts no
numerical attempt, even when bad Broyden has to invert it. Plans run only
during setup; later steps use secant updates.

`from_jacobian` factors a private copy of the Jacobian once and solves for each
unit vector in native scratch. Every status, factor and solution is checked,
and no Python object or memoryview is created per column. Only a complete,
finite inverse is published. If building or inverting fails, the initial point
and residual are kept, the approximation is all NaN and `approximation_valid`
is false; `initialization_calls` and `phase` show what was attempted. A failed
native residual sample reports the callback stage `'initial_jacobian'`, and a
Python exception propagates as the original instance.

If the starting point is already a root, no plan is sampled and nothing is
inverted, so a generated matrix stays all NaN and invalid, with zero
initialization attempts. A supplied finite approximation stays valid; an unused
nonfinite one is marked invalid. Valid means that a complete, finite
approximation is available, not that it is accurate or nonsingular. The
initial inversion has no pseudoinverse fallback.

## Results, History And Failures

`BroydenResult` is an immutable tuple with named fields: `root`, `f_root`,
`approximation`, `approximation_kind`, `approximation_iteration`, `iters`,
`f_calls`, `precision`, `error`, `termination_reason`, `callback_code`,
`callback_stage`, `linear_status`, `phase`, `approximation_valid`, and
`initialization_calls`. `BroydenStep` is the same type. `success`,
`converged` and `optimal` are true only for residual convergence. Results taken
while running have `termination_reason=None`. The integer `f_calls` counts
residual attempts; a cached initial residual counts as zero.

`approximation_kind` is `'jacobian'` or `'inverse_jacobian'`.
`approximation_iteration` starts at zero and gives the iteration of the last
completed secant update. After a finite trial, the solver accepts the point and
residual and checks the residual, step and budget **before** any update that
would go unused. A final result's approximation can therefore be older than the
returned point; the metadata says so, rather than presenting it as the
derivative there.

`phase` is `None`, `'linear'`, `'trial'`, `'update'`, `'initial_jacobian'` or
`'initial_inverse'`. `linear_status` keeps the latest LAPACK status and is reset
when a new linear phase starts. A singular direct solve reports `breakdown`;
nonfinite factors, solutions, residuals, differences or updates report
`nonfinite`. A zero denominator in a needed secant update reports `breakdown`.
A trial equal to the current point with an unsatisfied residual reports
unsuccessful `stagnation` first.

Each attempt to evaluate the residual at a finite trial point increments
`iters`, even if the callback fails or returns nonfinite values; arithmetic or
linear failures before the trial do not. A failed trial keeps the previously
accepted point, residual, matrix and metrics. A failed update keeps the newly
accepted point, residual and metrics with the previous **complete** matrix and
`approximation_iteration`; a partially updated matrix is never visible.
Callback errors keep the native status or the original Python exception, and
a `NativeCallbackError` carries a copied result. Once an error has been raised,
further calls on the finished solver only return its state.

`history` returns copied snapshots. `history_every` samples iterations, and the
terminal snapshot is included if there is room. A failure later in the same
iteration replaces that iteration's earlier record. Once the capacity is full,
later snapshots are dropped and `history_truncated` is true. Records already
copied are unaffected by later steps or by destroying the solver.

## Memory And Compiled Use

Good Broyden reserves \(6n+3n^2\) doubles and \(n\) C-int pivots; bad
Broyden with an explicit inverse reserves \(6n+2n^2\) doubles and no pivots.
`from_jacobian` needs another \(n^2\) doubles and \(n\) pivots for the initial
inversion. A numerical initial plan adds \(5n\) scratch doubles plus storage
for its explicit steps. History stores \(2n+n^2\) doubles and a metadata record
per slot. Products, capacities and byte limits are all checked before
allocation. These budgets cover CyRoot's native storage; Python exports and
memory used inside LAPACK come on top. The engine allocates no native storage
per step.

The [compiled example](../examples/native_broyden.md) uses the installed `.pxd`
declarations and headers with native callbacks, without NumPy C headers.
`broyden_ops.pxd` exposes the low-level update helper from
`broyden.hpp`. It takes non-overlapping packed column-major matrix and output
spans plus direction and target vectors, and checks the dimensions before
writing. A nonfinite or zero direction fills the whole output with NaN; invalid
dimensions leave the spans untouched. The numerical kernel follows the
[native callback and storage contracts](native-core.md).
