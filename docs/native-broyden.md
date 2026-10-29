# Native Square Broyden

`cyroot.native.BroydenGood` and `BroydenBad` are resumable square-system solvers
with a shared native state engine. Good Broyden takes a Jacobian matrix, native
`JacobianPlan`, or `None` for an automatic central plan. The ordinary bad Broyden
constructor takes an **inverse**
Jacobian approximation. Its explicitly named `BroydenBad.from_jacobian(...)`
factory accepts a Jacobian matrix/plan/`None` and builds the initial inverse.
The ordinary bad constructor does not invert the matrix you provide.
The established `cyroot.broyden` entry point now uses this engine as a one-shot
solve. Its `J_x0` argument remains a **Jacobian** guess for both variants; bad
Broyden uses the from-Jacobian initialization path.

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

## Established Entry Point Migration

The public signature is:

```text
broyden(F, x0, F_x0=None, J_x0=None, *, algo=2,
        etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
        history=0, history_every=1, history_bytes=8*1024*1024,
        workspace_bytes=64*1024*1024)
```

Here `eps` denotes binary64 machine epsilon. `algo` accepts exact built-in
integers `1` (good) and `2` (bad), or exact built-in strings `'good'` and `'bad'`.
Booleans, subclasses and objects implementing equality or integer conversion
protocols are rejected. Selection and all input conversions run under the
native callback reservations. The default remains bad Broyden.

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

`J_x0` accepts a matrix, an initialized matching native `JacobianPlan`, or
`None` for an automatic central plan. It never means an inverse in this public
entry point. For an existing inverse guess, use `BroydenBad` directly. There is
no callable analytic Jacobian role.

This migration changes the following behavior:

- `algo`, tolerances, iteration and resource options are keyword-only after
  `J_x0`. The `h` option is removed; configure `JacobianPlan(..., step=...)`.
- Python residuals receive immutable tuples rather than NumPy arrays. Replace
  expressions such as `point - 1.` with indexed sequence arithmetic, or explicitly
  convert inside the callback. That conversion uses Python and the GIL.
- Points, residuals and approximation matrices in the result are copied tuples.
  The return type is `BroydenResult`, with the fields described below; old tuple
  unpacking and ndarray methods need updating. `f_calls` is an integer.
- Only square nonempty systems are accepted, including when a cached initial
  residual is already zero. Shape errors raise; required singular solves report
  `breakdown` without a pseudoinverse or least-squares fallback.
- Native initial plans replace the old derivative sampler. Central initialization
  costs `2*n` residual samples plus one uncached base evaluation; forward/backward
  costs `n` plus the base. `F_x0` reuses the base and counts zero calls. These
  samples are skipped at an initial root and are never repeated after setup.

The same migration applies to `cyroot.vector_quasi_newton.broyden` and
`find_vector_root('broyden', ...)`. The function returns only the final result.
Although it accepts history allocation options, inspect intermediate records
through a retained `BroydenGood` or `BroydenBad.from_jacobian` state using
`step()`, `run()` and `.history`. [Klement](native-klement.md) now shares this
state engine with its own row-weighted update and established API migration.
[Barnes](native-barnes.md) shares its direct solve and initialization with a
deterministic retained-secant projection update.
The remaining quasi-Newton methods retain their existing interfaces.

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

Points and residuals must have matching nonzero length \(n\), and initial
matrices shape \((n,n)\). Sized sequences and native double buffers, including
strided NumPy arrays, are copied. Supplied matrix shapes and explicit
plan dimensions/initialization are checked even at a cached initial root. An
unused nonfinite explicit matrix does not prevent residual convergence; its
validity flag is false.
There is no initial analytic Jacobian callback role; pass a matrix or plan.

Python residual callbacks receive immutable tuple snapshots and return numeric
sequences or supported double buffers. A residual-kind `VectorCallback`, Cython
factory or C capsule descriptor can run without the GIL; contexts stay pinned.
All point, matrix, cache and option conversions occur while explicit adapters
are reserved. Setup and Python result/history exports use the GIL. NumPy and
SciPy remain package dependencies.

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

`None` selects the default central plan; an explicit `JacobianPlan` configures
scheme and steps. The initial residual is reused as the plan's base, including
when provided through `f_x0`. `initialization_calls` counts numerical initial
Jacobian attempts (zero or one, including failure); `f_calls` includes all
residual samples. Explicit matrix initialization counts no numerical derivative attempts, even
when bad Broyden must invert that Jacobian.
These plans run only during setup; subsequent steps use secant updates.

The inverse factory factors a private Jacobian copy once and solves unit right-
hand sides with native scratch. Each status, factor and solution is checked;
no Python operation or memoryview is created per column. Only a complete finite
matrix is published. Failed generation/inversion retains the initial point and
residual, with all-NaN approximation and `approximation_valid=False`.
`initialization_calls` and `phase` identify the attempted work. Native residual
sample failures report callback stage `'initial_jacobian'`; Python exceptions
preserve their original identity.

An initial root skips unused plan sampling and inversion. A generated matrix
therefore stays all-NaN/invalid, with zero initialization attempts. A supplied
finite explicit approximation stays valid; an unused nonfinite one is marked
invalid. Validity indicates a completely available finite approximation, not
accuracy or nonsingularity. The initial inversion has no pseudoinverse fallback.

## Results, History And Failures

`BroydenResult` is an immutable tuple with named fields: `root`, `f_root`,
`approximation`, `approximation_kind`, `approximation_iteration`, `iters`,
`f_calls`, `precision`, `error`, `termination_reason`, `callback_code`,
`callback_stage`, `linear_status`, `phase`, `approximation_valid`, and
`initialization_calls`. The last two fields append to the earlier tuple;
`BroydenStep` is the same type. Compiled consumers must rebuild current `.pxd`
declarations because the native object/history record layout changes.
`success`, `converged` and `optimal` are true only for residual convergence.
Running results have `termination_reason=None`. Residual attempts are counted
by the integer `f_calls`; cached initial residuals count zero.

`approximation_kind` is `'jacobian'` or `'inverse_jacobian'`.
`approximation_iteration` starts at zero and tracks the last fully committed
secant update. After a finite trial, the solver commits its point and residual,
then checks residual, step and budget stopping **before** any unused update.
Consequently a terminal result's approximation can predate the returned point.
The metadata makes this explicit instead of calling it an exact derivative.

`phase` is `None`, `'linear'`, `'trial'`, `'update'`, `'initial_jacobian'` or
`'initial_inverse'`. The exact latest LAPACK
`linear_status` is retained and resets on entering a new linear phase.
A singular required direct solve reports `breakdown`; nonfinite factors,
solutions, residuals, differences or updates report `nonfinite`. A zero secant
update denominator reports `breakdown` when an update is needed. An unchanged
trial with unsatisfied residual reports unsuccessful `stagnation` first.

A finite trial residual attempt increments `iters`, including callback or
nonfinite-output failures. Pretrial arithmetic/linear failures do not. Failed
trials preserve the previous accepted point/residual/matrix and metrics. An
update failure preserves the newly accepted point/residual/metrics and the
previous **complete** matrix and `approximation_iteration`; partially updated
matrices never become observable. Callback errors preserve native integer
status or original Python `BaseException` identity. A `NativeCallbackError`
contains a copied result. After error delivery, terminal accesses are passive.

`history` returns copied snapshots. `history_every` samples iteration
indices; a terminal snapshot is included when capacity permits. A state that
fails later in the same iteration replaces that iteration's earlier record.
Once capacity is full, later snapshots are dropped and `history_truncated` is
true. Existing copied records remain independent of future steps/destruction.

## Memory And Compiled Use

Good Broyden reserves \(6n+3n^2\) doubles and \(n\) C-int pivots; bad
Broyden with an explicit inverse reserves \(6n+2n^2\) doubles and no pivots.
`from_jacobian` uses an additional \(n^2\)-double LU segment and \(n\) pivots
during initial inversion. Any numerical initial plan adds \(5n\) scratch
doubles plus its persistent explicit step-vector storage. Existing explicit
matrix constructor workspace sizes remain unchanged. History stores
\(2n+n^2\) doubles plus a native metadata record per slot. Checked products,
native capacities and byte limits precede allocation. These budgets cover
cy-root's native storage; Python exports and provider-internal memory are
additional. The cy-root engine allocates no native storage per step.

The [compiled example](../examples/native_broyden.md) uses installed `.pxd`
declarations, headers and pinned native callbacks without NumPy C headers.
`broyden_ops.pxd` exposes the trusted raw update helper from `broyden.hpp`.
It accepts disjoint packed column-major matrix and output spans, direction and
target vectors, with checked-dimension status before writes. Nonfinite or zero-
direction failures poison all output; invalid dimensions leave spans untouched.
The numerical kernel uses the [native callback and storage contracts](native-core.md).
