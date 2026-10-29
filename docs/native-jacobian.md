# Native Finite-Difference Jacobians

`cyroot.native.JacobianPlan` supplies numerical Jacobians to the square vector
solvers. It evaluates the residual callback at perturbed points, using
reusable native scratch storage. With a compiled residual, stencil generation,
sampling, differencing and the Newton loop all run without the GIL; Python
residuals go through the usual Python bridge.

```python
from cyroot.native import JacobianPlan, VectorNewton

def residual(x):
    return (x[0]**2 + x[1] - 3., 2*x[0] + x[1]**2 - 6.)

plan = JacobianPlan(2, scheme='central')
solver = VectorNewton(residual, plan, (1.4, 1.7), history=16)
initial_jacobian = solver.result.df_root
result = solver.run()
assert result.success
assert abs(result.root[0] - 1.) < 1e-8
assert abs(result.root[1] - 2.) < 1e-8
```

`VectorNewton(function, None, x0)` creates a default central plan, and an
analytic Jacobian callback takes the analytic path instead. A plan is immutable
configuration: the dimension, the scheme and, optionally, copied steps. It holds
no callback or mutable sampling workspace, so independent solvers can share it,
also concurrently with separate callback adapters and contexts. The square plan
only serves solvers; the rectangular plan below can also be evaluated directly
from Python.

## Schemes And Steps

`JacobianPlan(dimension, scheme='central', step=None,
workspace_bytes=64*1024*1024)` accepts the scheme strings `'forward'`,
`'backward'` and `'central'`; integer scheme codes are not accepted. The
dimension must be a positive C `int` matching the square residual and solver.

An explicit `step` is a finite, nonzero number, used for every axis, or a
vector of such numbers, one per axis. Step vectors can be sized sequences or
buffers, including read-only, negative-stride, zero-stride and unaligned ones;
the plan copies their values, so later changes to the input do not affect it.
Steps are absolute displacements, not scaled further, and a negative step
reverses the sampling direction.

With `step=None`, the requested displacement at each new point is

\[
h_j = \eta\max(1, |x_j|),\qquad
\eta = \begin{cases}
2^{-26}, & \text{forward/backward},\\
\sqrt[3]{\epsilon}, & \text{central},
\end{cases}
\]

where \(\epsilon\) is the binary64 machine epsilon. These steps scale with the
point but do not adapt to the function; poor scaling, nonsmooth functions,
noise or cancellation may call for explicit steps or an analytic Jacobian.

Forward and backward schemes take one residual sample per axis plus the cached
base residual, with first-order truncation error for smooth functions. The
central scheme takes two samples per axis plus the base, with second-order
truncation error. The engine divides by the actual representable
displacements. For central differences, the rounded offsets \(a=x_j^+-x_j\)
and \(b=x_j^--x_j\) can differ in magnitude, so it uses weighted secants

\[
\frac{b}{b-a}\frac{F(x^+)-F(x)}{a}
+\frac{a}{a-b}\frac{F(x^-)-F(x)}{b}.
\]

The weights are computed from scaled offsets so that their difference cannot
overflow. The formula stays exact for quadratics when rounding makes the
stencil asymmetric.

## Counts, Caches And Failures

`f_calls` is `(residual_calls, jacobian_attempts)`. Residual calls include
base and trial evaluations and every stencil sample, failed ones too; each
numerical Jacobian adds one Jacobian attempt, and sampling adds no iterations.
A forward or backward Jacobian costs `n` residual samples and a central one
`2*n`, reusing the current residual as the base.

`f_x0` can supply the initial base residual, and `jac_x0` skips the initial
approximation. Nothing is sampled when the starting point is a root, or when a
new point ends the solve. As for any Newton result or history, a Jacobian that
was not evaluated contains NaNs.

All axes are validated before sampling. A step that rounds back to the original
coordinate gives an unsuccessful `breakdown` with no sample calls, and
nonfinite points, displacements, samples or arithmetic give `nonfinite`. Steps
are never nudged, swapped for a one-sided scheme, retried or regularized.
Overflow in a subtraction or division is a numerical failure, even if another
arrangement might stay finite.

A failed sample keeps the accepted point and residual and clears the whole
Jacobian, including columns already computed. Python exceptions propagate as
the original instance with the callback's traceback. Nonzero native codes
raise `NativeCallbackError` with the original code and
`callback_stage='jacobian'`, which the solver also exposes, showing that the
residual failed while building the Jacobian. Failed samples are counted, and
adapters stay reserved through exception transfer. Native callbacks follow the
same no-throw, borrowed-input and context-lifetime rules as
[analytic vector callbacks](native-vector.md).

## Storage And Limits

The plan's properties are `dimension`, `scheme`, `step` (`None` or a copied
tuple), `storage_bytes` and `workspace_bytes`. Automatic plans store no steps,
and explicit plans store `8*n` bytes of them; the constructor's
`workspace_bytes` argument limits that storage, while the property of the same
name reports the separate scratch requirement of `5*n*8` bytes.

A Newton solve with a plan adds five vectors, for the perturbations, two
samples and the prevalidated stencil coordinates. Its workspace budget is
`8*(10*n + 2*n*n) + sizeof(int)*n + plan.storage_bytes`, counting the plan's
storage even when the plan is shared, and history has its own byte budget.
These limits cover CyRoot's storage; LAPACK manages its own memory. Dimensions
and their products are checked before any narrowing conversion or allocation.

`derivatives.pxd` exposes the low-level `JacobianPlan.compute` for compiled
code. It takes non-overlapping caller-owned buffers, returns a `DiffStatus`,
writes a column-major Jacobian, updates a residual-call counter and reports the
native callback status separately. Its pointer, span and no-throw requirements
are documented in the declaration. It allocates nothing and does no Python
work. See the [compiled example](../examples/native_jacobian.md).

This square plan shares its stencil with the rectangular plan below.
[Traub–Steffensen](native-steffensen.md) reuses the forward stencil, with the
residual components as steps. [Hessian plans](native-hessian.md) provide
second derivatives, and [higher mixed partials](native-higher-mixed.md) and
[adaptive error estimates](numerical-derivatives.md) are also available.
Samples within one evaluation are taken one after another.

## Rectangular Jacobians

A residual with `n` inputs and `m` outputs has a Jacobian of shape `(m, n)`.
`RectangularJacobianPlan(n, m, scheme='central', step=None, *,
memory_bytes=64*1024*1024)` uses the same stencils and rounding rules. Its
`evaluate(function, x, f_x=None)` method accepts a plain Python function, or a
matching residual `RectangularCallback`, compiled or not. The explicit
dimensions let native callers allocate buffers of the right size. Without an
explicit plan, Vector Newton infers the dimensions from the starting point and
residual.

```python
from cyroot.native import RectangularJacobianPlan

def rectangular_residual(x):
    return (x[0]**2, x[1]**2, x[0]+x[1])

plan = RectangularJacobianPlan(2, 3, step=1e-3)
estimate = plan.evaluate(rectangular_residual, (1., 2.))
assert estimate.status == 'ok'
assert len(estimate.jacobian) == 3
assert all(len(row) == 2 for row in estimate.jacobian)
```

`JacobianEstimate` contains copied `jacobian` rows, integer `f_calls`, `status`
(`ok`, `nonfinite`, `breakdown` or `callback_failure`) and `callback_code`.
Forward and backward schemes take `n` samples and the central scheme `2*n`,
plus one call for the base unless `f_x` is given. A failed sample clears the
whole Jacobian. Python exceptions propagate as the original instance; a
nonzero native status raises `NativeCallbackError` with the estimate and the
stage `function` or `jacobian`. The adapter stays reserved through conversion,
sampling and exception transfer.

The read-only properties are `input_dimension`, `output_dimension`, `scheme`,
`step`, `storage_bytes`, `workspace_bytes` and `memory_bytes`. The scratch
takes `3*n+2*m` doubles. The memory budget covers the copied input and base,
the output, the scratch and the stored steps:
`(4*n+3*m+n*m+step_count)*sizeof(double)`, where `step_count` is zero for
automatic steps and `n` otherwise. Python objects and allocations made by
callbacks are not counted.

The compiled `compute` interface in `derivatives.pxd` takes non-overlapping,
caller-owned `x[n]`, base `[m]`, output `[m*n]` and scratch `[3*n+2*m]`
buffers, the rectangular native function and context, and the counters. The
output is column-major, at `row + column*m`. A call allocates nothing and does
no Python work, except that Python callbacks enter their own bridge. The caller
reserves the callback and keeps the pointers valid. `VectorNewton` and
`generalized_newton` accept this plan, and with `J=None` they infer the
dimensions and create a central plan, so Python callers need no explicit
dimensions. Least-squares stationarity alone never counts as root
convergence.
