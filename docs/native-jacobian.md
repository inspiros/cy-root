# Native finite-difference Jacobians

`cyroot.native.JacobianPlan` supplies numerical Jacobians to native square vector
Newton. It evaluates the existing residual callback at perturbed points using
reusable native scratch storage. With a compiled residual, stencil generation,
sampling, differentiation and the Newton loop run without acquiring the GIL.
Ordinary Python residuals use the existing Python bridge.

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

`VectorNewton(function, None, x0)` creates a default central plan. An analytic
Jacobian callback continues to select the analytic path. A plan is immutable
configuration: it holds dimension, scheme and optional copied steps. It does
not hold a callback or mutable sampling workspace. Independent solvers can
share a plan, including concurrent solves with separate callback adapters and
contexts. This preview exposes plans through Newton, without a standalone
Python derivative evaluator.

## Schemes and steps

`JacobianPlan(dimension, scheme='central', step=None,
workspace_bytes=64*1024*1024)` accepts the scheme strings `'forward'`,
`'backward'` and `'central'`. Dimension must be a positive C `int` and match
the square residual/solver. Integer scheme codes are not accepted.

Explicit `step` is a finite nonzero signed scalar or a length-dimension vector
of finite nonzero signed values. A scalar is copied to every axis. Step vectors
accept sized indexable sequences or native double buffers, including read-only,
negative-stride, zero-stride and unaligned inputs. The plan copies their logical
values; later input mutation cannot change it. Negative steps reverse sampling
direction. `step` is an absolute displacement, without additional scaling.

With `step=None`, the requested displacement at each new point is

\[
h_j = \eta\max(1, |x_j|),\qquad
\eta = \begin{cases}
2^{-26}, & \text{forward/backward},\\
\sqrt[3]{\epsilon}, & \text{central},
\end{cases}
\]

where \(\epsilon\) is binary64 machine epsilon. These are initial scale-aware
choices, not an adaptive accuracy guarantee. Poor scaling, nonsmooth functions,
noise or cancellation can require explicit steps or an analytic Jacobian.

Forward and backward schemes use one residual sample per axis plus the cached
base residual, with first-order truncation error for smooth functions. Central
uses two samples per axis plus the cached base residual, with second-order
truncation error. The engine divides by actual representable displacements.
For central differences, rounded offsets \(a=x_j^+-x_j\) and
\(b=x_j^--x_j\) can have unequal magnitudes. It uses weighted secants

\[
\frac{b}{b-a}\frac{F(x^+)-F(x)}{a}
+\frac{a}{a-b}\frac{F(x^-)-F(x)}{b}.
\]

Weights are computed from scaled offsets to avoid overflowing their difference.
This preserves quadratic consistency when rounding makes the stencil asymmetric.

## Counts, caches and failures

`f_calls` remains `(residual_calls, jacobian_attempts)`. Residual calls include
base evaluations, Newton trial evaluations and every stencil sample attempt,
including callbacks that fail. A finite-difference Jacobian adds one Jacobian
attempt. Sampling never adds solver iterations. Each uncached forward/backward
Jacobian costs `n` residual samples; central costs `2*n`. The current base
residual is reused, so it is not evaluated again for each plan.

`f_x0` can supply the initial base residual. `jac_x0` skips the initial Jacobian
approximation. At an initial root, or after convergence/stagnation/budget at a
new point, unused Jacobian sampling is skipped. The ordinary Newton result and
history rules apply: an unevaluated Jacobian contains NaNs.

All axes are validated before sampling. A requested step that rounds back to
the original coordinate produces unsuccessful `breakdown` with no sample
calls. Nonfinite points, displacements, samples or derivative arithmetic produce
`nonfinite`. There is no `nextafter` repair, automatic one-sided fallback,
step-size retry or regularization. Overflow of a subtraction or division is a
numerical failure even if a differently arranged formula could remain finite.

A failed sample retains the accepted Newton point/residual and clears the
entire Jacobian, including columns already computed. Python exceptions retain
their original identity and callback traceback. Native nonzero codes raise
`NativeCallbackError` with the original code and `callback_stage='jacobian'`;
the solver exposes that stage too. This identifies where the residual was used.
Failed sample attempts are counted and adapter reservations remain held through
exception transfer. Native callbacks must obey the same no-throw, borrowed
input and context lifetime rules as [analytic vector callbacks](native-vector.md).

## Storage and limits

Plan properties are `dimension`, `scheme`, `step` (None or a copied tuple),
`storage_bytes` and `workspace_bytes`. Automatic plans store no step vector;
explicit plans store `8*n` step bytes. The constructor's `workspace_bytes`
argument limits this persistent step storage. The property of the same name
reports the separate scratch requirement, `5*n*8` bytes.

Each planned Newton solve adds five vectors for perturbations, two samples and
prevalidated stencil coordinates. Its workspace budget includes
`8*(10*n + 2*n*n) + sizeof(int)*n + plan.storage_bytes`, including plan storage
even when that plan is shared. History has its existing separate byte budget.
These limits cover cy-root storage; LAPACK/provider internals manage their own
memory. Dimension/products are checked before native narrowing or allocation.

`derivatives.pxd` exposes `JacobianPlan.compute` for trusted compiled consumers.
It accepts disjoint caller-owned buffers and returns `DiffStatus`, writes a
column-major Jacobian, updates an attempted residual-call counter, and preserves
native callback status separately. Its pointer/span/no-throw preconditions are
documented in the declaration. It allocates no storage and performs no Python
work. See the [compiled example](../examples/native_jacobian.md).

This square plan shares its numerical stencil with the rectangular plan below.
[Native Traub–Steffensen](native-steffensen.md) reuses its forward stencil with
signed residual components as steps, recomputed at each accepted point.
Standalone [native Hessian plans](native-hessian.md) provide second derivatives
separately. Higher orders, adaptive error estimates and parallel sample scheduling
remain later work. NumPy/SciPy remain package dependencies.

## Rectangular Jacobians

A residual with `n` inputs and `m` outputs has a Jacobian of shape `(m, n)`.
`RectangularJacobianPlan(n, m, scheme='central', step=None, *,
memory_bytes=64*1024*1024)` uses the same stencils and rounded-coordinate rules.
Its `evaluate(function, x, f_x=None)` method accepts an ordinary Python function;
no callback wrapper is required. A matching residual `RectangularCallback` also
works, including compiled functions. The explicit dimensions let native callers
allocate correctly sized buffers. A future solver can infer these dimensions
from a Python initial point and base residual during setup.

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
Forward/backward schemes take `n` samples; central takes `2*n`. An uncached
base adds one call; `f_x` skips it. Sample failures clear the whole Jacobian.
Python exceptions keep their identity; native nonzero status raises
`NativeCallbackError` with the estimate and `function` or `jacobian` stage.
The adapter stays reserved through conversion, sampling and exception transfer.

The read-only properties are `input_dimension`, `output_dimension`, `scheme`,
`step`, `storage_bytes`, `workspace_bytes` and `memory_bytes`. Scratch needs
`3*n+2*m` doubles. The total memory budget includes copied input/base, output,
scratch and persistent steps: `(4*n+3*m+n*m+step_count)*sizeof(double)`, where
`step_count` is zero for automatic steps and `n` otherwise. Python objects and
callback-owned allocations are excluded.

The compiled `compute` interface in `derivatives.pxd` accepts disjoint caller-owned
`x[n]`, base `[m]`, output `[m*n]` and scratch `[3*n+2*m]` buffers plus the
rectangular native function/context and counters. Output is column-major at
`row + column*m`. Calls perform no cy-root allocation or Python work; Python
callbacks explicitly enter their bridge. The caller reserves the callback and
owns pointer lifetimes. Existing square plans and solvers keep their interfaces.
`VectorNewton` and `generalized_newton` now accept this plan. Their `J=None`
setup infers dimensions and creates a matching central plan automatically;
ordinary Python callers need no explicit dimensions. Linear least-squares
stationarity alone will not count as root convergence.
