# Root Finding Through Optimization

For a real residual function `F: R^n -> R^m`, define

\[
\phi(x)=\tfrac12\sum_{i=1}^m F_i(x)^2.
\]

Every root of `F` is a global minimum of this nonnegative objective, with value
zero, and every point where the objective is zero is a root. A positive local
minimum or a small objective gradient is not a root. The construction works
for square, overdetermined and underdetermined residuals alike.

## Value, Gradient And Exact Hessian

```python
from scipy.optimize import minimize
from cyroot.optimization import LeastSquaresObjective, root_via_optimizer

def F(x):
    return (x[0]**2 + x[1] - 2., x[0]*x[1] - .5)

def J(x):
    return ((2*x[0], 1.), (x[1], x[0]))

def H(x):
    return (((2., 0.), (0., 0.)), ((0., 1.), (1., 0.)))

objective = LeastSquaresObjective(F, 2, jacobian=J, hessian=H)
result = root_via_optimizer(
    objective, minimize, (1., .4),
    optimizer_options={'method': 'BFGS', 'jac': objective.gradient,
                       'options': {'gtol': 1e-12}},
)
assert result.success
```

`LeastSquaresObjective(F, input_dimension, output_dimension=None, *,
jacobian=None, hessian=None, workspace_bytes=64*1024*1024)` is also available
from `cyroot.native`. Without `output_dimension`, the residual is square.
Python providers receive a copied tuple and return indexable real values of
shape `(m,)`, `(m,n)` and `(m,n,n)` respectively. Native `VectorCallback`,
`RectangularCallback`, `HessianCallback` and `RectangularHessianCallback`
providers of the right kind and dimensions are also accepted, and their context
owners are kept alive as long as the adapter.

The methods `value(x)` (also `objective(x)`), `residual(x)`, `gradient(x)` and
`hessian(x)` return a float or copied immutable tuples. They compute

\[
\nabla\phi=J^T F,\qquad
\nabla^2\phi=J^TJ+\sum_i F_i\nabla^2F_i.
\]

The Hessian includes the residual-Hessian term. A Hessian provider requires a
Jacobian provider, and a missing provider raises an error before any sampling.
There is no implicit Gauss–Newton approximation or finite-difference fallback;
for numerical derivatives, pass derivative callables that compute them. Every
call samples afresh: `value` and `residual` call `F` once, `gradient` calls `F`
and `J` once each, and `hessian` calls `F`, `J` and `H` once each. Nothing is
cached between optimizer calls, and no optimizer options are forwarded behind
your back.

The adapter checks dimensions and storage and allocates its native scratch
once; `memory_bytes` reports that allocation, not counting Python boundary
objects or allocations made by providers. Exponent-scaled compensated
accumulation avoids needless intermediate overflow and underflow, but gives no
extra accuracy under ill-conditioned cancellation. Unrepresentable results or
nonfinite inputs raise `FloatingPointError`, and shape, kind and provider errors
raise exceptions too. Python callback exceptions propagate as they are; nonzero
native statuses raise `NativeCallbackError` with the callback stage and code.
Partially written callback output is detected.

## External Optimizers And Root Status

`root_via_optimizer(objective, optimizer, x0, *, etol=1e-10, ertol=4*eps,
optimizer_options=None)` calls `optimizer(objective, copied_x0, **options)`.
The returned object must have an `x` attribute, which fits
`scipy.optimize.minimize`; wrap other optimizers in a small function. Pass
`jac=objective.gradient`, `hess=objective.hessian`, the method, bounds and
tolerances through the optimizer's own options. Derivatives you ask the
optimizer to use must be backed by providers installed on the objective.

The result contains copied `root`, `f_root`, infinity-norm `error`, root
`success`, and `termination_reason`, plus `optimizer_success`, `optimizer_status`
and the original `optimizer_result`. Original optimizer exceptions propagate.
Root success requires

\[
\|F(x)\|_\infty\le \text{etol}+\text{ertol}\,\|F(x_0)\|_\infty.
\]

The scale uses the starting point. If the starting point is already a root,
the optimizer is not called and the optimizer fields are `None`. Otherwise the
residual at the returned point is evaluated and checked independently. If the
optimizer reports success but the root tolerance is not met, the reason is
`optimizer_stopped_without_root`. Conversely, a returned point that meets the
root tolerance is a successful root even when the optimizer reports failure,
whose status stays visible. An invalid or nonfinite returned point raises the
objective's exception. A result without a root says nothing about other points
or whether a root exists at all.

For example, `F(x,y)=(x^3-2*x+2,y)` has a strict positive local minimum of `phi`
at `(sqrt(2/3),0)`, although a root exists at negative `x`. Convergence results
for convex optimization do not apply, since squared nonlinear residuals are
generally not convex. A stationary point is a root if the Jacobian there has
full **row** rank; full column rank in an overdetermined system is not enough.

External Python optimizers run in Python under the GIL. Compiled code can call
the objective's numerical kernel without the GIL when the callbacks are native,
but that does not make an external optimizer's loop GIL-free. An objective and
its native adapters are reserved during each operation, including input
conversion, error transfer and result export; use separate objectives and
adapters for concurrent evaluations.

## Higher-Order Objective Derivatives

The product rule gives, for any positive order `r`,

\[
D^r\phi(x)[v_1,\ldots,v_r]=\frac12
\sum_{S\subseteq\{1,\ldots,r\}}
\left\langle D^{|S|}F(x)[v_S],
D^{r-|S|}F(x)[v_{S^c}]\right\rangle,
\]

where `D^0 F=F`. Directions in a subset keep their original order. Complementary
subsets have equal dot products, so the implementation uses one representative
per pair. This includes all mixed terms; applying the scalar formula separately
to residual coordinates would omit necessary contractions.

```python
from cyroot.optimization import ObjectiveDerivativePlan

plan = ObjectiveDerivativePlan(1, 1, order=3)
third = plan.evaluate(
    lambda x: (x[0]**3,),
    [lambda x: ((3*x[0]**2,),),
     lambda x: (((6*x[0],),),),
     lambda x: ((((6.,),),),)],
    (2.,),
)
assert third == (((480.,),),)  # third derivative of x**6/2
```

`ObjectiveDerivativePlan(n,m,r, *, max_terms=1_000_000,
memory_bytes=64*1024*1024)` is immutable. `evaluate(F, derivatives, x)` requires
exactly `r` Python callables, for `DF` through `D^rF`. It calls each one once
and packs its output before calling the next, so providers may reuse their own
scratch arrays. These calls run in Python, under the GIL.

Alternatively `transform([F_x, DF_x, ..., D^rF_x])` accepts already evaluated
tensors. Order-`k` residual derivatives have shape `(m,)+(n,)*k`; the output has
shape `(n,)*r`. No symmetry is imposed on the supplied tensors. Values are copied
and finite-checked; bad shapes/nonfinite final entries raise `ValueError`, complex
values raise `TypeError`. No numerical differentiation is performed.

The dense transform uses `m*n**r*2**(r-1)` products, plus subset bookkeeping.
`max_terms` bounds this work before allocation. The byte limit covers retained
native metadata, packed tensors and native output/scratch, excluding Python
objects and provider allocations. Positive order must fit the native subset bit
capacity. These limits make large requests explicit; native code does not remove
the exponential cost of high derivative orders.

## Compiled Interface

`objective.pxd` exposes the accumulator, value/gradient/Hessian kernels and
`LeastSquaresObjective.evaluate_raw`. A caller pins the state, enters its
reservation while holding the GIL, calls the raw method inside `nogil`, then
leaves with the GIL. Raw orders are `-1` residual, `0` value, `1` gradient and
`2` exact Hessian. Output spans must be disjoint from the state and input;
capacities are checked before sampling. Failure leaves caller output unchanged.
Statuses: `0` success, `1` nonfinite, `2` invalid/missing provider, `3` callback
failure. Inspect/transfer the original callback error before releasing its guard.
See the [compiled example](../examples/native_objective.pyx).

`objective_derivatives.pxd` exposes `ObjectiveDerivativePlan.compute` and the
allocation-free transform in `objective_derivatives.hpp`. Callers own packed
`D^kF` spans, capacities, output and scratch. Derivative tensors are residual-first:
`D[k][o + m*(j0 + n*(j1 + ...))]`; the objective output has its first input index
fastest. State is immutable, so native callers can share a plan with disjoint
caller-owned buffers. Status `1` denotes nonfinite input/final result; `2` denotes
invalid counts/capacities/work bounds. Output is unchanged on failure. This entry
point neither allocates nor calls Python and needs no new derivative-callback ABI.

## Native Optimizer Families

Each of the four optimization methods has a scalar and a vector function,
taking `(function, derivative_or_jacobian, x0)` positionally; pass `None` for a
native numerical derivative. Vector dimensions are keyword-only. The input
length can be inferred, but a rectangular Python residual needs
`output_dimension`, unless you pass a native residual adapter that declares its
shape.

```python
from cyroot import (scalar_levenberg_marquardt, scalar_bfgs, scalar_lbfgs,
                    scalar_powell_dogleg, vector_lbfgs)
from cyroot.optimization import OptimizationSolver

for solve in (scalar_levenberg_marquardt, scalar_bfgs,
              scalar_lbfgs, scalar_powell_dogleg):
    answer = solve(lambda x: x*x - 2., lambda x: 2*x, 1.5)
    assert answer.success
    assert abs(answer.root - 2.**.5) < 1e-9

f = lambda x: (x[0]-1., x[1]-2., x[0]+x[1]-3.)
answer = vector_lbfgs(f, None, (0., 0.), input_dimension=2, output_dimension=3)
assert answer.success
solver = OptimizationSolver('lbfgs', lambda x: x*x-2., 1.5,
                            jacobian=lambda x: 2*x, scalar=True, history=32)
first = solver.step()
answer = solver.run()
assert answer.success
assert solver.history[-1].root == answer.root
```

`OptimizationSolver` is a resumable front end for these methods. `result`,
`history`, `step()` and `run()` return `OptimizationResult`, with a scalar root
in scalar mode and tuple coordinates in vector mode; its `diagnostics` keep the
full native record. `objective` and `native_state` expose the underlying
objective and method state. The `LevenbergMarquardt`, `PowellDogleg`, `BFGS`
and `LBFGS` classes take `(objective, x0, **options)` directly.

| Method | Step And Acceptance | Distinct Controls |
| --- | --- | --- |
| Levenberg–Marquardt | Damped Gauss–Newton with augmented least squares and a reduction ratio. | `damping=1e-3`, `max_trials=16`, `rcond=1e-12`. |
| Powell dogleg | A scaled trust-region step between the Cauchy and Gauss–Newton directions. | `initial_radius=1`, `max_radius=1e6`, `min_radius=1e-15`. |
| BFGS | A dense inverse-Hessian approximation and bounded strong-Wolfe line search. | `c1=1e-4`, `c2=0.9`, `max_line_search=32`. |
| L-BFGS | BFGS-style line search with a bounded secant-pair ring. | `corrections=10`, plus BFGS line-search controls. |

These methods share the [root stopping defaults](stopping.md),
`max_evaluations=10000` and `gtol=1e-12`. `gtol` bounds the stationarity
measure \(\|J^\mathsf{T}F\|_\infty/(\|F\|_\infty\max_{ij}|J_{ij}|)\), which
does not change when the residual is multiplied by a constant. Stationarity
with an unsatisfied residual is an unsuccessful result. Where a scalar
derivative vanishes, the solve can instead end with `no_acceptable_step`,
`line_search_failure` or `stagnation`.

Only the residual tolerance makes these results successful; a tiny step, a
successful linear solve or a stationary point is not enough. BFGS and L-BFGS
have no bound constraints. L-BFGS limits its correction storage but still uses
a dense residual Jacobian; for a truly matrix-free solve, use
[Newton–Krylov](native-newton-krylov.md). Native callbacks remove Python calls
from the numerical loop, and Python callbacks work as well.

## Native Least-Squares Solver

[Levenberg–Marquardt](native-levenberg-marquardt.md) minimizes the residual
objective with bounded, damped Gauss–Newton steps, using analytic or native
numerical Jacobians, and keeps an inspectable history. It accepts the objective
described above and checks root convergence separately from stationarity.
