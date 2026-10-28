# Root finding through optimization

For a real residual function `F: R^n -> R^m`, define

\[
\phi(x)=\tfrac12\sum_{i=1}^m F_i(x)^2.
\]

Every zero of `F` gives a zero global minimum of this nonnegative objective.
Conversely, a zero objective gives a root. A positive local minimum or a small
objective gradient does not establish a root. The same construction supports
square, overdetermined and underdetermined residuals.

## Value, gradient and exact Hessian

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
from `cyroot.native`. Omitting `output_dimension` means a square residual.
Ordinary Python providers receive a copied tuple and return indexable real
values with shapes `(m,)`, `(m,n)` and `(m,n,n)` respectively. Compatible native
`VectorCallback`, `RectangularCallback`, `HessianCallback`, and
`RectangularHessianCallback` providers are accepted with their appropriate kinds
and dimensions. Native context owners stay pinned for the adapter's lifetime.

The methods `value(x)` (also `objective(x)`), `residual(x)`, `gradient(x)` and
`hessian(x)` return a float or copied immutable tuples. They compute

\[
\nabla\phi=J^T F,\qquad
\nabla^2\phi=J^TJ+\sum_i F_i\nabla^2F_i.
\]

The Hessian includes the residual-Hessian term. A Hessian provider requires a
Jacobian provider; missing providers raise an error before sampling. There is
no implicit Gauss–Newton approximation or finite-difference fallback. A user
who wants numerical derivatives can supply explicit derivative callables.
Every call takes fresh samples: value/residual calls `F` once; gradient calls
`F,J` once each; exact Hessian calls `F,J,H` once each. There is no sample cache
between optimizer calls and no hidden forwarding of optimizer-specific options.

The adapter checks dimensions/storage and allocates native scratch once.
`memory_bytes` reports that allocation; Python boundary objects and provider
allocations are additional. Exponent-scaled compensated accumulation avoids
unnecessary intermediate overflow/underflow but does not provide extra accuracy
for ill-conditioned cancellation. Unrepresentable final results or nonfinite
inputs raise `FloatingPointError`; shape/kind/provider errors raise exceptions.
Original Python callback exceptions are propagated; native nonzero statuses raise
`NativeCallbackError` with callback stage/code. Partial callback output is detected.

## External optimizers and root status

`root_via_optimizer(objective, optimizer, x0, *, etol=1e-10, ertol=0.,
optimizer_options=None)` calls `optimizer(objective, copied_x0, **options)`.
The returned object must have an `x` attribute. This convention fits
`scipy.optimize.minimize`; wrap other optimizer interfaces in an ordinary
callable. Pass `jac=objective.gradient`, `hess=objective.hessian`, method,
bounds and algorithm tolerances through the optimizer's own options. Explicit
optimizer derivative requests must match the providers installed on the adapter.

The result contains copied `root`, `f_root`, infinity-norm `error`, root
`success`, and `termination_reason`, plus `optimizer_success`, `optimizer_status`
and the original `optimizer_result`. Original optimizer exceptions propagate.
Root success requires

\[
\|F(x)\|_\infty\le \text{etol}+\text{ertol}\,\|F(x_0)\|_\infty.
\]

The initial scale is fixed. A root at the initial point skips optimization and
returns `None` optimizer metadata. Otherwise the returned optimizer point is
independently sampled and checked. Optimizer success with an unsatisfied root
tolerance gives `optimizer_stopped_without_root`. Conversely, a returned point
meeting the root tolerance is a successful root even when the optimizer reports
failure; its original status remains visible. Invalid/nonfinite optimizer points
raise the corresponding objective exception. A no-root result is inconclusive
about other points or global root existence.

For example, `F(x,y)=(x^3-2*x+2,y)` has a strict positive local minimum of `phi`
at `(sqrt(2/3),0)`, despite a root existing at negative `x`. A convex optimizer's
convergence theorem does not automatically apply: squared nonlinear residuals
are generally nonconvex. A stationary point implies a root if its Jacobian has
full **row** rank; full column rank in an overdetermined system is insufficient.

External Python optimizers use Python and the GIL. The objective's numerical
kernel can be called without the GIL by compiled code using native callbacks;
this does not turn external optimizer orchestration into a GIL-free solve loop.
An objective instance and its attached native adapters are reserved during each
operation, including input conversion, error transfer and result export. Use
separate states/adapters for concurrent evaluations.

## Higher-order objective derivatives

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
exactly `r` ordinary Python-callable providers for `DF` through `D^rF`; it calls
each provider once and packs each output before calling the next. Providers may
reuse their own scratch arrays. These convenience calls use Python/GIL.

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

## Compiled interface

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

## Native least-squares solver

[Levenberg–Marquardt](native-levenberg-marquardt.md) solves the residual objective
using bounded damped Gauss–Newton steps, analytic/native numerical Jacobians and
inspectable history. It accepts the objective adapter described above and checks
root convergence independently of stationarity.
