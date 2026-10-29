# Native Matrix Halley And Super-Halley

`cyroot.native.VectorHalley` and `VectorSuperHalley` solve square and
rectangular real systems with analytic or numerical Jacobians and Hessians.
They share the callback, cache, history and stopping engine of
[vector Chebyshev](native-vector-chebyshev.md), and their updates add a coupled
solve with a secondary matrix.

```python
from cyroot.native import VectorHalley, VectorSuperHalley

def residual(x):
    a, b = x
    return (a*a + a*b + b - 5., 2*a + b*b - 6.)

def jacobian(x):
    a, b = x
    return ((2*a + b, a + 1.), (2., 2*b))

def hessian(x):
    return (((2., 1.), (1., 0.)), ((0., 0.), (0., 2.)))

for method in (VectorHalley, VectorSuperHalley):
    solver = method(residual, jacobian, hessian, (1.25, 1.75), history=16)
    first = solver.step()
    result = solver.run()
    assert result.success
    assert solver.history[-1].root == result.root
```

## Coupled Matrix Updates

The formulas and LU policy in this section describe square systems. For
rectangular systems, see the [least-squares extension](native-rectangular-halley.md).

At each accepted point, first solve \(Js=F\) and form the directional Hessian
matrix \(A\), where

\[
A_{o,k}=\sum_j H_{o,j,k}s_j,\qquad Av=H[s,v].
\]

Halley solves

\[
(J-\tfrac12A)v=F,\qquad \Delta x=-v.
\]

Super-Halley solves

\[
(J-A)v=F,\qquad \Delta x=-\tfrac12(s+v).
\]

The Super-Halley mean avoids overflow from adding two large finite vectors and
premature underflow from halving two subnormal components separately. Each
method factors \(J\) to get \(s\), then factors its own secondary matrix to get
\(v\), reusing the pivot storage. No inverse is formed, and a singular
secondary system has no fallback.

These are matrix extensions of the scalar methods of the same names and, with
accurate derivatives, converge cubically near a smooth root with a nonsingular
Jacobian. They can be derived from the operator \(B=J^{-1}A\) and
\((I-\alpha B)v=s\) with \(\alpha=1/2\) or \(1\), without assuming that \(J\)
and \(A\) commute. Zero components of \(s\) are fine. The componentwise
`generalized_halley` formula of 1.x behaved differently; the function now uses
matrix Halley. The [migration guide](vector-higher-order-migration.md) lists the
numerical and interface changes.

The whole ordered Hessian contributes; analytic Hessians are neither checked
for symmetry nor symmetrized. The directional contraction uses compensated,
exponent-scaled arithmetic to extend the intermediate range, with the same
accuracy limits as the
[Chebyshev contraction](native-vector-chebyshev.md#update-and-linear-algebra).
Secondary entries are formed with fused multiply-add before they are checked
for finiteness. Inaccurate derivatives, cancellation, ill-conditioned systems,
final overflow and the binary64 range of the secondary matrix remain
limitations. There is no damping, line search or automatic fallback.

## Providers, Caches And Execution

Both constructors have the same signature as `VectorChebyshev`:

`VectorHalley(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
hess_x0=None, *, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
max_iter=200, history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.

`VectorSuperHalley` takes the same arguments, and `vector_halley(...)` and
`vector_super_halley(...)` construct a solver and run it to completion. These
fixed methods have no `alpha` argument; [VectorModifiedHalley](native-vector-modified.md)
accepts any finite parameter and uses their exact arithmetic at the matching
values. The read-only `method` property returns `'chebyshev'`, `'halley'` or
`'super_halley'`, and subclasses keep the update of the class they extend.

The provider shapes below are for square systems; rectangular callbacks and
plans use the shapes in the [rectangular manual](native-rectangular-halley.md).

Residuals are Python functions or residual `VectorCallback` adapters. Jacobians
are Python functions, Jacobian `VectorCallback` adapters, `JacobianPlan`s of
matching size, or `None` for a default central plan. Hessians are Python
functions, `HessianCallback` adapters, `HessianPlan`s of matching size, or
`None` for a default central plan. Points and residuals have length \(n\),
Jacobians shape \((n,n)\) and Hessians shape \((n,n,n)\), indexed
`[residual][input][input]`. Every supplied cache is converted and its shape
checked while the callbacks are reserved, even when the starting point is a
root. When the starting residual already converges, unused derivatives are
neither evaluated nor checked. Numerical plans reuse the residual at the
current point but have their own stencils and steps.

All the [Chebyshev provider and cache rules](native-vector-chebyshev.md#providers-and-cached-values)
apply, including context lifetimes, native field validation, copied Python
callback inputs and the handling of derivative failures. Every numerical
derivative evaluation counts one attempt for its derivative and adds all of its
samples to the residual count. `step()` and `run()` share one iteration budget;
the default `max_iter` is 200, and zero or `None` is unlimited.

## Diagnostics And Retained State

`result` is a `NewtonMethodsReturnType`, with tuple root and residual,
`df_root=(jacobian, hessian)` and `f_calls=(F, J, H)`. Steps are immutable
`VectorChebyshevStep` records (also named `VectorHigherOrderStep`), which
include the `linear_stage` of the most recent linear operation.

`linear_stage` is `None`, `'jacobian'` or `'secondary'`: the current or most
recent linear phase, including the directional contraction and the formation
of the secondary matrix. `linear_status` holds the latest LAPACK status in that
phase, and is zero until a LAPACK call is made in it. Both are also copied into
step and history records; the common result type does not include them.

A nonzero factorization or solve status reports `breakdown` with its code and
stage. Nonfinite factors report `nonfinite` even if LAPACK reports success.
The directional contraction, the secondary matrix and its solution are checked
before any trial callback; a failure there keeps the previous accepted point,
residual, derivatives and metrics and uses no iteration. Each attempt to
evaluate the residual at a finite candidate counts one iteration, even if the
callback fails. A failed trial keeps the previous accepted state; an accepted
trial clears the derivatives, updates the metrics and checks the stopping rules
and budget before any more derivative work. A candidate equal to the current
point is still evaluated before success or stagnation is decided.

Callback failures report the stage `'function'`, `'jacobian'` or `'hessian'`,
with the exact native status or the original Python exception and traceback.
After the failing call raises, later `step()` and `run()` calls only return the
state. Immutable history follows the
[native recording rules](native-core.md#step-inspect-resume) and survives
resuming or destroying the solver. Adapter reservations cover setup
conversions, execution, exports and error transfer; reentering an active solver
or adapter is rejected.

## Workspace And Native Consumers

For square systems, the matrix methods add an \(n^2\)-double secondary matrix
to the Chebyshev workspace, for a total of \(8n+3n^2+n^3+w\) doubles and \(n\)
C-int pivots, plus the plans' explicit steps, where \(w\) is the largest
numerical-plan scratch. (Chebyshev needs \(8n+2n^2+n^3+w\) doubles.) For
rectangular systems they add an \(mn\)-double secondary matrix to the
rectangular Chebyshev workspace. Every method stores \(2n+n^2+n^3\) doubles plus
a metadata record per history slot. Integer products, capacities and the
separate workspace and history byte limits are checked before allocation, and
nothing is allocated per step. Python owners and exports and memory used inside
LAPACK are outside these budgets.

The [compiled example](../examples/native_vector_halley.md) uses native
residual, Jacobian and Hessian callbacks with the installed `.pxd` files and
headers, without NumPy C headers. `vector_chebyshev.pxd` declares the shared
engine and both subclasses. `tensor_ops.pxd` exposes the native directional
contraction and a helper for the finite negative mean of two scalars. The
directional output is column-major, `out[o+n*k]`, and the input Hessian is
indexed `H[o+n*(j+n*k)]`. Callers own the spans and must pass valid,
non-overlapping ones. Invalid dimensions fail before anything is written; any
other contraction failure fills the whole directional matrix with NaN.

With native callbacks and numerical plans, execution happens outside the GIL;
Python callbacks, setup and result and history export take it. The
parameterized rectangular update uses the
[general correction](native-rectangular-modified.md), while the exact fixed
values of `alpha` keep the fixed formulas. The public higher-order vector
functions run this engine; see the
[migration guide](vector-higher-order-migration.md).
