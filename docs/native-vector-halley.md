# Native matrix Halley and Super-Halley

`cyroot.native.VectorHalley` and `VectorSuperHalley` solve square and rectangular real systems
with analytic or numerical Jacobians and Hessians. They share the existing
[vector Chebyshev](native-vector-chebyshev.md) callback, cache, history and
stopping engine. Their updates use a coupled secondary matrix solve.

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

## Coupled matrix updates

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
method factors \(J\) for \(s\), then factors its private secondary matrix for
\(v\). The same pivot allocation is reused after the Jacobian solve. No inverse
is formed; no fallback is taken on a singular secondary system.

These are matrix extensions of the corresponding scalar methods, with local
cubic convergence under smoothness, nonsingular-Jacobian and accurate-derivative
assumptions. They can be derived from the operator
\(B=J^{-1}A\), \((I-\alpha B)v=s\), with \(\alpha=1/2\) or \(1\).
The derivation does not commute \(J\) and \(A\). Zero components of \(s\) are
valid. The previous `generalized_halley` componentwise formula had different
semantics; that entry point now uses matrix Halley. See the
[breaking migration guide](vector-higher-order-migration.md) for the explicit
numerical and interface changes.

The full ordered Hessian contributes; analytic Hessians are neither checked for
symmetry nor symmetrized. Native directional contraction uses compensated,
exponent-scaled arithmetic to extend intermediate range, with the same accuracy
limitations as [Chebyshev contraction](native-vector-chebyshev.md#update-and-linear-algebra).
Secondary entries are formed with fused multiply-add before finite validation.
Inaccurate derivatives, cancellation, ill-conditioned systems, final overflow
and intermediate binary64 matrix range remain limitations. There is no damping,
line search or automatic fallback. Rectangular systems use the separately
documented [least-squares extension](native-rectangular-halley.md).

## Providers, caches and execution

Both constructors have the same signature as `VectorChebyshev`:

`VectorHalley(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
hess_x0=None, *, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
max_iter=200, history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.

`VectorSuperHalley` uses the identical arguments. Convenience functions
`vector_halley(...)` and `vector_super_halley(...)` construct and run a solver.
These fixed classes have no `alpha` argument. The separate
[VectorModifiedHalley](native-vector-modified.md) class accepts a finite parameter
and shares their exact special paths. The read-only
`method` property returns `'chebyshev'`, `'halley'` or `'super_halley'`; subclasses
of the fixed methods retain the corresponding native update.

The provider shapes below describe square systems; matching rectangular callbacks
and plans use the shapes in the [rectangular manual](native-rectangular-halley.md).

Residuals accept Python functions or residual-kind `VectorCallback` adapters.
Jacobians accept Python functions, Jacobian-kind `VectorCallback`, matching
initialized `JacobianPlan`, or `None` for a default central plan. Hessians accept
Python functions, `HessianCallback`, matching initialized `HessianPlan`, or `None`
for a default central plan. Points/residuals have length \(n\), Jacobians shape
\((n,n)\), Hessians shape \((n,n,n)\), indexed `[residual][input][input]`.
All supplied caches are independently converted and shape checked under callback
reservations, including at an initial root. Unused derivative finite checks and
callbacks are skipped when the residual already converges. Numerical plans
reuse the finite base residual but have independent stencils and step choices.

All [Chebyshev provider and cache contracts](native-vector-chebyshev.md#providers-and-cached-values)
apply, including pinned contexts, native field validation, copied Python callback
boundaries and derivative-prefix failure handling. Every numerical derivative
invocation counts one attempt in its role and adds all stencil attempts to the
residual count. `step()` and `run()` share one total iteration budget;
The default `max_iter` is 200; zero or `None` is unlimited.

## Diagnostics and retained state

`result` remains `NewtonMethodsReturnType`, with tuple root/residual,
`df_root=(jacobian,hessian)` and `f_calls=(F,J,H)`. The shared immutable step type
is available as both `VectorChebyshevStep` and `VectorHigherOrderStep` (aliases
of the same class). It contains all previous fields plus appended `linear_stage`.
This extends the preview tuple length; compiled consumers must rebuild against
the current installed declarations after the native record layout change.

`linear_stage` is `None`, `'jacobian'` or `'secondary'`. It reports the current
or most recent linear-system phase, including directional contraction and
secondary formation. `linear_status` retains the latest exact LAPACK status in
that phase; it is zero before a LAPACK call in the new phase. These properties
are also copied into step/history records. They are solver/snapshot diagnostics;
the common result type is unchanged.

A nonzero factorization or solve status reports `breakdown` with its exact code
and stage. Nonfinite factors report `nonfinite` even if LAPACK returns zero.
Directional contraction, secondary formation and solution are checked before
any trial callback. Failures there retain the previous finite accepted point,
residual, derivatives and metrics and consume no iteration. A finite candidate
residual attempt counts one iteration, even if the callback fails. Trial failure
retains the previous accepted state; finite accepted trials clear derivatives,
update metrics and check stopping/budget before further derivative work.
Finite unchanged candidates are evaluated before residual success or stagnation.

Callback failures retain stage `'function'`, `'jacobian'` or `'hessian'`, exact
native status or original Python exception identity/traceback. After the failing
call delivers the error, terminal `step()` and `run()` are passive. Immutable
history follows the [native recording rules](native-core.md#stepping-and-history)
and survives resumption/destruction. Adapter reservations cover setup conversions,
execution, exports and error transfer; active solver/adapter reentry is rejected.

## Workspace and native consumers

The following workspace formulas describe square systems. Rectangular matrix
methods add one \(mn\)-double secondary matrix to rectangular Chebyshev workspace.

Matrix methods add one \(n^2\)-double secondary matrix to Chebyshev workspace.
Their total is \(8n+3n^2+n^3+w\) doubles, \(n\) C-int pivots, plus persistent
explicit plan steps, with \(w\) the largest selected numerical-plan scratch.
Chebyshev retains \(8n+2n^2+n^3+w\) doubles. All methods share history stride
\(2n+n^2+n^3\) doubles plus native record metadata per slot. Checked integer
products, native capacities and separate workspace/history byte limits apply
before allocation. Native workspace/history does not allocate per step.
Python owners/exports and provider internals are outside these byte budgets.

The [compiled example](../examples/native_vector_halley.md) uses native residual,
Jacobian and Hessian callbacks with installed `.pxd` and headers, without NumPy
C headers. `vector_chebyshev.pxd` declares the shared engine and both subclasses;
`tensor_ops.pxd` exposes native directional contraction and the finite scalar
negative mean helper. Raw directional output is column-major `out[o+n*k]`;
input Hessian indexing is `H[o+n*(j+n*k)]`. Pointer spans are trusted, disjoint,
and caller-owned. Invalid dimensions fail before writes; other contraction
failures poison the entire directional matrix.

Fully native callbacks and numerical plans execute outside the GIL. Python
callbacks, setup and result/history exports still use it. The package retains
NumPy/SciPy dependencies. Parameterized rectangular updates use the [general correction](native-rectangular-modified.md);
exact alpha selectors retain the fixed formulas. Optional providers are later work.

The established square higher-order entry points now forward to this engine;
see the [breaking migration guide](vector-higher-order-migration.md).
