# Parameterized native matrix modified Halley

`cyroot.native.VectorModifiedHalley` extends the shared vector engine
with a finite real parameter `alpha`. It accepts the same analytic or numerical
Jacobian/Hessian providers, cached values, stepping and bounded copied history
as [matrix Halley and Super-Halley](native-vector-halley.md).

```python
from cyroot.native import VectorModifiedHalley

def residual(x):
    a, b = x
    return (a*a + a*b + b - 5., 2*a + b*b - 6.)

def jacobian(x):
    a, b = x
    return ((2*a + b, a + 1.), (2., 2*b))

def hessian(x):
    return (((2., 1.), (1., 0.)), ((0., 0.), (0., 2.)))

solver = VectorModifiedHalley(residual, jacobian, hessian, (1.25, 1.75),
                              alpha=.25, history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

## Coupled update and exact special paths

For general parameter values, the method computes

\[
Js=F,\qquad A_{o,k}=\sum_j H_{o,j,k}s_j,\qquad
(J-\alpha A)v=F,\qquad Jt=Av,\qquad
\Delta x=-s-\tfrac12t.
\]

The original Jacobian LU factors and pivot ordering are retained for the final
correction solve. The secondary matrix has its own pivots. Matrix multiplication
order matters; \(J\) and \(A\) need not commute. The solver forms no inverse
and never computes \(t=(v-s)/\alpha\), which would lose accuracy near zero.
The general-path step uses fused multiply-add to keep a subnormal half-correction
until the final addition is rounded.

Exactly normalized `alpha=0` (including negative zero) selects the existing
Chebyshev path. Exactly `.5` selects matrix Halley, and exactly `1` selects
Super-Halley, retaining their arithmetic and workspace footprints. Other finite
values use the full correction path. These special dispatches do not change the
public `method` value, which is always `'modified_halley'` for this class.

For square systems, local cubic convergence requires the usual smoothness, nonsingular Jacobian and
accurate derivatives. Parameter choices can make the secondary matrix singular,
change convergence behavior or amplify conditioning problems. There is no
parameter auto-selection, damping or automatic fallback. All finite alpha values also support a specified
[rectangular least-squares correction](native-rectangular-modified.md). Exactly
`alpha=0/.5/1` select distinct fixed methods; rectangular steps can be
discontinuous near `.5` and `1`, as explained in that manual.
The established square higher-order entry points now use this engine and the
old componentwise default is removed; see the
[breaking migration guide](vector-higher-order-migration.md). Unrelated legacy
rectangular methods retain their existing interfaces.

## Configuration and providers

The constructor is
`VectorModifiedHalley(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
hess_x0=None, *, alpha=.5, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
max_iter=200, history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.
`vector_modified_halley(function, jacobian, hessian, x0, **options)` constructs
and runs one solver. The read-only `alpha` property returns the normalized
binary64 value. Python subclasses retain the native parameterized method;
overriding a public property does not alter native execution.

`alpha` is converted once while the solver and all explicit callback adapters
are reserved. Finite custom real conversions are supported. Text, sequences,
booleans, complex values, NumPy boolean/complex scalars and nonfinite parameters
are rejected, including at a cached initial root. Failed conversion releases
reservations. The fixed Chebyshev/Halley/Super-Halley constructors retain their
signatures and reject an `alpha` argument.

The following provider shapes and workspace formulas describe square systems.
For rectangular general and special paths, see the
[rectangular manual](native-rectangular-modified.md).

Residuals accept Python functions or residual-kind `VectorCallback`; Jacobians
accept Python functions, Jacobian-kind `VectorCallback`, matching initialized
`JacobianPlan`, or `None` for the default central plan. Hessians accept Python
functions, `HessianCallback`, matching initialized `HessianPlan`, or `None`.
Points/residuals have length \(n\), Jacobians shape \((n,n)\), and Hessians
shape \((n,n,n)\) indexed `[residual][input][input]`. Native provider kinds,
initialization flags and dimensions are checked directly. All supplied cache
conversions and shapes are checked under reservations; unused derivative finite
checks are skipped when the initial residual converges.

All [shared provider/cache contracts](native-vector-chebyshev.md#providers-and-cached-values)
apply. Numerical Jacobian and Hessian plans independently reuse the base residual.
Their sample steps need not agree, and their stencil samples are not shared.
Result `f_calls=(F,J,H)` counts each derivative-plan invocation in its own role
and every residual sample in \(F\), including failed attempts. Cached values
count zero new calls. Python callback inputs are copied tuples; native contexts
are pinned by their adapters. No new tensor callback ABI is introduced here.

## Diagnostics, failure state and range

The common result remains `NewtonMethodsReturnType`, with tuple root/residual,
`df_root=(jacobian,hessian)`, counts and termination reason. Immutable snapshots
remain `VectorHigherOrderStep`, also named `VectorChebyshevStep`. `linear_stage`
adds the value `'correction'` for the final matrix-vector contraction and retained
Jacobian solve. The earlier values remain `None`, `'jacobian'` and `'secondary'`.
The common tuple field set and history record layout do not change in this stage;
the native solver object gains fields, so compiled consumers must rebuild against
current installed declarations.

Each new phase resets `linear_status` to zero before work, then preserves the
exact latest LAPACK status from that phase. Nonzero factor/solve status reports
`breakdown`. Nonfinite LU factors report `nonfinite` even with status zero.
Nonfinite directional contraction, secondary formation, solution or correction
stops before any trial callback; previous accepted point, residual, derivatives
and metrics remain available. Secondary entries use fused multiply-add.

Matrix-vector contraction uses scaled binary exponents and compensated summation,
matching the native tensor helpers' range policy. It avoids premature product
overflow/underflow but does not increase precision. Ill-conditioned systems,
derivative errors, cancellation, final overflow and other binary64 intermediate
range limits remain. Finite completion does not certify derivative accuracy or
convergence. There is no recovery through a different update on failure.

All shared state rules remain: finite trial residual attempts count iterations;
failed trials preserve accepted state; accepted trials clear derivatives and
check stopping/budget before further derivative work. Jacobian failure clears
both derivative roles, while Hessian failure preserves a valid Jacobian. Native
callback errors retain exact code/stage; Python exceptions retain identity and
traceback. After an error is delivered, terminal `step()`/`run()` are passive.
`step()` and `run()` share a total budget; the default is 200, and zero or `None` is unlimited.

## Workspace and compiled use

Let \(B=8n+2n^2+n^3+w\) doubles, where \(w\) is the largest selected numerical
Jacobian/Hessian scratch requirement. `alpha=0` needs \(B\) doubles and \(n\)
C-int pivots. `.5` and `1` need \(B+n^2\) doubles and \(n\) reused pivots.
The general path needs \(B+2n^2\) doubles and \(2n\) pivots. Both plans'
persistent explicit step storage also counts against `workspace_bytes`, even
if shared. History remains \(2n+n^2+n^3\) doubles plus native metadata per slot.
Checked products, byte budgets and native capacities precede allocation.
Python owners/exports and backend internals use additional memory. Native cy-root
workspace/history does not allocate per step.

The [compiled example](../examples/native_vector_modified.md) uses installed
Cython declarations, native callbacks and pinned context ownership without NumPy
C headers. `vector_chebyshev.pxd` declares the shared engine and parameterized
subclass. `tensor_ops.pxd` exposes `matrix_vector` for trusted disjoint packed
column-major matrix, direction and output spans. Invalid dimensions fail before
writes; other contraction failures poison the whole output.

Fully native callbacks and plans execute outside the GIL. Python callbacks,
setup and result/history export use it. NumPy/SciPy remain package dependencies;
callback/storage APIs remain previews. Remaining native solvers, shape policies
and optional linear algebra providers are later work.
