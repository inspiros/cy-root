# Parameterized Native Matrix Modified Halley

`cyroot.native.VectorModifiedHalley` adds a finite real parameter `alpha` to
the shared vector engine. It accepts the same analytic or numerical Jacobian
and Hessian providers, cached values, stepping and bounded copied history as
[matrix Halley and Super-Halley](native-vector-halley.md).

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

## Coupled Update And Exact Special Paths

For general parameter values, the method computes

\[
Js=F,\qquad A_{o,k}=\sum_j H_{o,j,k}s_j,\qquad
(J-\alpha A)v=F,\qquad Jt=Av,\qquad
\Delta x=-s-\tfrac12t.
\]

The Jacobian's LU factors and pivots are reused for the final correction
solve, and the secondary matrix has its own pivots. The order of matrix
products matters, since \(J\) and \(A\) need not commute. No inverse is formed,
and \(t\) is never computed as \((v-s)/\alpha\), which would lose accuracy for
small \(\alpha\). The general step uses fused multiply-add so that a subnormal
half-correction is kept until the final addition is rounded.

`alpha` exactly zero (including negative zero) uses the Chebyshev path, exactly
`.5` matrix Halley and exactly `1` Super-Halley, with their arithmetic and
workspace sizes. Other finite values use the general correction. The `method`
property is `'modified_halley'` for this class whichever path is used.

For square systems, the method converges cubically near a smooth root with a
nonsingular Jacobian, given accurate derivatives. Some parameter values can
make the secondary matrix singular, change the convergence behavior or amplify
conditioning problems. There is no automatic parameter choice, damping or
fallback. Every finite `alpha` also has a
[rectangular least-squares correction](native-rectangular-modified.md); because
`alpha=0`, `.5` and `1` select the fixed methods, rectangular steps can jump
near `.5` and `1`, as that page explains. The public higher-order vector
functions run this engine and the componentwise default of 1.x is gone; see the
[migration guide](vector-higher-order-migration.md).

## Configuration And Providers

The constructor is
`VectorModifiedHalley(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
hess_x0=None, *, alpha=.5, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
max_iter=200, history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.
`vector_modified_halley(function, jacobian, hessian, x0, **options)`
constructs a solver and runs it to completion. The read-only `alpha` property
returns the value as a binary64 float. Python subclasses keep the native
parameterized method; overriding a property does not change what runs.

`alpha` is converted once, while the solver and all explicit callback adapters
are reserved. Any object with a finite real conversion is accepted; text,
sequences, booleans, complex values, NumPy boolean and complex scalars and
nonfinite values are rejected, even when the cached starting point is a root. A
failed conversion releases the reservations. The fixed Chebyshev, Halley and
Super-Halley constructors reject an `alpha` argument.

The provider shapes and workspace formulas below are for square systems; for
rectangular systems, see the [rectangular manual](native-rectangular-modified.md).

Residuals are Python functions or residual `VectorCallback` adapters.
Jacobians are Python functions, Jacobian `VectorCallback` adapters,
`JacobianPlan`s of matching size, or `None` for the default central plan.
Hessians are Python functions, `HessianCallback` adapters, `HessianPlan`s of
matching size, or `None`. Points and residuals have length \(n\), Jacobians
shape \((n,n)\) and Hessians shape \((n,n,n)\), indexed
`[residual][input][input]`. Provider kinds, initialization and dimensions are
checked, and so are the conversion and shape of every supplied cache; when the
starting residual already converges, unused derivatives are not checked for
finiteness.

All the [shared provider and cache rules](native-vector-chebyshev.md#providers-and-cached-values)
apply. Numerical Jacobian and Hessian plans each reuse the residual at the
current point, but their steps can differ and they share no samples.
`f_calls=(F, J, H)` counts each plan evaluation for its derivative and every
residual sample in \(F\), failures included; cached values count as zero.
Python callbacks receive copied tuples, and adapters keep native contexts
alive.

## Diagnostics, Failure State And Range

The result is a `NewtonMethodsReturnType`, with tuple root and residual,
`df_root=(jacobian, hessian)`, counts and termination reason. Snapshots are
immutable `VectorHigherOrderStep` records (also named `VectorChebyshevStep`).
Besides `None`, `'jacobian'` and `'secondary'`, `linear_stage` can be
`'correction'` for the final matrix-vector product and the second Jacobian
solve.

Each new phase resets `linear_status` to zero and then keeps the latest LAPACK
status in that phase. A nonzero factorization or solve status reports
`breakdown`, and nonfinite LU factors report `nonfinite` even with status zero.
A nonfinite directional contraction, secondary matrix, solution or correction
stops the solve before any trial callback, keeping the previous accepted point,
residual, derivatives and metrics. Secondary entries use fused multiply-add.

The matrix-vector product uses scaled binary exponents and compensated
summation, like the other native tensor helpers. This avoids premature overflow
and underflow in products but does not increase precision. Ill-conditioned
systems, derivative errors, cancellation, final overflow and the binary64 range
of intermediates remain limitations. Finishing with finite values says nothing
about derivative accuracy or convergence, and a failure is not retried with a
different update.

The shared state rules apply: each attempt to evaluate the residual at a finite
trial point is an iteration, a failed trial keeps the accepted state, and an
accepted trial clears the derivatives and checks the stopping rules and budget
before any more derivative work. A Jacobian failure clears both derivatives,
while a Hessian failure keeps a valid Jacobian. Native callback errors keep
their code and stage, and Python exceptions propagate as the original instance
with its traceback. After an error is raised, later `step()` and `run()` calls
only return the state. `step()` and `run()` share one budget; the default
`max_iter` is 200, and zero or `None` is unlimited.

## Workspace And Compiled Use

Let \(B=8n+2n^2+n^3+w\) doubles, where \(w\) is the largest selected numerical
Jacobian/Hessian scratch requirement. `alpha=0` needs \(B\) doubles and \(n\)
C-int pivots. `.5` and `1` need \(B+n^2\) doubles and \(n\) reused pivots.
The general path needs \(B+2n^2\) doubles and \(2n\) pivots. Both plans'
explicit steps also count against `workspace_bytes`, even if the plans are
shared. History takes \(2n+n^2+n^3\) doubles plus a metadata record per slot.
Products, byte budgets and capacities are checked before allocation, and
nothing is allocated per step. Python owners and exports and memory used inside
LAPACK come on top.

The [compiled example](../examples/native_vector_modified.md) uses the
installed Cython declarations and native callbacks, with contexts kept alive by
their adapters and without NumPy C headers. `vector_chebyshev.pxd` declares the
shared engine and the parameterized subclass. `tensor_ops.pxd` exposes
`matrix_vector` for valid, non-overlapping packed column-major matrix,
direction and output spans. Invalid dimensions fail before anything is
written; any other contraction failure fills the whole output with NaN.

With native callbacks and plans, execution happens outside the GIL; Python
callbacks, setup and result and history export take it. The callback and
storage contracts are described in the [native core guide](native-core.md).
