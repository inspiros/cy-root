# Native Vector Chebyshev

`cyroot.native.VectorChebyshev` solves square and rectangular real systems
with analytic or numerical Jacobians and Hessians. This page describes square
systems; the [rectangular extension](native-rectangular-chebyshev.md) uses
rank-aware least-squares corrections. The solver supports Python and compiled
callbacks, resumable execution and bounded copied history. Its numerical loop
uses native storage, SciPy's Cython LAPACK interface and native tensor
contraction, with no NumPy operations or per-step allocations.

```python
from cyroot.native import VectorChebyshev

def residual(x):
    a, b = x
    return (a*a + a*b + b - 5., 2*a + b*b - 6.)

def jacobian(x):
    a, b = x
    return ((2*a + b, a + 1.), (2., 2*b))

def hessian(x):
    return (((2., 1.), (1., 0.)), ((0., 0.), (0., 2.)))

solver = VectorChebyshev(residual, jacobian, hessian, (1.1, 1.9), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

## Update And Linear Algebra

At an accepted point the solver computes

\[
Js=F,\qquad q=H[s,s],\qquad Jt=q,\qquad
\Delta x=-s-\tfrac12t.
\]

The Hessian contraction is
\(q_o=\sum_{j,k}H_{o,j,k}s_js_k\). Every ordered pair of inputs contributes;
analytic Hessians are neither checked for symmetry nor symmetrized. A zero
component of \(s\) is fine. The solver copies \(J\), factors it once with
LAPACK `dgetrf` and reuses the factors for both `dgetrs` solves; no inverse is
formed. A nonzero LAPACK status reports `breakdown`, with the status in
`linear_status`. Nonfinite factors or solutions report `nonfinite` even when
LAPACK reports success, since that status alone does not guarantee finite
factors.

With accurate derivatives, the method converges cubically near a smooth root
with a nonsingular Jacobian. [Matrix Halley and Super-Halley](native-vector-halley.md)
share this state engine with their own coupled updates, and the public
higher-order vector functions run it; see the
[migration guide](vector-higher-order-migration.md).

The contraction uses binary64 mantissas with explicit binary exponents and
compensated summation. This avoids premature overflow and underflow in the
individual triple products and lets some sums of individually unrepresentable
terms come out finite; it extends the range, not the precision. Inaccurate
derivatives, cancellation, ill-conditioned linear systems and overflow in the
final values remain limitations, and LAPACK or the step arithmetic can still
report a failure conservatively. There is no damping, line search or
least-squares fallback for a singular square system.

## Providers And Cached Values

The constructor is
`VectorChebyshev(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
hess_x0=None, *, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
max_iter=200, history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.
`vector_chebyshev(function, jacobian, hessian, x0, **options)` constructs a
solver and runs it to completion. The default `max_iter` is 200; zero or `None`
is unlimited. Native callbacks use versioned descriptors.

`function` is a Python residual or a residual `VectorCallback`. `jacobian` is a
Python function, a Jacobian `VectorCallback`, a `JacobianPlan` of matching
size, or `None` for a default central plan. `hessian` is a Python function, a
`HessianCallback`, a `HessianPlan` of matching size, or `None` for a default
central plan. Native dimensions, flags and kinds are validated, also for
callable subclasses and when the cached starting point is a root.

Points and residuals have length \(n\), Jacobians shape \((n,n)\) and Hessians
shape \((n,n,n)\), indexed `[residual][input][input]`. Jacobians can be nested
sequences or two-dimensional buffers. Hessians from Python providers or caches
are read by nested indexing, so NumPy arrays of the right shape work, including
strided and read-only ones; they are always copied. Hessian entries must be
real scalars. Python callbacks receive a copied point tuple, which they may
keep, and may reuse their output containers.

`f_x0`, `jac_x0` and `hess_x0` are independent caches of the values at the
starting point and count as zero evaluations. All of them are converted and
their shapes checked during setup. If the starting point is already a root,
unused derivatives are not checked for finiteness and the supplied caches are
kept. Otherwise the Jacobian is evaluated before the Hessian. Numerical plans
reuse the residual at the current point but sample the axes independently,
since the plans can use different steps.
See [Jacobian plans](native-jacobian.md) and [Hessian plans](native-hessian.md)
for rounded stencils, sample counts and accuracy limits.

## State, Counts And Failures

`step()` and `run()` share one iteration budget and the
[shared stopping rules](stopping.md). An iteration is one attempt to evaluate
the residual at a finite trial point, even if the callback fails. A candidate
equal to the current point is still evaluated before success or stagnation is
decided. Accepting a finite candidate clears the Jacobian and Hessian, and the
stopping rules and budget are checked before more derivative work. A failed
trial keeps the previous finite point, residual, derivatives and metrics and
updates only the counts and diagnostics.

`result` is a `NewtonMethodsReturnType`. Its `root` and `f_root` are tuples, and
`df_root` is `(jacobian, hessian)` as nested tuples. `f_calls` is `(F, J, H)`:
each plan evaluation counts once for its derivative, and all of its residual
samples also count in \(F\); failed attempts count too. Derivatives that are
not available are NaN. A Jacobian failure clears both derivatives; a Hessian
failure keeps a valid Jacobian and clears the Hessian.

`VectorChebyshevStep` contains `root`, `f_root`, `df_root`, `iters`, `f_calls`,
`precision`, `error`, `termination_reason`, `callback_code`, `callback_stage`,
`linear_status` and `linear_stage` (`None`, `'jacobian'` or `'secondary'`).
`VectorHigherOrderStep` is another name for the same class. Chebyshev uses only
the Jacobian stage, and its read-only `method` property returns `'chebyshev'`.
Snapshots and history are immutable copies that survive later steps and the
solver's destruction; history follows the
[native recording rules](native-core.md#step-inspect-resume).
`result`, `done`, `history`, `history_truncated`, `callback_code`,
`callback_stage`, `linear_status` and `linear_stage` read the state without
advancing it.

A nonzero native callback status raises `NativeCallbackError` with the exact
`native_status`, a copied result and the stage `'function'`, `'jacobian'` or
`'hessian'`, also for residual samples taken by numerical plans. Python
exceptions propagate as the original instance, with their traceback. The
solver is then finished: after the failing call raises, later `step()` or
`run()` calls only return the failed state. Every callback output is set to NaN
beforehand, so a value the callback did not write can never be accepted.

All explicit adapters are inspected before any is reserved. They stay reserved
through the conversion of dimensions, options and caches, allocation, numerical
execution, result conversion and exception transfer. Reentrant or concurrent
use of an active solver or adapter raises an error. Independent callbacks can
share immutable plans, each with its own workspace. Callers that share mutable
native contexts must synchronize them.

## Storage Budgets

For dimension \(n\), define \(w\) as the larger of \(5n\) for a numerical
Jacobian and \(2n^2+10n\) for a numerical Hessian; omit a term when that role
is analytic. The solver allocates \(8n+2n^2+n^3+w\) doubles and \(n\) C-int
pivots up front. The workspace budget also counts both plans' explicit steps,
even when the plans are shared. History costs another \(2n+n^2+n^3\) doubles
plus a metadata record per requested slot. Integer products and capacities are
checked before allocation. Python owners and exports and memory used inside
LAPACK are outside these budgets. Workspace and history are allocated once;
nothing is allocated per step.

## Compiled Hessian Callbacks

Installed `tensor_callbacks.pxd` exposes
`make_hessian_callback(function, dimension, context, keepalive)` with signature
`int callback(const double* x, size_t n, double* out, void* context) noexcept nogil`.
Write all \(n^3\) entries in residual-first column-major order:
`out[o + n*(j + n*k)]`. The input is borrowed only for the call; do not modify
it or keep its pointer. Return zero on success and a nonzero status on failure,
and never let Python or C++ exceptions cross this ABI. The adapter keeps the
supplied keepalive object alive as long as it exists.

C and C++ code can instead put a `cyroot_hessian_descriptor` from `callbacks.h`
in a capsule named `cyroot.hessian_callback.v1` and call
`HessianCallback.from_capsule(capsule, keepalive=owner)`. The adapter checks the
name, version, minimum size, dimension and that the function is not null, then
copies the function and context and keeps the owners alive. The capsule name
cannot prove that the pointers are valid. This tensor ABI is separate from the
`cyroot.vector_callback.v1` residual and Jacobian capsules.

When every evaluated role, including the residual sampled by numerical plans,
has a native callback, execution stays outside the GIL. Python callbacks take
it for each evaluation, as do setup and result and history export. The
[compiled example](../examples/native_vector_chebyshev.md) uses the installed
headers and declarations without NumPy C headers. Rectangular corrections are
described in the [rectangular Chebyshev guide](native-rectangular-chebyshev.md).
