# Native vector Chebyshev

`cyroot.native.VectorChebyshev` solves square and rectangular real systems with
analytic or numerical Jacobians and Hessians. The
[rectangular extension](native-rectangular-chebyshev.md) uses rank-aware
least-squares corrections; the square behavior is described below. It supports Python and compiled callbacks,
resumable execution and bounded copied history. The numerical loop uses native
storage, SciPy's Cython LAPACK provider and native tensor contraction, without
NumPy operations or per-step allocations.

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

## Update and linear algebra

At an accepted point the solver computes

\[
Js=F,\qquad q=H[s,s],\qquad Jt=q,\qquad
\Delta x=-s-\tfrac12t.
\]

The Hessian contraction is
\(q_o=\sum_{j,k}H_{o,j,k}s_js_k\). The full ordered input pairs contribute;
analytic Hessians are neither checked for symmetry nor symmetrized. A zero
component of \(s\) is valid. The solver copies \(J\), factors it once with
LAPACK `dgetrf`, then uses the same factors for both `dgetrs` solves. It forms
no matrix inverse. A nonzero LAPACK status reports `breakdown` with exact
`linear_status`. Nonfinite factors or solve outputs report `nonfinite`, even
when LAPACK returns zero: successful status alone does not certify finite LU.

The method has local cubic convergence under the usual smoothness, nonsingular
Jacobian and accurate derivative assumptions. This is the vector Chebyshev
update. [Native matrix Halley and Super-Halley](native-vector-halley.md) share
the state engine with separate coupled updates. The established square
higher-order entry points now use this engine; see the
[breaking migration guide](vector-higher-order-migration.md).

Contraction uses binary64 mantissas, explicit binary exponents and compensated
summation. It avoids premature overflow/underflow in individual triple products
and permits some finite sums of individually unrepresentable terms. This extends
range, not precision. Inaccurate derivatives, cancellation, ill-conditioned
linear systems and final overflow remain limitations. LAPACK and candidate-step
arithmetic can still report conservative numerical failure. No damping,
line search, least-squares fallback or rectangular policy is provided.

## Providers and cached values

The constructor is
`VectorChebyshev(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
hess_x0=None, *, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
max_iter=200, history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.
`vector_chebyshev(function, jacobian, hessian, x0, **options)` constructs and
runs one solver. `max_iter=0` is unlimited; `None` uses the default 200.
All native callback interfaces remain drafts.

`function` accepts a Python residual or residual-kind `VectorCallback`.
`jacobian` accepts a Python function, Jacobian-kind `VectorCallback`, matching
initialized `JacobianPlan`, or `None` for a default central plan. `hessian`
accepts a Python function, `HessianCallback`, matching initialized `HessianPlan`,
or `None` for a default central plan. Native dimensions, flags and kinds are
validated directly, including callable subclasses and cached initial roots.

Points and residuals have length \(n\), Jacobians shape \((n,n)\), Hessians
shape \((n,n,n)\), indexed `[residual][input][input]`. Jacobians accept copied
nested sequences or supported rank-two native double buffers. Python Hessian
providers and cached Hessians use copied nested indexing; NumPy arrays with
valid shape, including strided and read-only arrays, work through that boundary.
No rank-three zero-copy buffer contract is introduced. Hessian entries must
be real scalars. Python callbacks receive copied point tuples and may retain
them or reuse their output containers.

`f_x0`, `jac_x0` and `hess_x0` independently cache values at the supplied point.
Caches count zero new evaluations. All supplied caches are converted and shape
checked during guarded setup. An initially converged residual skips unused
derivative finite checks and preserves supplied caches. Otherwise the Jacobian
is acquired before the Hessian. Numerical plans reuse the finite base residual;
their axis samples are independent because the plans can use different steps.
See [Jacobian plans](native-jacobian.md) and [Hessian plans](native-hessian.md)
for rounded stencils, sample counts and accuracy limits.

## State, counts and failures

`step()` and `run()` share one total iteration budget and the
[shared stopping rules](stopping.md). An iteration is an attempted finite trial
residual, including failed callbacks. Finite unchanged candidates are evaluated
before deciding residual success or stagnation. Accepted finite candidates clear
the Jacobian and Hessian; stopping and budget checks precede further derivative
work. A failed trial preserves the previous finite point, residual, derivatives
and metrics, while updating attempts and diagnostics.

`result` is `NewtonMethodsReturnType`. Its `root` and `f_root` are tuples;
`df_root` is `(jacobian, hessian)` with nested tuples. `f_calls` is `(F, J, H)`:
every derivative-plan invocation counts once in its role, while all stencil
residual attempts also count in \(F\). Failed attempts count too. Unavailable
derivatives are NaN. Jacobian failure clears both derivative roles; Hessian
failure preserves a valid Jacobian and clears the Hessian.

`VectorChebyshevStep` contains `root`, `f_root`, `df_root`, `iters`, `f_calls`,
`precision`, `error`, `termination_reason`, `callback_code`, `callback_stage`
and `linear_status`, with appended `linear_stage` (`None`, `'jacobian'` or
`'secondary'`). The shared `VectorHigherOrderStep` alias names the same tuple
class. This extends the preview tuple length and native record layout; compiled
consumers must rebuild against current declarations. Chebyshev uses only the
Jacobian linear stage. The read-only `method` property returns `'chebyshev'`.
Snapshots and history exports are immutable copies and
survive further steps or solver destruction. History uses the
[native initial/sampled/terminal recording rules](native-core.md#stepping-and-history).
Properties `result`, `done`, `history`, `history_truncated`, `callback_code`,
`callback_stage`, `linear_status` and `linear_stage` expose state without advancing it.

Native nonzero callback statuses raise `NativeCallbackError` with exact
`native_status`, copied result and stage `'function'`, `'jacobian'` or
`'hessian'`, including residual samples used by numerical derivative plans.
Python exceptions retain identity and traceback. The solver becomes terminal;
after the failing call delivers the error, later `step()`/`run()` calls inspect
the failed state passively. Every callback output is seeded NaN, so unwritten
values cannot silently become accepted data.

All explicit adapters are inspected before any reservation is acquired. They
remain reserved across dimension/options/cache coercion, allocation, numerical
execution, boxing and exception transfer. Reentrant or concurrent access to the
same active solver or adapter raises an error. Independent callbacks can share
immutable plans with independent workspaces. Producers sharing mutable native
contexts must synchronize those contexts themselves.

## Storage budgets

For dimension \(n\), define \(w\) as the larger of \(5n\) for a numerical
Jacobian and \(2n^2+10n\) for a numerical Hessian; omit a term when that role
is analytic. The solver preallocates
\(8n+2n^2+n^3+w\) doubles plus \(n\) C-int pivots. The workspace budget also
counts both plans' explicit step storage, even when those plans are shared.
History separately costs \(2n+n^2+n^3\) doubles plus native record metadata
per requested slot. Checked integer products and native capacities apply before
allocation. Python owners/exports and provider-internal allocations are outside
these byte budgets. All cy-root numerical workspace/history is allocated once;
there are no per-step cy-root allocations.

## Compiled Hessian callbacks

Installed `tensor_callbacks.pxd` exposes
`make_hessian_callback(function, dimension, context, keepalive)` with signature
`int callback(const double* x, size_t n, double* out, void* context) noexcept nogil`.
Write all \(n^3\) entries in residual-first column-major order:
`out[o + n*(j + n*k)]`. Input is borrowed only during the call; do not mutate
or retain its pointer. Return zero on success, exact nonzero status on failure.
Do not throw Python or C++ exceptions through this ABI. The adapter pins its
producer-supplied keepalive for its lifetime.

C/C++ producers may supply `cyroot_hessian_descriptor` from `callbacks.h` in a
capsule named `cyroot.hessian_callback.v1` and use
`HessianCallback.from_capsule(capsule, keepalive=owner)`. The adapter validates
name, version, minimum size, dimension and nonnull function, then copies the
function/context fields and retains owners. A descriptor name cannot prove
arbitrary pointer validity. This is a separate tensor ABI; existing
`cyroot.vector_callback.v1` residual/Jacobian capsules retain their meanings.

With native callbacks for every evaluated role, including the residual sampled
by numerical plans, execution stays outside the GIL. Python callbacks reacquire
it per evaluation; setup and result/history export also use it. The
[compiled example](../examples/native_vector_chebyshev.md) uses installed
headers/declarations without NumPy C headers. The package still requires NumPy
and SciPy; optional providers and remaining vector methods are later stages.
