# Native vector Newton and established API

`cyroot.native.VectorNewton` solves nonempty real systems with an analytic
Jacobian or a [native finite-difference plan](native-jacobian.md). Pass `None`
for a default central plan. It uses private native workspaces and SciPy's Cython LAPACK
provider. Neither its numerical loop nor its storage uses NumPy operations.
The package still requires NumPy and SciPy; the backend is not yet optional.

```python
from cyroot.native import VectorNewton

def residual(x):
    return (x[0]**2 + x[1] - 3., x[0] + 2*x[1] - 3.)

def jacobian(x):
    return ((2*x[0], 1.), (1., 2.))

solver = VectorNewton(residual, jacobian, (1.2, .7), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

The solution near this starting point is `(1.5, .75)`.
`vector_newton(function, jacobian, x0, **options)` runs a complete solve.
The established `cyroot.generalized_newton` entry point also uses this engine.
The square higher-order methods have their own
[breaking migration guide](vector-higher-order-migration.md).
[Traub–Steffensen](native-steffensen.md) also reuses this state and square LU,
with residual-dependent divided differences instead of an analytic Jacobian or
ordinary finite-difference plan.
[Robinson](native-robinson.md) reuses the same state with two initial points,
deterministic probe directions and reconstruction from actual rounded secants.

## Migrating `generalized_newton`

The public signature is
`generalized_newton(F, J, x0, F_x0=None, J_x0=None, *, etol=1e-10,
ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200, history=0,
history_every=1, history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`,
where `eps` is binary64 machine epsilon. `J` remains a required argument; pass
`None` for automatic central differences, an initialized matching `JacobianPlan`
(square) or `RectangularJacobianPlan` for configured differences, or an analytic
Python/native Jacobian callback.

```python
from cyroot import generalized_newton
from cyroot.native import JacobianPlan

def residual(point):
    x, y = point
    return (x*x + y - 3., x + 2*y - 3.)

result = generalized_newton(residual, JacobianPlan(2, scheme='central'),
                            (1.2, .7))
assert result.success
assert max(abs(a-b) for a, b in zip(result.root, (1.5, .75))) < 1e-9
```

The top-level export, `cyroot.vector_newton.generalized_newton` and
`find_vector_root('generalized_newton', ...)` share this behavior. Options after
`J_x0` are keyword-only. The default `max_iter` is 200; zero or `None` is unlimited. The old `h`
option is removed: configure a native plan's `step` instead. The 1.x derivative
approximation classes were removed; see [finite differences](finite-differences.md).

Python residual and Jacobian callbacks now receive immutable tuples. Replace
array expressions such as `point - 1.` with indexed sequence arithmetic or an
explicit conversion inside your callback. Inputs, cache shapes and providers
are validated even at an initially converged root. Square linear systems use LU;
singular square systems return unsuccessful `breakdown`, with no implicit
pseudoinverse fallback. Rectangular systems use rank-aware least-squares steps.

For \(F:\mathbb R^n\to\mathbb R^m\), the Jacobian has \(m\) rows and
\(n\) columns. Ordinary Python functions need no dimension arguments: the
point determines \(n\), and the cached or first residual determines \(m\).
The first residual is reused rather than evaluated twice. For example:

```python
from cyroot import generalized_newton

def tall(point):
    x, y = point
    return (x-1., y-2., x+y-3.)

result = generalized_newton(tall, None, (0., 0.))
assert result.success
assert max(abs(a-b) for a, b in zip(result.root, (1., 2.))) < 1e-9
```

An initial nonfinite point skips callbacks. With no cached residual or explicit
provider dimensions, its unsuccessful result uses \(m=n\) placeholders because
the output shape cannot be discovered. A failed first callback also retains an
inspectable terminal state with these placeholders if its shape is unknown.
Explicit adapters/plans constrain the
shape even at an initial root. Arbitrary higher-dimensional residual tensors
are not accepted: points and residuals are vectors, Jacobians are matrices.

The result remains `NewtonMethodsReturnType`, with tuple `root`, `f_root` and
Jacobian rows in `df_root`. The one-shot function returns the final result;
retain a `VectorNewton` state for `step()` and copied intermediate history.
Supplied `F_x0` and `J_x0` skip their initial evaluations and contribute zero
calls. Counts, terminal Jacobian placeholders and failure behavior follow the
native engine described below.

## Python boundary

Inputs and callback outputs accept sized, indexable sequences or direct
native-format double buffers. `x0` has length `n`, residuals length `m`; Jacobians
have shape `(m, n)`, with rows expressed as nested sequences. NumPy arrays,
`array.array`, memoryviews and `DoubleBuffer` work where their dimensions and
native double format match. Other buffer dtypes, indirect buffers and empty
systems are rejected. Read-only, negative-stride, zero-stride and unaligned
buffers are copied using their logical indices. User data is never overwritten.
Generators are not a supported substitute for a sized sequence.

Each Python callback receives an immutable copied tuple of the current point.
It can retain that tuple safely. It returns a fresh or reused residual/Jacobian
sequence or buffer; the bridge immediately packs the values into private native
storage. Python execution, conversion and buffer acquisition acquire the GIL.
Use native callbacks below for a numerical run that does not reacquire it.

`run()` and `result` return `NewtonMethodsReturnType`. Point and residual are
immutable tuples, and `df_root` is a tuple of Jacobian rows. `step()` and
`history` return immutable `VectorNewtonStep` records with the same numerical
fields plus callback stage/code, LAPACK `linear_status` and `linear_rank`.
The solver exposes `input_dimension`, `output_dimension`, `linear_status` and
`linear_rank`. Rank is the last rectangular solve's effective rank, or `None`
when unavailable, after a failed rectangular solve and for square LU. It describes that
linear step, not necessarily the Jacobian at a subsequently accepted point.

## Numerical behavior

For square systems, the step solves `J delta = -F` with pivoted LU (`dgesv`),
then evaluates the candidate point. A singular LU factorization
returns unsuccessful `breakdown`; the solver's `linear_status` and terminal
history retain LAPACK's nonzero `info`. This failure attempts no residual or
iteration and preserves the previous point, residual and Jacobian. Negative
`info` is also a backend failure. Zero means LAPACK reported success or no
factorization was needed. The solver additionally checks that all LU entries
are finite: elimination can overflow despite zero `info`, yielding a finite
incorrect direction. Such factors report `nonfinite` before any trial.

For rectangular systems, the same iteration uses native LAPACK `dgelsy` through
the [least-squares plan](native-least-squares.md), with `rcond=1e-12` and its
conservative maximum dimension of 32767. It minimizes the linear residual and
selects a minimum Euclidean norm correction when the linear solution is not
unique. Rank-deficient rectangular systems are allowed; square LU does not
switch backend when singular. A minimum-norm correction does not guarantee a
globally minimum-norm nonlinear root. LAPACK `info` and effective rank are
reported separately; failed solves do not attempt a candidate.

Only residual convergence reports root success. An inconsistent overdetermined
system may reach a least-squares stationary point with a nonzero residual;
that is unsuccessful stagnation or an iteration limit, not a root. No damping,
line search, regularization or global convergence guarantee is supplied.

Tolerance and total-budget arguments follow [native scalar Newton](native-newton.md)
and [stopping rules](stopping.md). Norms use maximum absolute components.
Relative residual tolerance uses the fixed initial residual norm; relative
step tolerance uses the maximum of consecutive iterate norms. A small step
without a small residual is unsuccessful stagnation. `max_iter=0` is unlimited.

`f_calls` is `(residual_calls, jacobian_calls)` for this solve. With a numerical
plan, residual counts include stencil samples and Jacobian counts include
approximation attempts. A complete central approximation samples `2*n`
residuals; forward/backward approximations sample `n`, reusing the current base
residual. The uncached initial base evaluation adds one residual call. Supplying
`f_x0` or `jac_x0` skips the corresponding initial callback; supplied values
must match their callbacks. Initial residual convergence skips an unused
Jacobian evaluation and finite-value check. Supplied cache shapes and formats
are still validated. Each attempted finite candidate residual is one iteration.
Nonfinite arithmetic or a failed factorization before that call adds no
iteration or residual call.

Failed/nonfinite candidate residuals preserve the previous finite point,
residual, Jacobian and metrics, with updated attempt counts. A finite candidate
residual is accepted before checking convergence, stagnation and budget. The
new Jacobian is evaluated only if another step is needed; otherwise `df_root`
contains NaNs. A failed/nonfinite Jacobian leaves the newly accepted point and
residual with an entirely NaN Jacobian. No partial or stale matrix is reported.

Python exceptions, including output shape/conversion errors, preserve their
identity and callback traceback. Native nonzero codes raise `NativeCallbackError`
with `native_status`, copied `result` and `callback_stage` (`'function'` or
`'jacobian'`). Numerical failures return ordinary unsuccessful results.

## Native callbacks

`vector.pxd` declares:

```cython
ctypedef int (*VectorFunction)(const double* x, size_t n,
                              double* out, void* context) noexcept nogil
```

The square callback ABI is preserved. Use `make_vector_callback(function, dimension, context, keepalive,
jacobian=False)` for residuals, and `jacobian=True` for Jacobians. Residuals
write `n` doubles. Jacobians write `n*n` doubles in **column-major** order:
element `(row, column)` is `out[row + column*n]`. Success must write every
element and return zero; output is seeded with NaNs to catch omitted writes.
Nonzero status stops without consuming partial output. Callbacks must not
throw Python/C++ exceptions across the ABI or mutate/retain the borrowed
input pointer. All dimensions/products must be represented safely by the producer.

`VectorCallback(function, dimension, jacobian=False)` creates a Python adapter.
Native capsules use `cyroot.vector_callback.v1` and the
`cyroot_vector_descriptor` in `callbacks.h`: version/size, dimension, kind
(`CYROOT_VECTOR_RESIDUAL` or `CYROOT_VECTOR_JACOBIAN`), function and context.
`VectorCallback.from_capsule(capsule, keepalive=owner)` validates and copies
the descriptor. Pointer provenance and actual context/code lifetime remain
the producer's responsibility. For rectangular systems, use
[`RectangularCallback`](native-rectangular-callbacks.md) and its distinct native
factory/capsule with explicit `n` and `m`; Jacobian layout is `row + column*m`.
These adapters also work for square systems. Both callback ABIs are unchanged;
the draft `VectorNewton` Cython object/record layout has changed, so compiled
consumers must rebuild. The Cython/C ABI is still a draft.

Selected explicit adapters are reserved together before setup performs arbitrary
argument conversion, including point/cache packing, tolerances, budgets and
history options. Reservations also cover step/run result construction, history
export, output conversion and exception transfer. Active adapter reuse and
concurrent/reentrant solver access are rejected. Adapter kind and dimension must
match their role; residuals require a residual adapter, and analytic
Jacobians require a Jacobian adapter. Square `VectorCallback` declares `m=n`;
it cannot be paired with a rectangular shape. Other native callback types and
uninitialized or mismatched plans are rejected, including at an initial root. Distinct adapters sharing native state require
producer synchronization. See the [compiled example](../examples/native_vector.md).

## Workspace and history limits

`workspace_bytes` defaults to 64 MiB. Before native allocation, the solver
checks dimension/products, LAPACK's C `int` limit and the workspace budget.
For analytic Jacobians, the workspace contains five vectors, two square matrices and `n` C-int pivots:
`8*(5*n + 2*n*n) + sizeof(int)*n` bytes. LAPACK gets private packed storage.
cy-root allocates no new native workspaces or history entries during the loop;
the backend library controls its own internal allocation/threading behavior.
The workspace budget does not include memory allocated inside that provider.
Numerical plans add `3*n+2*m` scratch doubles and their explicit step storage to
this budget, as detailed in [native Jacobians](native-jacobian.md).

For rectangles, native storage contains `3*n+3*m+m*n` doubles plus the
least-squares plan's scratch (`m*n + max(m,n) + lwork` doubles), and `n` C-int
pivots. Numerical differences add the scratch and steps above. The allocation-
free LAPACK query determines `lwork` before final budget validation. A temporary
`n`-double point buffer is freed before persistent workspace allocation. Python
boundary copies, callback/provider allocations and Python object overhead are
excluded from this native budget. When shape discovery is needed, the initial
callback runs after preliminary bounds checks but before full shape-dependent
workspace/history validation.

`history=0` disables recording. Otherwise history is preallocated and bounded
by `history_bytes` (default 8 MiB), including native metrics and copies of
the point, residual and entire Jacobian for each entry. `history_every` samples
iterations; initial and terminal states are eligible regardless of sampling.
A terminal failure at the same iteration updates its existing record. Full
storage sets `history_truncated` and drops later entries. Exported tuples
remain valid and unchanged after resuming or deleting the solver.

There is no native cancellation/signal check per iteration. Native callbacks
allow independent solves without the GIL; shared callback state needs producer
synchronization. Higher-order vector methods still require square systems.

[Native Wolfe–Bittner](native-wolfe-bittner.md) supports arbitrary nonempty
point pools and rectangular residuals through rank-aware affine interpolation,
with separately inspectable best and active states.
