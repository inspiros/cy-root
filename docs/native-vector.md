# Native Vector Newton

`cyroot.native.VectorNewton` solves nonempty real systems with an analytic
Jacobian or a [native finite-difference plan](native-jacobian.md); pass `None`
for a default central plan. It uses private native workspaces and SciPy's
Cython LAPACK interface. Neither its numerical loop nor its storage uses NumPy.

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
`vector_newton(function, jacobian, x0, **options)` runs a solver to completion,
and the public `cyroot.generalized_newton` runs the same engine. The
higher-order vector methods have their own
[migration guide](vector-higher-order-migration.md).
[Traub–Steffensen](native-steffensen.md) reuses this state and the square LU
solve, with residual-based divided differences in place of a Jacobian.
[Robinson](native-robinson.md) also reuses it, with two starting points,
fixed probe directions and secants built from the actual rounded points.

## Public Entry Point: `generalized_newton`

The public signature is
`generalized_newton(F, J, x0, F_x0=None, J_x0=None, *, etol=1e-10,
ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200, history=0,
history_every=1, history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`,
where `eps` is the binary64 machine epsilon. `J` is required: pass `None` for
automatic central differences, a `JacobianPlan` (square) or
`RectangularJacobianPlan` of matching shape for configured differences, or an
analytic Python or native Jacobian callback.

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

`cyroot.vector_newton.generalized_newton` and
`find_vector_root('generalized_newton', ...)` are the same function. Options
after `J_x0` are keyword-only. The default `max_iter` is 200; zero or `None` is
unlimited. There is no `h` option: set the `step` of a native plan instead. The
1.x derivative approximation classes are gone; see
[Finite Differences](finite-differences.md).

Python residual and Jacobian callbacks receive immutable tuples, not NumPy
arrays. Replace array expressions such as `point - 1.` with indexed arithmetic,
or convert explicitly inside the callback. Inputs, cache shapes and providers
are validated even when the starting point is already a root. Square linear
systems use LU, and a singular one returns an unsuccessful `breakdown` rather
than falling back to a pseudoinverse. Rectangular systems take rank-aware
least-squares steps.

For \(F:\mathbb R^n\to\mathbb R^m\), the Jacobian has \(m\) rows and
\(n\) columns. Python functions need no dimension arguments: the point gives
\(n\), and the cached or first residual gives \(m\). That first residual is
reused, not evaluated twice. For example:

```python
from cyroot import generalized_newton

def tall(point):
    x, y = point
    return (x-1., y-2., x+y-3.)

result = generalized_newton(tall, None, (0., 0.))
assert result.success
assert max(abs(a-b) for a, b in zip(result.root, (1., 2.))) < 1e-9
```

A nonfinite starting point calls no callbacks. Without a cached residual or
explicit provider dimensions, the output shape cannot be discovered, so the
unsuccessful result uses \(m=n\) placeholders. The same placeholders appear in
the terminal state when the first callback fails before the shape is known.
Explicit adapters and plans fix the shape even when the starting point is a
root. Points and residuals must be vectors and Jacobians matrices; higher
dimensional arrays are rejected.

The result is a `NewtonMethodsReturnType`, with tuple `root` and `f_root` and
the Jacobian rows in `df_root`. The function returns only the final result;
keep a `VectorNewton` to call `step()` and read intermediate history. Supplied
`F_x0` and `J_x0` skip their initial evaluations and count as zero calls.
Counts, terminal Jacobian placeholders and failures behave as described for the
native engine below.

## Python Boundary

Inputs and callback outputs can be sized, indexable sequences or buffers.
`x0` has length `n` and residuals length `m`; Jacobians have shape `(m, n)`,
given as a sequence of rows. NumPy arrays, `array.array`, memoryviews and
`DoubleBuffer` work when their dimensions match. Native float64 buffers are
read directly; other real dtypes, such as integers or float32, are converted
element by element. Complex and boolean buffers, indirect buffers and empty
systems are rejected. Read-only, negative-stride, zero-stride and unaligned
buffers are copied by logical index. User data is never overwritten. Generators
are not accepted in place of a sized sequence.

Each Python callback receives an immutable tuple copy of the current point,
which it may keep. It returns a new or reused residual or Jacobian sequence or
buffer, whose values are packed into private native storage at once. Python
execution, conversion and buffer access hold the GIL; use the native callbacks
below for a numerical run that never takes it.

`run()` and `result` return `NewtonMethodsReturnType`. Point and residual are
immutable tuples, and `df_root` is a tuple of Jacobian rows. `step()` and
`history` return immutable `VectorNewtonStep` records with the same numerical
fields plus callback stage/code, LAPACK `linear_status` and `linear_rank`.
The solver exposes `input_dimension`, `output_dimension`, `linear_status` and
`linear_rank`. The rank is the effective rank of the last rectangular solve,
or `None` for square LU and after a failed rectangular solve. It describes that
linear step, not necessarily the Jacobian at the next accepted point.

## Numerical Behavior

For square systems, each step solves `J delta = -F` with pivoted LU (`dgesv`)
and then evaluates the candidate point. A singular factorization returns an
unsuccessful `breakdown`, and the solver's `linear_status` and terminal history
keep LAPACK's nonzero `info`. No residual is evaluated, no iteration is used,
and the previous point, residual and Jacobian are kept. A negative `info` is
also a backend failure; zero means success or that no factorization was needed.
The solver also checks that every LU entry is finite, because elimination can
overflow while LAPACK still reports success and then give a finite but wrong
direction. Such factors report `nonfinite` before any trial.

For rectangular systems, the same iteration uses LAPACK `dgelsy` through the
[least-squares plan](native-least-squares.md), with `rcond=1e-12` and a
conservative maximum dimension of 32767. It minimizes the linear residual and,
when the linear solution is not unique, takes the correction of smallest
Euclidean norm. Rank-deficient rectangular systems are allowed, but a singular
square system never switches to this solver. A minimum-norm correction does not
make the nonlinear root found the one of minimum norm. LAPACK `info` and the
effective rank are reported separately, and a failed solve evaluates no
candidate.

Only residual convergence counts as finding a root. An inconsistent
overdetermined system can reach a least-squares stationary point with a nonzero
residual; that ends in unsuccessful stagnation or the iteration limit, not a
root. There is no damping, line search or regularization, and no global
convergence guarantee.

Tolerances and the iteration budget work as for
[native scalar Newton](native-newton.md) and as described in
[Stopping And Results](stopping.md). Norms are maximum absolute components. The
relative residual tolerance uses the initial residual norm, and the relative
step tolerance the larger of two consecutive iterate norms. A small step
without a small residual is unsuccessful stagnation. `max_iter=0` is unlimited.

`f_calls` is `(residual_calls, jacobian_calls)` for this solve. With a
numerical plan, stencil samples count as residual calls and each approximation
as a Jacobian call. A central approximation samples `2*n` residuals; forward
and backward approximations sample `n` and reuse the current residual. An
uncached initial residual adds one call. Supplying `f_x0` or `jac_x0` skips
that initial callback, so the value must be what the callback would return. If
the starting point is already a root, the Jacobian is neither evaluated nor
checked for finiteness, though the shapes and formats of supplied caches are
still validated. Each attempt to evaluate the residual at a finite candidate is
one iteration; nonfinite arithmetic or a failed factorization before that call
uses no iteration and no residual call.

A failed or nonfinite candidate residual keeps the previous finite point,
residual, Jacobian and metrics, and updates only the counts. A finite candidate
residual is accepted before convergence, stagnation and the budget are checked.
The new Jacobian is evaluated only if another step is needed; otherwise
`df_root` contains NaNs. If the Jacobian at a newly accepted point fails or is
nonfinite, the point and residual stand and the Jacobian is entirely NaN; a
partial or stale matrix is never reported.

Python exceptions, including output shape and conversion errors, propagate as
the original instance with the callback's traceback. A nonzero native status
raises `NativeCallbackError` with `native_status`, a copied `result` and
`callback_stage` (`'function'` or `'jacobian'`). Numerical failures return
ordinary unsuccessful results.

## Native Callbacks

`vector.pxd` declares:

```cython
ctypedef int (*VectorFunction)(const double* x, size_t n,
                              double* out, void* context) noexcept nogil
```

This is the square callback ABI. Use `make_vector_callback(function,
dimension, context, keepalive, jacobian=False)` for residuals and
`jacobian=True` for Jacobians. Residuals write `n` doubles. Jacobians write
`n*n` doubles in **column-major** order: element `(row, column)` is
`out[row + column*n]`. A successful call writes every element and returns zero;
outputs are filled with NaN beforehand to expose missed writes. A nonzero
status stops the solve without using the partial output. Callbacks must not
let Python or C++ exceptions cross the ABI, and must neither modify nor keep
the borrowed input pointer. The producer must make sure its dimensions and
their products are representable.

`VectorCallback(function, dimension, jacobian=False)` creates a Python adapter.
Native capsules use `cyroot.vector_callback.v1` and the
`cyroot_vector_descriptor` in `callbacks.h`: version and size, dimension,
kind (`CYROOT_VECTOR_RESIDUAL` or `CYROOT_VECTOR_JACOBIAN`), function and
context. `VectorCallback.from_capsule(capsule, keepalive=owner)` validates and
copies the descriptor; the producer remains responsible for the pointers and
for keeping the context and code alive. For rectangular systems, use
[`RectangularCallback`](native-rectangular-callbacks.md), with its own factory
and capsule and explicit `n` and `m`; its Jacobian layout is `row + column*m`.
It also works for square systems. Compiled consumers built against 1.x must be
rebuilt against the 2.0 declarations and versioned callback headers.

All explicit adapters are reserved together before setup converts any
argument, including the point, caches, tolerances, budgets and history
options. The reservation also covers result construction, history export,
output conversion and exception transfer. Reusing an active adapter, or
concurrent or reentrant use of the solver, is rejected. Each adapter's kind and
dimension must match its role: the residual needs a residual adapter and an
analytic Jacobian a Jacobian adapter. A square `VectorCallback` declares `m=n`
and cannot be used with a rectangular shape. Other native callback types and
uninitialized or mismatched plans are rejected, even when the starting point is
a root. Distinct adapters that share native state need synchronization by
their owner. See the [compiled example](../examples/native_vector.md).

## Workspace And History Limits

`workspace_bytes` defaults to 64 MiB. Before allocating, the solver checks
the dimensions and their products, LAPACK's C `int` limit and the workspace
budget. With an analytic Jacobian, a square workspace holds five vectors, two
square matrices and `n` C-int pivots: `8*(5*n + 2*n*n) + sizeof(int)*n` bytes.
LAPACK works on private packed copies. CyRoot allocates no native workspace or
history during the loop; the LAPACK library manages its own internal memory
and threads, which the budget does not include. Numerical plans add `3*n+2*m`
scratch doubles plus their explicit steps, as detailed in
[native Jacobians](native-jacobian.md).

For rectangular systems, native storage holds `3*n+3*m+m*n` doubles, the
least-squares plan's scratch (`m*n + max(m,n) + lwork` doubles) and `n` C-int
pivots, plus any numerical-difference scratch and steps as above. `lwork` comes
from a LAPACK workspace query, which allocates nothing, before the final budget
check. A temporary `n`-double point buffer is freed before the persistent
workspace is allocated. Copies at the Python boundary, allocations inside
callbacks and Python object overhead are not counted. When the output shape has
to be discovered, the first callback runs after preliminary bounds checks but
before the full workspace and history checks.

`history=0` disables recording. Otherwise history is allocated up front and
bounded by `history_bytes` (default 8 MiB), counting the metrics and copies of
the point, residual and whole Jacobian for every entry. `history_every` samples
iterations; initial and terminal states are always eligible. A terminal failure
in the same iteration updates its existing record. When storage is full,
`history_truncated` is set and later entries are dropped. Exported tuples stay
valid and unchanged after the solver resumes or is deleted.

Native solves cannot be cancelled and do not check for signals. With native
callbacks, independent solves can run without the GIL; shared callback state
needs synchronization by its owner.

[Native Wolfe–Bittner](native-wolfe-bittner.md) accepts arbitrary nonempty
point pools and rectangular residuals through rank-aware affine interpolation,
and reports its best and active states separately.
