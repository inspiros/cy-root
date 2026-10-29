# Native core preview

`cyroot.native` exposes the new execution and ownership model. Its callback and
storage interfaces are drafts. Established Householder, Broyden,
`generalized_newton` and square higher-order vector entry points also use this
engine; see the [Newton manual](native-vector.md),
[Broyden manual](native-broyden.md) and
[vector migration guide](vector-higher-order-migration.md). Other solvers remain
available through `cyroot`; NumPy and SciPy remain package dependencies.
The new scalar loop and storage primitives do not use NumPy operations.

## Stepping and history

```python
from cyroot.native import Bisection

solver = Bisection(lambda x: x*x - 2., 0., 2., history=64)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

`step()` performs at most one bisection iteration and returns an immutable
`ScalarStep`. `run()` finishes through the same state machine. Iteration limits
apply to the whole solver, including earlier `step()` calls. Completed solvers
are idempotent; their state is not resumed with a new budget. Iterating over a
solver yields its remaining steps. `result` copies the current result without
advancing; inspect `done` to distinguish an unfinished solve from termination.
A running result uses `termination_reason='unspecified'`; a running step uses
`termination_reason=None`.

The [shared stopping rules](stopping.md) apply. Cached endpoint values count as
zero evaluations, move with their coordinates when endpoints are reversed,
and must match the supplied function. Evaluation counts belong to each solve,
even when several solvers use the same callback adapter. A failed callback
attempt counts once. An iteration counts an attempted midpoint evaluation;
trial failures preserve the previous finite root, residual and bracket.

History is disabled by default (`history=0`). A positive `history` preallocates
that many native records. It records the initial state, every `history_every`
iterations, and a terminal state even between sampling points when capacity
remains. Terminal failures record the retained state with updated counters and
reason. When full, recording stops and `history_truncated` becomes true; the
numerical solve continues. `history_bytes` defaults to 8 MiB and bounds the
requested native record allocation. Python exports add their own object
storage. History exports are copied immutable tuples and remain valid after
further steps or solver destruction.

## Python and native callbacks

Ordinary Python functions work through `ScalarCallback`, created automatically
when needed. The solver releases the GIL for native execution and the Python
adapter reacquires it around each function call and scalar conversion. Python
exceptions, including `BaseException` subclasses, propagate as the original
exception instance with the callback traceback.

Compiled Cython consumers can supply a native function:

```cython
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native import Bisection

cdef int residual(double x, double* out, void* context) noexcept nogil:
    out[0] = x*x - 2.
    return 0

def solve():
    callback = make_scalar_callback(residual, NULL, None)
    return Bisection(callback, 0., 2.).run()
```

Native callbacks write `out[0]` and return zero for success. A nonzero status
stops execution without consuming the output and raises `NativeCallbackError`.
The exception exposes `native_status` and a copied `result`; an existing solver
also exposes `callback_code` and its failed `result`. The reason is
`callback_failure`. A callback must not throw a C++ or Python exception through
this ABI. Returning a nonfinite scalar is numerical `nonfinite` termination.

`context` permits parameters without globals. The adapter retains the producer's
`keepalive` Python object for the entire adapter lifetime. That object must own
or pin the context allocation and any callback code/resources. Native callbacks
must acquire the GIL themselves if they use Python; declare and implement them
without Python calls to obtain uninterrupted native execution. A single adapter
cannot be used concurrently or reentrantly. Separate adapters that share a
context still require producer-managed synchronization. Solver mutation and
state reads are rejected while that solver is active.

The native run deliberately has no Python signal checks between steps. Callback
status is the present native failure mechanism; cancellation and observers are
future work. A long native callback must manage its own bounded execution.

The [compiled example](../examples/native_scalar.md) includes context ownership
and a build script. Extension builds use `get_include()` for native headers and
`get_cython_include()` for declaration lookup, including editable installs.
The example needs a C++17 compiler, Cython 3.3.0, setuptools and matching Python
headers, and no NumPy C headers.

A C/C++ producer may instead provide a capsule named
`cyroot.scalar_callback.v1`, pointing to `cyroot_scalar_descriptor` from
`callbacks.h`, and call `ScalarCallback.from_capsule(capsule, keepalive=owner)`.
The adapter validates its name, version, minimum size and nonnull function and
copies the descriptor fields. The producer is responsible for a valid in-process
pointer and context lifetime; a capsule name cannot validate arbitrary memory.

## Storage and borrowing

`cyroot.native.DoubleBuffer` accepts direct native-format double
buffers of rank one or two. It can borrow an `array.array`, memoryview or NumPy
array, or make private writable contiguous storage with `copy=True`.

```python
from array import array
from cyroot.native import DoubleBuffer

source = array('d', [1., 2., 3.])
borrowed = DoubleBuffer(source)
private = DoubleBuffer(source, copy=True)
private[0] = 7.
assert borrowed[0] == 1.
assert memoryview(borrowed).readonly
```

Borrowing pins the exporter through a managed Python memoryview, takes a copy
of shape/stride metadata, and exports read-only access. The original owner can
still mutate its storage; synchronize mutations during native reads. Negative
and zero byte strides and unaligned logical origins are supported. Indirect
buffers and other formats/ranks are rejected. Private copies own a fixed
allocation, release the original exporter, and may export writable buffers.

`buffers.pxd` provides the C++17 `OwnedDoubleBuffer`, `VectorView` and
`MatrixView` declarations. Owners are movable and noncopyable, allocate checked
zero-initialized storage and clean up without Python. Views copy metadata,
use signed byte strides, and never own or retain storage. Their raw-pointer
constructors require a valid backing allocation; `from_span` also checks the
supplied storage bounds. Arithmetic checks cannot establish provenance or
lifetime of an arbitrary pointer. Keep the owner or `DoubleBuffer` guard alive
until all native access has finished. Do not reset an owner while its views are
in use.

This preview supplies the primitives needed by the scalar and first vector ports. It
is not a general tensor library, parallel solver API, or dependency-free build.
[Native Newton](native-newton.md) now accepts analytic Python/native derivative
callbacks through the same interface. [Native vector Newton](native-vector.md)
uses an analytic Jacobian or native numerical plan and private SciPy LAPACK
workspaces. The established `generalized_newton` API forwards to this engine:
Python callbacks receive tuples, `J=None` selects central differences, and options
after `J_x0` are keyword-only. Its `h` option and legacy derivative providers are
removed, and required singular systems report breakdown without a pseudoinverse.
Selected explicit adapters remain reserved through argument conversion and
result/history export. Native finite-difference [Jacobian plans](native-jacobian.md) and
[scalar derivative plans](native-scalar-derivative.md) can use residual
callbacks without Python numerical operations. [Native secant](native-secant.md)
uses the scalar callback interface without derivatives. Higher-order Newton
methods in the [Halley family](native-higher-order.md), including Chebyshev,
modified Halley, Super-Halley and scalar tangent hyperbolas, accept analytic or
numerical first and [second derivatives](native-second-derivative.md).
[Runtime-order scalar differences and a Householder recurrence kernel](native-higher-derivative.md)
are integrated into [resumable Householder](native-householder.md).
[Native vector Hessian plans](native-hessian.md) provide square residual
second derivatives and feed [native vector Chebyshev](native-vector-chebyshev.md),
which accepts independent analytic/numerical Jacobian and Hessian providers.
[Native matrix Halley and Super-Halley](native-vector-halley.md) share that
state engine and add a secondary coupled matrix solve.
[Parameterized matrix modified Halley](native-vector-modified.md) also retains
the primary factorization for a final correction solve, with exact special paths.

Callback exceptions and native nonzero statuses are raised by the call that
performed the failing work. A callback failure makes the solver terminal.
After catching it, `run()` and `step()` return the unchanged failed result or
snapshot, with no new evaluations or repeated exception. `result`, `history`,
counts and callback diagnostics remain available for inspection. This applies
to every native solver in this preview.

[Native square good/bad Broyden](native-broyden.md) adds residual-only stepping
with explicit direct/inverse initial approximations, transactional secant updates
and copied approximation metadata. The established `cyroot.broyden` wrapper
now runs this engine, taking a Jacobian matrix or native plan through `J_x0`
for either variant. It defaults to bad Broyden, returns `BroydenResult`, uses
tuple callback inputs/results, and requires square systems. Its options after
`J_x0` are keyword-only and `h` is removed. Retain a native solver state for
intermediate steps and history; see the manual for initialization counts and
failure semantics.

[Native rank-aware least squares](native-least-squares.md) supplies a separate
immutable DGELSY plan for rectangular and rank-deficient linear systems.
Compiled callers can reuse their own workspaces without the GIL; Python
`solve` packs inputs and allocates each call. Rank uses LAPACK's estimated
triangular condition threshold, and a finite rank-deficient solution can be
successful. Existing square solvers retain their singular-system behavior;
linear success alone does not establish nonlinear root convergence.

[Rectangular callbacks](native-rectangular-callbacks.md) add separate input/output
dimensions for Python/native residuals and Jacobians. Native Newton accepts these
alongside the existing square callbacks; their descriptor ABIs remain separate.

[Rectangular Jacobian plans](native-jacobian.md#rectangular-jacobians) share the
square plan's native stencil and accept ordinary Python functions or compiled
rectangular residuals. `VectorNewton` and established `generalized_newton` now
infer ordinary Python residual dimensions and use native rank-aware rectangular
steps. Square Newton retains strict LU; higher-order and Broyden methods remain
square. Root success requires residual convergence.
SciPy remains the native LAPACK provider; optionalizing it is not a current priority.

[Native Klement](native-klement.md) reuses Broyden's square state, initialization
and history with a row-weighted secant update. The established `klement` entry
point now uses this engine; `h` is removed, options after `J_x0` are keyword-only,
and callbacks/results use tuples. Retain `Klement` for intermediate steps.

[Native Traub–Steffensen](native-steffensen.md) shares Newton's square state and
forward-difference stencil with signed residual-dependent steps, actual rounded
displacements, copied history and native callbacks. The established entry point
now uses this engine; zero/collapsed perturbations report breakdown without an
automatic-step fallback. Other quasi-Newton ports remain separate work.

[Native Barnes](native-barnes.md) reuses the direct Broyden state, LU,
initialization and history. Deterministic projection replaces its random
orthogonal direction, preserving actions on recent secants. The established
entry point removes `formula`/`h`, takes keyword-only options after `J_x0` and
returns `BroydenResult` tuple snapshots. Dependent updates report breakdown.

[Native Robinson](native-robinson.md) shares Newton's state with two initial
points, identity/Householder probes and native secant-matrix reconstruction.
It corrects the old direction orientation and rounded-displacement assumptions;
random mode is removed, options after `F_x1` are keyword-only and results use
Newton tuple snapshots and paired counts.

Wolfe–Bittner uses the existing native vector extension with bounded multipoint
buffers and a DGELSY affine solve plus an independent consistency check. All
established vector quasi-Newton routes now use native states. See
[its manual](native-wolfe-bittner.md) for shapes, best/active snapshots and counts.

[`RectangularHessianPlan`](native-rectangular-hessian.md) extends numerical
Hessian evaluation to `(m,n,n)` with independent input/residual dimensions,
sharing the existing rounded stencil and preserving square native interfaces.
Analytic [`RectangularHessianCallback`](native-rectangular-tensors.md) and
native tensor contractions also support independent dimensions. The shared [Chebyshev path](native-rectangular-chebyshev.md) now uses native
rank-aware corrections for rectangles. Fixed [Halley and Super-Halley](native-rectangular-halley.md)
also support rectangles. The [general modified-Halley path](native-rectangular-modified.md)
uses three rank-aware solves and retains exact alpha method selectors.


[`TensorDifferencePlan`](native-higher-mixed.md) adds standalone dense mixed
partial derivatives above order two for independent input/output dimensions.
It shares unique sample points and uses actual-rounded-node weights with native
compiled execution. Its byte and term budgets bound dense exponential growth.
Legacy derivative APIs and solver provider contracts are unchanged at this stage.


The established [vector finite-difference APIs](vector-derivative-migration.md)
now route all derivative orders through these native plans. NumPy arrays remain
at the compatibility boundary; explicit compiled callbacks avoid that per-sample
bridge. Native solver provider contracts remain unchanged.


Users can [compile typed Python-syntax callbacks](compiling-callbacks.md) with
Cython pure Python mode and export descriptors for the native loop. A compiled
Python wrapper passed as an ordinary callable still uses the GIL; the small
native bridge in the example bypasses that wrapper.


All 26 supported scalar entry points now dispatch to native state machines; see
[the scalar migration guide](scalar-native-migration.md) for the shared bracket
solver, Homeier, interpolation, complex Muller and interval Hybisect contracts.

[Vrahatis generalized bisection](native-vrahatis.md) now has a native solver,
bounded sign-polygon discovery and copied sweep history. It accepts square and
rectangular residual descriptors; multidimensional vertex signs alone do not
certify root existence. This completes native routes for supported vector solvers;
release/platform qualification remains separate work.
