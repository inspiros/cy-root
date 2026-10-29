# Native Core

CyRoot separates Python setup and inspection from its native numerical
engine. `cyroot.native` exposes resumable solver states, versioned callback
adapters, owned storage, and derivative and linear-algebra plans, and the
public `cyroot` functions run on the same engines. The native numerical loops
make no Python-level NumPy calls.

## Step, Inspect, Resume

```python
from cyroot.native import Bisection

solver = Bisection(lambda x: x*x - 2., 0., 2., history=64)
first = solver.step()
answer = solver.run()
assert answer.success
assert solver.history[-1].root == answer.root
assert solver.done
```

`step()` advances one outer iteration, and `run()` finishes the solve through
the same state machine. `result` returns a copy of the current result without
advancing. The iteration budget includes earlier `step()` calls. A finished
solver stays finished: further calls return its final result instead of
starting again with a new budget. Iterating over a solver yields its remaining
steps. Snapshots taken while running have `termination_reason=None`, and some
public result types report `unspecified` until the solve ends; check `done`
and the reason.

Histories are opt-in and bounded. `history` sets the maximum number of
snapshots and `history_every` samples iterations. The initial and terminal
states are recorded if there is room, and a terminal diagnostic in the same
iteration can replace that iteration's record. Once full, a history keeps its
existing entries and sets `history_truncated`, so increase the capacity if you
need the final iterate. Exported tuples are unaffected by later steps and by
destroying the solver.

`as_root_snapshot` gives every single-root result the same fields and keeps the
original record in `diagnostics`. Solver-specific counters, ranks, callback
stages and failure states are described on the method pages, and hybisect also
has a multi-root search and event interface. See
[Stopping And Results](stopping.md).

## Python And Native Callbacks

A Python callback is wrapped in an adapter and takes the GIL for its call and
result conversion. A compiled native callback has a raw status-and-output
signature, which lets the solver run its numerical loop without the GIL. For a
scalar residual the signature is:

```cython
cdef int residual(double x, double* out, void* context) noexcept nogil:
    out[0] = x*x - 2.
    return 0
```

Wrap that function with `make_scalar_callback(function, context, keepalive)`
from `cyroot.native.scalar`. The adapter keeps `keepalive` alive, and
`keepalive` must in turn keep the context and any code or data it refers to
alive; the context itself stays owned by its producer. The
[compiled scalar example](../examples/native_scalar.md) uses a Cython object
that owns a parameter struct, and the
[pure Python syntax example](compiling-callbacks.md) uses Cython annotations and
a native bridge in a `.py` module.

A native callback must write every required output before returning zero, and
return a nonzero status on failure. Python and C++ exceptions must not cross
the ABI, and any Python access needs the GIL acquired explicitly. A `nogil`
declaration allows the function to run without the GIL; it neither releases
the GIL nor makes the function thread-safe.

Different roles use different descriptors and layouts:

| Role | Guide |
| --- | --- |
| Scalar residual or derivative | [Newton](native-newton.md) |
| Square/rectangular residual and Jacobian | [Vector Newton](native-vector.md), [Rectangular Callbacks](native-rectangular-callbacks.md) |
| Residual Hessian tensors | [Tensor Callbacks](native-rectangular-tensors.md) |
| Jacobian forward/transpose products | [Jacobian Products](jacobian-vector-products.md) |
| Complex residual | [Scalar Migration](scalar-native-migration.md), [Numerical Derivatives](numerical-derivatives.md) |
| Interval enclosure for a root search | [Scalar Migration](scalar-native-migration.md#hybisect-interval-search) |

Capsule names include an ABI version. Adapters check descriptor sizes,
versions, kinds, dimensions and that function pointers are not null; they
cannot check where a pointer came from or how long it stays valid. Keep owners
alive, and rebuild compiled consumers against the headers and declarations of
the installed CyRoot.

## Ownership And Reentry

Explicit callback adapters stay reserved while a solver converts arguments,
runs native code, transfers errors and builds its outputs. Reentrant or
concurrent use of an active solver or adapter is rejected. Separate adapters
that point at the same mutable native context still need synchronization by the
context's owner. Construction and inspection run in Python, but independent
solvers with independent callbacks can run their numerical loops concurrently.
Samples are never evaluated in parallel automatically.

Derivative and linear-algebra plans are immutable: they hold configuration, not
shared mutable scratch. Independent users can share a plan, each with its own
buffers and properly synchronized contexts. Sharing a plan does not make an
active callback safe to reuse.

A fully native run never checks for Python signals, so native callbacks should
finish in bounded time and report failures through their status. Native solves
cannot be cancelled, and there are no observer callbacks.

## Callback Failures

Python exceptions propagate as the original instance, with their traceback.
Nonzero native statuses raise `NativeCallbackError` with `native_status`, a
copied `result` and the `callback_stage`; some methods also report a derivative
order. Call counts include failed attempts. Native callback outputs are filled
with NaN before each call, so a partial write can never expose stale scratch.

A failed trial generally keeps the previously accepted point and residual. A
derivative that fails after an accepted step keeps that step and marks the
missing derivative as unavailable. A failed quasi-Newton update keeps the
accepted point and the last complete matrix. Each method page describes its
failure state; an approximate matrix is never a fresh derivative.

Once the error has been raised, later `step()` and `run()` calls return the
final state without sampling again or raising again, and the state can still
be inspected. Numerical breakdown and nonfinite arithmetic return unsuccessful
results as each method describes.

## Storage And Borrowed Views

`cyroot.native.DoubleBuffer` accepts direct, native-format float64 buffers
with one or two dimensions. It borrows the exporter's memory, or with
`copy=True` makes a private, writable, contiguous copy.

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

A borrowing guard keeps the exporter alive and copies its shape and strides.
The data is read-only through the guard, but its owner can still change it, so
synchronize such writes. Negative and zero strides, read-only exporters and
unaligned data are supported. This low-level class rejects indirect buffers and
other formats or dimensions; solver inputs, by contrast, convert real
non-float64 buffers element by element and reject complex and boolean ones.

`buffers.pxd` exposes the C++17 `OwnedDoubleBuffer`, `VectorView` and
`MatrixView`. Owners can be moved but not copied. Views borrow a pointer with
signed byte strides; their raw constructors and span access do no checking of
their own. Dimension checks cannot establish that a pointer is valid, long
enough, non-overlapping or alive, so keep the owner or guard alive and never
resize or reset the storage while native code is using it.

## Resource Budgets

Native storage is allocated before stepping starts. Solvers check dimension
products and have byte budgets for workspace and history, and plans also bound
stencil nodes, tensor entries or terms wherever the work grows
combinatorially. Budget names and what they cover vary by method, and they do
not limit the process's peak memory: Python snapshots, callback allocations,
allocator overhead and memory used inside LAPACK come on top.

See [Finite Differences](finite-differences.md) to choose a derivative plan,
[Native Least Squares](native-least-squares.md) for DGELSY workspaces, and
[Newton–Krylov](native-newton-krylov.md) to avoid storing a dense Jacobian.
Compiled kernels remove Python overhead, not conditioning limits or the
inherent cost of dense high-order tensors.
