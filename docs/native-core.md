# Native Core

CyRoot separates Python setup and inspection from the native numerical engine.
`cyroot.native` exposes resumable solver states, versioned callback adapters,
owned storage, and derivative/linear-algebra plans. Public `cyroot` functions
use these engines too. NumPy and SciPy remain dependencies; Python-level NumPy
operations are not required inside native numerical loops.

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

`step()` advances one outer iteration; `run()` finishes through the same state
machine. `result` copies the current result without advancing it. Iteration
budgets include earlier `step()` calls. A completed state is terminal: further
calls return its frozen result rather than restarting with a new budget.
Iterating over a solver yields its remaining steps. Running snapshots use
`termination_reason=None`; some public result types use `unspecified` before
termination. Inspect `done` and the explicit reason.

Histories are bounded and opt-in. `history` sets the maximum snapshot count;
`history_every` samples iteration indices. Initial and terminal states are
eligible when capacity permits. A same-iteration terminal diagnostic can replace
the matching record. Once full, histories keep their existing entries and set
`history_truncated`. Exported tuples remain independent of later steps and of
solver destruction. Increase the capacity if the final iterate must be retained.

`as_root_snapshot` provides common single-root inspection while retaining the
original record in `diagnostics`. Solver-specific counters, ranks, callback
stages, and failed-state fields are documented on the method pages. Hybisect
also has a multi-root search/event interface. See [Stopping And Results](stopping.md).

## Python And Native Callbacks

An ordinary Python callback is supported through an adapter. Its call and
result conversion acquire the GIL. A compiled native callback uses a raw
status/output signature; the solver can run its numerical loop without it.
For scalar residuals the signature is:

```cython
cdef int residual(double x, double* out, void* context) noexcept nogil:
    out[0] = x*x - 2.
    return 0
```

Use `make_scalar_callback(function, context, keepalive)` from
`cyroot.native.scalar` to wrap that function. The adapter pins `keepalive`, which
must preserve the context and any code/data it refers to. The context remains
producer-owned. The [compiled scalar example](../examples/native_scalar.md)
shows a Cython object owning a parameter struct. The
[pure Python syntax example](compiling-callbacks.md) uses Cython annotations and
a native bridge in a `.py` module.

Native callbacks must write every required output before returning zero.
Return a nonzero status for failure. Do not let Python or C++ exceptions cross
the ABI. Any Python access needs an explicit GIL acquisition. A `nogil`
declaration permits native execution without the GIL; it does not itself
release the GIL or establish thread safety.

Different roles use different descriptors and layouts:

| Role | Guide |
| --- | --- |
| Scalar residual or derivative | [Newton](native-newton.md) |
| Square/rectangular residual and Jacobian | [Vector Newton](native-vector.md), [Rectangular Callbacks](native-rectangular-callbacks.md) |
| Residual Hessian tensors | [Tensor Callbacks](native-rectangular-tensors.md) |
| Jacobian forward/transpose products | [Jacobian Products](jacobian-vector-products.md) |
| Complex residual | [Scalar Migration](scalar-native-migration.md), [Numerical Derivatives](numerical-derivatives.md) |
| Interval enclosure for a root search | [Scalar Migration](scalar-native-migration.md#hybisect-interval-search) |

Capsule names include an ABI version. Adapters validate descriptor sizes, versions,
kinds, dimensions, and nonnull function pointers; they cannot validate arbitrary
pointer provenance or prove a producer's lifetime promise. Keep owners alive and
rebuild compiled consumers against the installed 2.0 headers and declarations.

## Ownership And Reentry

Explicit callback adapters stay reserved while a solver converts arguments,
executes native operations, transfers errors, and constructs its outputs.
Reentrant or concurrent use of an active state/adapter is rejected. Separate
adapters that point at the same mutable native context still need producer-side
synchronization. Construction and inspection are Python operations; independent
states with independent callbacks can execute their numerical loops concurrently.
There is no automatic parallel sample scheduler.

Immutable derivative and linear-algebra plans contain configuration rather than
shared mutable evaluation scratch. They may be reused by independent consumers
with disjoint buffers and appropriately synchronized contexts. A shared plan does
not make a shared active callback safe to reuse.

There are no Python signal checks inside a fully native run. Native callbacks
should have bounded execution and report failures through their status. Native
cancellation and observer callbacks are not provided.

## Callback Failures

Python exceptions preserve their original instance and traceback. Native nonzero
statuses raise `NativeCallbackError` with `native_status`, a copied `result`, and
`callback_stage`; some methods also expose a derivative order. Call counters
include attempted samples. Native callback outputs are poisoned before use so
partial writes cannot reuse old scratch.

A failed trial generally preserves the previous accepted point/residual. A
failed derivative following an accepted step retains that step and marks the
missing derivative invalid. Quasi-Newton update failures preserve the accepted
point and their last complete matrix. Read each method's failed-state contract
rather than interpreting an approximate matrix as a fresh exact derivative.

After the error has been delivered, terminal `step()` and `run()` calls return
the frozen state without resampling or raising it again. Inspection remains
available. Numerical breakdown and nonfinite arithmetic return unsuccessful
results according to the method's policy.

## Storage And Borrowed Views

`cyroot.native.DoubleBuffer` accepts direct native-format double buffers of rank
one or two. It borrows an exporter or makes private writable contiguous storage
with `copy=True`.

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

A borrowed guard pins the exporter and copies shape/stride metadata. Its view is
read-only through the guard, but the original owner may still mutate its data;
synchronize such writes. Signed and zero strides, read-only exporters, and
unaligned logical origins are supported. Indirect buffers and other formats or
ranks are rejected by this low-level buffer class. Solver boundaries additionally
convert real non-float64 buffers element by element and reject complex/bool buffers.

`buffers.pxd` exposes the C++17 `OwnedDoubleBuffer`, `VectorView`, and
`MatrixView`. Owners are movable and noncopyable. Views borrow pointers and
signed byte strides; raw constructors and span access are trusted compiled
interfaces. Dimension checks cannot establish actual pointer validity, allocation
length, nonoverlap, or lifetime. Keep the owner/guard alive and never resize or
reset storage while native access is active.

## Resource Budgets

Native storage is prepared before numerical stepping. Solvers validate dimension
products and expose byte budgets for workspace and history. Plans also bound
stencil nodes, tensor entries, or terms where work grows combinatorially.
The budget names and included arrays differ by method; they are not a limit on
process peak memory. Python snapshots, callback allocations, allocator overhead,
and backend-internal storage may be additional.

Use [Finite Differences](finite-differences.md) for derivative plan selection,
[Native Least Squares](native-least-squares.md) for DGELSY workspaces, and
[Newton–Krylov](native-newton-krylov.md) to avoid storing a dense Jacobian.
Compiled numerical kernels remove Python operations, not conditioning limits or
the inherent cost of dense high-order tensors.
