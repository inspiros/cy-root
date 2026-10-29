# Rectangular Callbacks

`RectangularCallback` adapts functions with `n` inputs and `m` outputs. It is
separate from the existing square `VectorCallback`. `VectorNewton` and
`generalized_newton` accept this adapter for residuals and Jacobians, including
native numerical differences. Other vector solver families still require their
square adapters. Ordinary Python Newton callers need no adapter or explicit
dimensions; see the [Newton manual](native-vector.md).

```python
from cyroot.native import RectangularCallback

residual = RectangularCallback(lambda x: (x[0], x[1], x[0]+x[1]), 2, 3)
assert residual.evaluate((2., 3.)) == (2., 3., 5.)
jacobian = RectangularCallback(lambda x: ((1., 0.), (0., 1.), (1., 1.)),
                               2, 3, jacobian=True)
assert jacobian.evaluate((2., 3.)) == ((1., 0.), (0., 1.), (1., 1.))
```

Python functions receive a copied immutable point tuple. Residuals return a
vector of length `m`; Jacobians return `m` rows of length `n`. Sized sequences
and native double buffers are accepted, including strided and read-only buffers.
Shapes and buffer formats must match. Returned tuples are independent copies.
Nonfinite values are preserved for the eventual solver to classify.

`evaluate(point, *, workspace_bytes=64*1024*1024)` reserves the adapter before
conversion and keeps it reserved through output packing and exception transfer.
Reusing an active adapter raises `RuntimeError`. Python exceptions retain their
identity. The budget covers `n + m` doubles for residuals, or `n + m*n` for a
Jacobian; Python object storage and callback-owned allocations are excluded.
The read-only properties are `input_dimension`, `output_dimension`, `jacobian`
and `native`. Dimensions must be positive, fit the C-int ABI, and fit native
storage capacities.

## Compiled Functions

`cyroot.native.rectangular` declares the 2.0 callback signature:

```cython
ctypedef int (*RectangularFunction)(const double* x, size_t n, size_t m,
                                  double* out, void* context) noexcept nogil
```

Use `make_rectangular_callback(function, n, m, context, keepalive=None,
jacobian=False)`. Residuals write `m` doubles; Jacobians write `m*n` doubles
in column-major order, `out[row + column*m]`. Native zero status means the
callback completed; omitted writes remain NaN. Nonzero status poisons the entire
output and Python `evaluate` raises `NativeCallbackError` with its status and
`function`/`jacobian` stage. Native code must obey the signature, avoid exceptions
across the ABI, and acquire the GIL before using Python.

For a native loop, reserve with `_enter()` while holding the GIL, call
`evaluate_raw(x, out)` without the GIL, transfer any saved Python `_error`, then
release with `_leave()` in a `finally` block. The caller pins the adapter and
supplies valid disjoint input/output spans of the declared sizes. This raw path
allocates nothing; Python-backed callbacks explicitly acquire the GIL inside
the bridge. Context lifetime and synchronization across distinct adapters sharing
native state remain the producer's responsibility.

`RectangularCallback.from_capsule(capsule, keepalive=owner)` accepts
`cyroot.rectangular_callback.v1`. `callbacks.h` declares its separate
`cyroot_rectangular_descriptor` with size/version, input/output dimensions,
kind (`CYROOT_RECTANGULAR_RESIDUAL` or `CYROOT_RECTANGULAR_JACOBIAN`), function
and context. The descriptor is validated and copied; supplied owners stay pinned.
Existing scalar, square-vector and Hessian descriptors are unchanged.
