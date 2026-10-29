# Rectangular Callbacks

`RectangularCallback` adapts functions with `n` inputs and `m` outputs,
separately from the square `VectorCallback`. Every solver that handles
rectangular systems accepts it for residuals and Jacobians, including with
native numerical differences, and Robinson and Traub–Steffensen accept it when
`n = m`. Broyden, Klement and Barnes need a square `VectorCallback`. Plain
Python callbacks need neither an adapter nor explicit dimensions; see
[Vector Newton](native-vector.md).

```python
from cyroot.native import RectangularCallback

residual = RectangularCallback(lambda x: (x[0], x[1], x[0]+x[1]), 2, 3)
assert residual.evaluate((2., 3.)) == (2., 3., 5.)
jacobian = RectangularCallback(lambda x: ((1., 0.), (0., 1.), (1., 1.)),
                               2, 3, jacobian=True)
assert jacobian.evaluate((2., 3.)) == ((1., 0.), (0., 1.), (1., 1.))
```

Python functions receive an immutable copy of the point as a tuple. Residuals
return a vector of length `m`, and Jacobians `m` rows of length `n`. Sized
sequences and buffers, including strided and read-only ones, are accepted if
their shapes match. Returned tuples are independent copies. Nonfinite values
are passed through for the solver to deal with.

`evaluate(point, *, workspace_bytes=64*1024*1024)` reserves the adapter before
converting anything and keeps it reserved through output packing and exception
transfer; reusing an active adapter raises `RuntimeError`. Python exceptions
propagate as the original instance. The budget covers `n + m` doubles for a
residual or `n + m*n` for a Jacobian, not counting Python objects or
allocations made by the callback. The read-only properties are
`input_dimension`, `output_dimension`, `jacobian` and `native`. Dimensions must
be positive and fit both the C `int` ABI and native storage.

## Compiled Functions

`cyroot.native.rectangular` declares the callback signature:

```cython
ctypedef int (*RectangularFunction)(const double* x, size_t n, size_t m,
                                  double* out, void* context) noexcept nogil
```

Use `make_rectangular_callback(function, n, m, context, keepalive=None,
jacobian=False)`. Residuals write `m` doubles; Jacobians write `m*n` doubles
in column-major order, `out[row + column*m]`. A zero status means the callback
finished; entries it did not write stay NaN. A nonzero status turns the whole
output into NaN, and Python `evaluate` raises `NativeCallbackError` with the
status and the stage `function` or `jacobian`. Native code must follow the
signature, keep exceptions from crossing the ABI and acquire the GIL before
touching Python.

In a native loop, reserve the adapter with `_enter()` while holding the GIL,
call `evaluate_raw(x, out)` without the GIL, raise any saved Python `_error`,
and release the adapter with `_release()` in a `finally` block. The caller
keeps the adapter alive and passes valid, non-overlapping input and output
spans of the declared sizes. This raw path allocates nothing; adapters around
Python callbacks acquire the GIL inside their bridge. The producer is
responsible for its context's lifetime and for synchronizing distinct adapters
that share native state.

`RectangularCallback.from_capsule(capsule, keepalive=owner)` accepts capsules
named `cyroot.rectangular_callback.v1`. `callbacks.h` declares their
`cyroot_rectangular_descriptor`, with size and version, input and output
dimensions, kind (`CYROOT_RECTANGULAR_RESIDUAL` or
`CYROOT_RECTANGULAR_JACOBIAN`), function and context. The descriptor is
validated and copied, and the supplied owners are kept alive.
