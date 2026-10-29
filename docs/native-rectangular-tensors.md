# Analytic Rectangular Hessians And Native Contractions

`RectangularHessianCallback` adapts an analytic Hessian of
\(F:\mathbb R^n\to\mathbb R^m\). It returns a copied tensor of shape
\((m,n,n)\), like the [rectangular numerical plan](native-rectangular-hessian.md),
and keeps every entry as given, without imposing symmetry.

```python
from cyroot.native import RectangularHessianCallback

# This illustrative tensor is asymmetric: the adapter preserves every entry.
tensor = (((1., 2.), (3., 4.)),
          ((5., 6.), (7., 8.)),
          ((9., 10.), (11., 12.)))
callback = RectangularHessianCallback(lambda x: tensor, 2, 3)
result = callback.evaluate((1., 2.))
assert callback.shape == (3, 2, 2)
assert result == tensor
assert result[0][0][1] != result[0][1][0]
assert not callback.native
```

Construct with `RectangularHessianCallback(function, input_dimension,
output_dimension)`. Both dimensions must be positive integers fitting C `int`;
the complete input and tensor storage must fit native capacities. The read-only
properties are `input_dimension`, `output_dimension`, `shape` and `native`.
The adapter serves only as a Hessian, never as a residual or Jacobian. The
[Chebyshev](native-rectangular-chebyshev.md) and fixed
[Halley and Super-Halley](native-rectangular-halley.md) methods use it for
rectangular systems, and the
[general modified Halley](native-rectangular-modified.md) method uses it in its
three-solve rectangular correction.

`evaluate(point, *, workspace_bytes=64*1024*1024)` copies the point, calls the
callback and returns the tensor as nested tuples. Python callbacks receive a
tuple of length \(n\) and return an indexable real tensor of shape \((m,n,n)\).
Points can be negative-stride arrays or read-only float64 buffers. The result is
independent of the tensor the callback returned. Complex entries, text and
malformed shapes fail conversion. The adapter only evaluates the callback; it
neither approximates nor checks anything numerically.

Evaluation needs \(n+mn^2\) native doubles, checked against `workspace_bytes`
before allocation; Python owners, copied tuples and allocator overhead come on
top. The callback is reserved through the budget and point conversions, the
call, result export and exception creation. Reentry and reinitialization are
rejected, and independent adapters can run concurrently. Python callbacks take
the GIL; compiled callbacks run without it.

Native outputs start as NaN. A nonzero status makes the whole tensor NaN and
raises `NativeCallbackError` with the original integer code, the stage
`'hessian'` and the all-NaN tensor as `.result`. Python exceptions propagate as
the original instance with its traceback, and are cleared from the adapter once
raised. A zero status is passed through even if entries are still NaN or are
otherwise nonfinite: as with residual and Jacobian adapters, the consumer must
check the whole tensor. The adapter never invents error codes for partial
native output.

## Compiled Callbacks And Capsules

Cython consumers import from `cyroot.native.tensor_callbacks`:

- `RectangularHessianCallback`, `RectangularHessianFunction` and
  `make_rectangular_hessian_callback(function, n, m, context, keepalive=None)`.
- `cyroot_rectangular_hessian_descriptor` for capsules, or the equivalent
  C declaration in `callbacks.h`.

The callback signature is
`int function(const double* x, size_t n, size_t m, double* out,
void* context) noexcept nogil`. The input is borrowed and read-only. On success,
write all \(mn^2\) entries, entry \((o,j,k)\) at offset `o + m*(j + n*k)`. The
producer owns the context and supplies an owner object to keep it alive.
Callbacks must not throw, and must acquire the GIL before using Python.

The capsule name is `cyroot.rectangular_hessian_callback.v1`. The descriptor
contains `struct_size`, `abi_version`, `input_dimension`, `output_dimension`,
`function` and `context`, and the macros
`CYROOT_RECTANGULAR_HESSIAN_CALLBACK_ABI` and
`CYROOT_RECTANGULAR_HESSIAN_CAPSULE_NAME` in `callbacks.h` identify it.
`RectangularHessianCallback.from_capsule(capsule, keepalive=None)` checks the
name, descriptor size, version, dimensions and that the function is not null,
and keeps both the capsule and the owner alive.

A Cython loop reserves the adapter with `_enter()`, keeps it alive, and calls
`evaluate_raw(x, out)` on valid, non-overlapping `x[n]` and `out[m*n*n]` spans;
it then raises any saved `_error` before calling `_release()` in a `finally`
block. Reserving and raising need the GIL; `evaluate_raw` is `noexcept nogil`,
allocates nothing and returns the native status, so a loop can reuse fixed
buffers. The installed consumer test shows the complete reserve and
error-transfer pattern.

## Tensor Contractions

Allocation-free helpers in `tensor_ops.hpp` and `tensor_ops.pxd` take separate
input/output dimensions:

| Helper | Operation | Output span/layout |
| --- | --- | --- |
| `contract_rectangular_hessian(H,s,n,m,out)` | \(\sum_{j,k}H_{o,j,k}s_js_k\) | `out[m]` |
| `contract_rectangular_hessian_direction(H,s,n,m,out)` | \(\sum_jH_{o,j,k}s_j\) | `out[m*n]`, `o+m*k` |
| `rectangular_matrix_vector(A,s,n,m,out)` | \(\sum_jA_{o,j}s_j\) | `out[m]` |

Tensor spans are `H[m*n*n]`, direction spans `s[n]`, and matrices are
column-major `A[m*n]`. All spans must be valid, non-overlapping and alive for
the call, and the caller is responsible for storage and synchronization. Every
ordered pair of inputs is used, with no symmetry assumed. The square helpers
are thin wrappers with \(m=n\).

Status zero is success. One is a nonfinite input or result and makes the whole
output NaN. Two is an invalid dimension or span-size overflow and leaves the
spans untouched. Every tensor or matrix entry is validated, even where its
direction weight is zero. Exponent-scaled compensated arithmetic extends the
range, as for square systems, but not the precision, and does not protect
against derivative noise or cancellation. The helpers never allocate, call
Python or touch the GIL.

These helpers are building blocks. The solvers that use them are described in
[rectangular Chebyshev](native-rectangular-chebyshev.md),
[rectangular Halley and Super-Halley](native-rectangular-halley.md) and
[rectangular modified Halley](native-rectangular-modified.md).
