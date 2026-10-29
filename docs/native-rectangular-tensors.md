# Analytic Rectangular Hessians And Native Contractions

`RectangularHessianCallback` adapts a supplied Hessian of
\(F:\mathbb R^n\to\mathbb R^m\). Its copied tensor shape is \((m,n,n)\),
matching the [rectangular numerical plan](native-rectangular-hessian.md).
Analytic entries are preserved without imposing symmetry on either input axis.

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
The adapter has a distinct Hessian role; it is not a callable residual/Jacobian
provider. The [Chebyshev path](native-rectangular-chebyshev.md) now supports rectangles;
fixed [Halley/Super-Halley](native-rectangular-halley.md) also support rectangles,
and [general modified Halley](native-rectangular-modified.md) uses its separate
three-solve rectangular correction.

`evaluate(point, *, workspace_bytes=64*1024*1024)` copies the point, invokes the
callback and returns a nested tuple tensor. Python callbacks receive a tuple of
length \(n\) and return an indexable real tensor with shape \((m,n,n)\).
Negative-stride arrays and read-only native double point buffers are supported.
The result owns its values independently of the input tensor. Complex entries,
text and malformed tensor shapes fail conversion. This is a callback adapter,
not a numerical approximation or convergence check.

Evaluation requires \(n+mn^2\) native doubles, checked against `workspace_bytes`
before allocation. Python owners, copied tuples and allocator overhead are
additional. The callback is reserved across budget/point conversions, invocation,
result export and exception creation. Reentry and reinitialization are rejected;
independent adapters can execute concurrently. Python callbacks use the GIL;
compiled callbacks execute without it.

Native outputs start as NaN. A nonzero callback status clears the entire tensor
and raises `NativeCallbackError` with the original integer code, stage `'hessian'`
and copied all-NaN tuple tensor as `.result`. Original Python exceptions preserve
identity and traceback and are cleared from the adapter when transferred. Code
zero is passed through even when entries remain NaN or are nonfinite: numerical
consumers must check the complete tensor, as they do for residual/Jacobian
adapters. The adapter does not invent error codes for partial native output.

## Compiled Callbacks And Capsules

Cython consumers import from `cyroot.native.tensor_callbacks`:

- `RectangularHessianCallback`, `RectangularHessianFunction` and
  `make_rectangular_hessian_callback(function, n, m, context, keepalive=None)`.
- `cyroot_rectangular_hessian_descriptor` for capsules, or the equivalent
  C declaration in `callbacks.h`.

The callback signature is
`int function(const double* x, size_t n, size_t m, double* out,
void* context) noexcept nogil`. Input is borrowed and read-only. Write every
one of the \(mn^2\) tensor entries at offset `o + m*(j + n*k)` on success.
The producer owns the context and supplies a keepalive owner. Callbacks must
not throw or use Python without explicitly acquiring the GIL.

The separate capsule name is `cyroot.rectangular_hessian_callback.v1`.
The descriptor contains `struct_size`, `abi_version`, `input_dimension`,
`output_dimension`, `function` and `context`. Macros
`CYROOT_RECTANGULAR_HESSIAN_CALLBACK_ABI` and
`CYROOT_RECTANGULAR_HESSIAN_CAPSULE_NAME` identify it in `callbacks.h`.
`RectangularHessianCallback.from_capsule(capsule, keepalive=None)` validates the
name, descriptor size, version, dimensions and nonnull function. Both the capsule
and supplied owner are pinned. Existing square Hessian and rectangular
residual/Jacobian capsule names and layouts are unchanged.

Trusted Cython loops reserve the adapter with `_enter()`, pin its lifetime and
call `evaluate_raw(x, out)` on disjoint valid `x[n]`, `out[m*n*n]` spans, then
transfer any saved original `_error` before `_release()` in a `finally` block.
Reservation and Python exception transfer require the GIL; `evaluate_raw` is
`noexcept nogil`, allocates nothing and preserves native statuses. Native loops
can reuse fixed caller-owned buffers. See the installed consumer test for the
complete reservation/error-transfer pattern. Rebuild consumers using these 2.0
declarations.

## Tensor Contractions

Allocation-free helpers in `tensor_ops.hpp` and `tensor_ops.pxd` take separate
input/output dimensions:

| Helper | Operation | Output span/layout |
| --- | --- | --- |
| `contract_rectangular_hessian(H,s,n,m,out)` | \(\sum_{j,k}H_{o,j,k}s_js_k\) | `out[m]` |
| `contract_rectangular_hessian_direction(H,s,n,m,out)` | \(\sum_jH_{o,j,k}s_j\) | `out[m*n]`, `o+m*k` |
| `rectangular_matrix_vector(A,s,n,m,out)` | \(\sum_jA_{o,j}s_j\) | `out[m]` |

Tensor spans are `H[m*n*n]`, direction spans `s[n]`, and matrices are
column-major `A[m*n]`. All spans must be valid, disjoint and live for the call;
the caller owns synchronization and storage. All ordered input pairs are used,
without a symmetry assumption. Old square helper names/signatures remain thin
wrappers using \(m=n\).

Return status zero means success; one means nonfinite input or final result and
poisons the complete output; two means invalid dimensions or span-size overflow
and leaves spans untouched. Every supplied tensor/matrix entry is validated even
when its direction weight is zero. Exponent-scaled compensated arithmetic
preserves the existing square range handling; it extends range, not precision,
and gives no guarantee against derivative noise or cancellation. No allocation,
Python or GIL operation occurs in these helpers.

These interfaces provide tensor plumbing. The [rectangular Chebyshev extension](native-rectangular-chebyshev.md) specifies
its least-squares policy explicitly. The fixed [Halley/Super-Halley extensions](native-rectangular-halley.md) are
also specified; other rectangular methods still need separate mathematical review. SciPy and NumPy remain package dependencies.
