# Jacobian Products

A Jacobian need not be stored to apply it to a direction. For a residual
\(F:\mathbb{R}^n\to\mathbb{R}^m\), a forward product maps
\(v\in\mathbb{R}^n\) to \(J(x)v\in\mathbb{R}^m\). Its transpose product
maps \(w\in\mathbb{R}^m\) to \(J(x)^\mathsf{T}w\in\mathbb{R}^n\).
[Newton–Krylov](native-newton-krylov.md) uses these operations instead of a dense
Jacobian. A product that exploits sparsity or structure can avoid both the
Jacobian's storage and much of its computational cost.

## Supplied Products

```text
JacobianVectorCallback(function, input_dimension, output_dimension)
JacobianTransposeCallback(function, input_dimension, output_dimension)
callback.evaluate(point, direction, *, workspace_bytes=64*1024*1024)
```

Both adapters are exported from `cyroot` and `cyroot.native`.
`input_dimension=n` and `output_dimension=m` describe the original residual in
both classes, even though the transpose maps a direction of length `m` to a
result of length `n`. Python callbacks receive copied tuples and may return
numeric sequences or arrays; `evaluate` returns a tuple.

```python
from cyroot import JacobianVectorCallback, JacobianTransposeCallback

# F(x) = (x0*x0 + x1, x0*x1, 2*x0 - x1)
jvp = JacobianVectorCallback(
    lambda x, v: (2*x[0]*v[0] + v[1], x[1]*v[0] + x[0]*v[1],
                  2*v[0] - v[1]), 2, 3,
)
vjp = JacobianTransposeCallback(
    lambda x, w: (2*x[0]*w[0] + x[1]*w[1] + 2*w[2],
                  w[0] + x[0]*w[1] - w[2]), 2, 3,
)
x, v, w = (1., 2.), (3., 4.), (5., 6., 7.)
assert jvp.evaluate(x, v) == (10., 10., 2.)
assert vjp.evaluate(x, w) == (36., 4.)
assert sum(a*b for a, b in zip(jvp.evaluate(x, v), w)) == sum(
    a*b for a, b in zip(v, vjp.evaluate(x, w))
)
```

Dimensions must be positive and fit native capacities, and inputs must be
finite. Real numeric buffers are converted at the boundary; complex and boolean
buffers are rejected. The output shape is checked. Called directly, an adapter
returns nonfinite callback output as it is, for the caller to interpret; a
solver treats it as a failure. Reentrant use of an active adapter is
rejected.

## Numerical Directional Products

```text
JacobianVectorPlan(n, m, scheme='central', step=None, *,
                   memory_bytes=64*1024*1024)
plan.evaluate(function, x, direction, f_x=None)
```

The plan is immutable and holds neither a residual callback nor any mutable
sampling workspace. `evaluate` returns an estimate with `product`, `f_calls`,
`status` and `callback_code`, where the status is `ok`, `nonfinite`,
`breakdown` or `callback_failure`. Python callback exceptions propagate; a
nonzero native status raises `NativeCallbackError` with the estimate attached.

```python
from cyroot import JacobianVectorPlan

f = lambda x: (x[0]*x[0] + x[1], x[0]*x[1], 2*x[0] - x[1])
plan = JacobianVectorPlan(2, 3)
estimate = plan.evaluate(f, (1., 2.), (3., 4.), f_x=f((1., 2.)))
assert estimate.status == 'ok'
assert estimate.f_calls == 2
assert max(abs(a-b) for a, b in zip(estimate.product, (10., 10., 2.))) < 1e-7
zero = plan.evaluate(f, (1., 2.), (0., 0.))
assert zero.product == (0., 0., 0.) and zero.f_calls == 0
```

A forward or backward product needs one shifted residual and a central
product two, plus one call for the base residual unless `f_x` is given. A zero
direction needs no samples. `step` is a positive absolute displacement along
\(v/\|v\|_\infty\); automatic steps scale with the point's magnitude and the
binary64 precision. The weights use the actual rounded displacements. If a
coordinate of the direction is lost to rounding or the sample geometry
collapses, the result is `breakdown` before any sampling; numerical failures
fill the product with NaN.

Scratch takes `2*n+3*m` doubles, and a standalone native evaluation `4*n+5*m`.
`workspace_bytes` reports the scratch and `memory_bytes` the standalone
allocation budget, neither counting Python objects or storage owned by
callbacks.

There is no public numerical transpose plan. In rectangular Newton–Krylov, a
numerical JVP plan comes with a matching internal transpose built from the same
coordinate columns. Each product then costs `O(n)` residual samples, unlike a
directional JVP, but keeps LSQR's operator and its transpose consistent; see
[Newton–Krylov](native-newton-krylov.md).

## Cython And C/C++ Producers

`cyroot.native.jvp` declares `JacobianVectorFunction` and
`make_jacobian_vector_callback`. `cyroot.native.vjp` declares
`JacobianTransposeFunction` and `make_jacobian_transpose_callback`.
Both native function signatures are:

```cython
cdef int product(const double* x, const double* direction,
                 size_t n, size_t m, double* out, void* context) noexcept nogil
```

The forward callback reads `x[n]` and `direction[n]` and writes all of
`out[m]`; the transpose callback reads `x[n]` and `direction[m]` and writes all
of `out[n]`. The inputs are borrowed, read-only and valid only during the call.
Return zero only after writing every output, and a nonzero status on failure.
No exception may cross the ABI; acquire the GIL explicitly before touching
Python objects.

The C headers `jvp_callbacks.h` and `vjp_callbacks.h` define version-1
descriptors, with capsule names `cyroot.jacobian_vector_callback.v1` and
`cyroot.jacobian_transpose_callback.v1`; pass them to the matching adapter's
`from_capsule(capsule, keepalive=None)`. The Cython factories take
`(function, n, m, context, keepalive=None)` and keep that owner alive. Raw
contexts stay owned by their producer, and separate adapters that share a
mutable context need their own synchronization. Outputs are filled with NaN
before each call, and a nonzero status makes the whole output NaN. The
[native ownership rules](native-core.md) also apply to direct `evaluate_raw`
callers.
