# Jacobian Products

A Jacobian need not be stored to apply it to a direction. For a residual
\(F:\mathbb{R}^n\to\mathbb{R}^m\), a forward product maps
\(v\in\mathbb{R}^n\) to \(J(x)v\in\mathbb{R}^m\). Its transpose product
maps \(w\in\mathbb{R}^m\) to \(J(x)^\mathsf{T}w\in\mathbb{R}^n\).
[Newton–Krylov](native-newton-krylov.md) uses these operations instead of a dense
Jacobian. An analytic sparse or structured implementation can avoid both its
storage and much of its computational cost.

## Supplied Products

```text
JacobianVectorCallback(function, input_dimension, output_dimension)
JacobianTransposeCallback(function, input_dimension, output_dimension)
callback.evaluate(point, direction, *, workspace_bytes=64*1024*1024)
```

Both adapters are exported from `cyroot` and `cyroot.native`.
`input_dimension=n` and `output_dimension=m` describe the original residual in
both classes, even though the transpose maps a direction of length `m` to a
result of length `n`. Ordinary Python callbacks receive copied tuples and may
return numeric sequences or arrays. Adapter evaluation returns a copied tuple.

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

Dimensions must be positive and fit native capacities. Inputs must be finite.
Real numeric buffers are converted at the boundary; complex and boolean buffers
are rejected. Output shape is checked. A standalone adapter preserves nonfinite
callback output for its consumer to interpret; a solver treats it as failure.
Reentrant use of an active adapter is rejected.

## Numerical Directional Products

```text
JacobianVectorPlan(n, m, scheme='central', step=None, *,
                   memory_bytes=64*1024*1024)
plan.evaluate(function, x, direction, f_x=None)
```

The plan is immutable and holds no residual callback or mutable sampling
workspace. Its standalone estimate has `product`, `f_calls`, `status`, and
`callback_code`. Status is `ok`, `nonfinite`, `breakdown`, or `callback_failure`.
Python callback exceptions propagate; a native nonzero status raises
`NativeCallbackError` with the estimate attached.

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

A forward/backward product needs one shifted residual; central needs two.
An uncached base adds one call. Zero directions require no samples. `step` is a
positive absolute displacement along \(v/\|v\|_\infty\); automatic steps scale
with the point magnitude and binary64 precision. Rounded sample displacements
are used in the weights. Lost active coordinates or collapsed geometry cause
`breakdown` before sampling; numerical failures poison the product with NaNs.

Scratch uses `2*n+3*m` doubles; standalone native evaluation uses `4*n+5*m`
doubles. `workspace_bytes` reports scratch and `memory_bytes` the standalone
allocation budget, excluding Python objects and callback-owned storage.

There is no separate public numerical transpose plan. In rectangular
Newton–Krylov, selecting a numerical JVP plan creates a matching internal
transpose operation from the same coordinate columns. This costs `O(n)` residual
samples per product, unlike a standalone directional JVP. The distinction keeps
LSQR's operator and transpose consistent; see the solver guide.

## Cython And C/C++ Producers

`cyroot.native.jvp` declares `JacobianVectorFunction` and
`make_jacobian_vector_callback`. `cyroot.native.vjp` declares
`JacobianTransposeFunction` and `make_jacobian_transpose_callback`.
Both native function signatures are:

```cython
cdef int product(const double* x, const double* direction,
                 size_t n, size_t m, double* out, void* context) noexcept nogil
```

The forward callback reads `x[n]`, `direction[n]`, and writes all `out[m]`.
The transpose callback reads `x[n]`, `direction[m]`, and writes all `out[n]`.
Inputs are borrowed, const, and valid only during the call. Return zero only
after writing every output; return a nonzero status for failure. No exception
may cross the ABI. Acquire the GIL explicitly if accessing Python.

The C headers `jvp_callbacks.h` and `vjp_callbacks.h` define version-1 descriptors.
Capsule names are `cyroot.jacobian_vector_callback.v1` and
`cyroot.jacobian_transpose_callback.v1`; use the corresponding adapter's
`from_capsule(capsule, keepalive=None)`. Cython factories take
`(function, n, m, context, keepalive=None)` and pin that owner. Raw contexts
remain producer-owned. Separate adapters sharing mutable context require their
own synchronization. Callback outputs are poisoned before use, and a nonzero
status poisons the whole output. The [native ownership rules](native-core.md)
also apply to direct `evaluate_raw` callers.
