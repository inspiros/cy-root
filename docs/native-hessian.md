# Native Vector Hessians

`cyroot.native.HessianPlan` estimates the Hessian of a square residual
\(F:\mathbb R^n\to\mathbb R^n\) using Python or compiled residual callbacks.
For different input/output lengths, use
[`RectangularHessianPlan`](native-rectangular-hessian.md), which shares its stencil.
The numerical path uses native storage and arithmetic, with no NumPy
operations, Python objects or allocations between samples. Python setup,
callback bridges and result export hold the GIL.

```python
from cyroot.native import HessianPlan

plan = HessianPlan(2, step=.125)
estimate = plan.evaluate(lambda x: (x[0]**2 + 3*x[0]*x[1],
                                    2*x[1]**2 - x[0]*x[1]), (1., 2.))
assert estimate.status == 'ok'
assert estimate.hessian[0] == ((2., 3.), (3., 0.))
assert estimate.hessian[1] == ((0., -1.), (-1., 4.))
assert estimate.f_calls == 9
```

## Configuration And Memory

The immutable constructor is
`HessianPlan(dimension, scheme='central', step=None, *, memory_bytes=64*1024*1024)`.
The dimension is a positive integer that fits the native vector callback ABI.
Schemes are `'forward'`, `'backward'` and `'central'`. An explicit step is a
finite, nonzero real number or a vector with one such value per axis, and
different signs and sizes per axis are allowed. Steps are copied, also from
read-only, negative-stride, zero-stride or unaligned buffers. Text, boolean,
complex, nonfinite and malformed steps are rejected.

`step=None` chooses
\(h_j=\max(1,|x_j|)\epsilon^{1/4}\) centrally, or
\(h_j=\max(1,|x_j|)\epsilon^{1/3}\) one-sided, with binary64 machine epsilon.
These are rules of thumb that ignore constants depending on the function. An
explicit scalar applies to every axis, and a negative step reverses the
sampling direction.

Properties `dimension`, `scheme`, `step`, `shape`, `sample_count`,
`storage_bytes`, `workspace_size`, `workspace_bytes`, `output_bytes`,
`evaluation_bytes` and `memory_bytes` are read-only. `sample_count` excludes
the base residual, `workspace_size` is in doubles and the other sizes are in
bytes. `memory_bytes` reports the native storage actually required, not the
configured limit.

For dimension \(n\), the output has \(n^3\) doubles and the numerical scratch
\(2n^2+10n\). An evaluation also copies the point and the base residual,
another \(2n\) doubles. Explicit steps add \(n\) persistent doubles; automatic
plans store none. Construction checks the total, including the storage one
evaluation will need, against `memory_bytes` before allocating, and products
and native capacities are checked however large the budget. Python owners,
nested-tuple exports and allocator overhead come on top, and concurrent
evaluations each allocate their own evaluation storage.

## Rounded Stencils

The plan uses the actual floating-point shifted coordinates and
displacements. Central samples along an axis are requested at
\(x_j+h_j,x_j-h_j\), forward samples at \(x_j+h_j,x_j+2h_j\) and backward
samples at \(x_j-h_j,x_j-2h_j\). Every coordinate, displacement, distinctness
condition and interpolation coefficient is validated before any stencil
callback, although the base residual may be evaluated before that check.

For actual displacements \(a,b\), the diagonal uses the three-point interpolant

\[
D_{jj}F=\frac{2}{a-b}
\left(\frac{F(x+ae_j)-F(x)}a-\frac{F(x+be_j)-F(x)}b\right).
\]

Central mixed partials use tensor products of the three-point first-derivative
weights. Define \(\alpha_j(a)=b/(b-a)\) and
\(\alpha_j(b)=a/(a-b)\). For axes \(j,k\), the plan sums four terms

\[
D_{jk}F=\sum_{p,q}\alpha_j(p)\alpha_k(q)
\frac{[F(x+pe_j+qe_k)-F(x+pe_j)]-[F(x+qe_k)-F(x)]}{pq}.
\]

The weights use the normalized actual displacements. Corner weights shared by
several terms are applied after a compensated sum, so small contributions are
not lost. The final scaling combines binary exponents before converting to
binary64, which avoids premature overflow or underflow when axis scales are
extreme. (The usual symmetric four-corner formula is biased when rounding makes
opposite offsets unequal.) One-sided mixed partials use a single rectangle of
first offsets. Compensated summation reduces accumulation error but does not
remove cancellation or noise.

One-sided approximations are first order. Central mixed partials are second
order even for moderately unequal spacing, but the central diagonal needs
symmetric spacing to be second order, so severe rounding asymmetry can cost
accuracy. An `ok` estimate means the computation finished, not that the
derivative is accurate; check it by varying the step. Nonfinite intermediate
arithmetic is a failure, even when another ordering could give a finite
value.

## Counts, Results And Failures

`evaluate(function, x, f_x=None)` accepts a Python function or a residual
`VectorCallback` of matching dimension; Jacobian and uninitialized adapters are
rejected. The point and the optional cached `f_x`, which must belong to the
same point and function, are copied while the callback is reserved.

Samples along each axis are taken first (the first offset, then the second)
and reused; then come the unordered pairs of axes. Central pairs visit the
corners `(+,+)`, `(+,-)`, `(-,+)`, `(-,-)`, where the signs refer to the
requested steps, explicit signs included. A supplied base costs nothing.
Otherwise central sampling costs \(2n^2\) residual attempts and one-sided
sampling \(n(n+3)/2\), every attempt counting, failed ones included.

The immutable result is
`HessianEstimate(hessian, f_calls, status, callback_code)`. Its nested-tuple
shape is `(residual, input, input)`. Only the two input axes are symmetric:
the plan computes one triangle and copies it to the other exactly. Returned
estimates have status `'ok'`, `'nonfinite'` or `'breakdown'`. Duplicate or
collapsed nodes report `breakdown`, and nonfinite coordinates, coefficients,
samples or arithmetic report `nonfinite`. A failure fills the whole Hessian with
NaN, including entries finished before it. Every callback output is set to NaN
beforehand, so values a callback did not write can never become derivatives.

Python callback exceptions propagate as the original instance with its
traceback. A nonzero native code raises `NativeCallbackError` with the exact
`native_status`, the failed estimate as `result` (with status
`'callback_failure'`), and the stage `'function'` for the base or `'hessian'`
for a stencil sample. The reservation covers conversion, allocation, execution,
result construction and exception transfer. An active adapter cannot be used
concurrently or reentrantly, but the plan itself can serve independent
callbacks concurrently or in nested evaluations, each with its own scratch and
with native contexts synchronized where needed.

## Native Consumers And Scope

The installed `hessians.pxd` exposes `HessianConfig`, a function to copy it,
`compute_hessian(...) noexcept nogil` and the plan's `compute`. The native
output offset is `residual + n*(input_j + n*input_k)`, extending the
column-major Jacobian layout. Callers pass valid, non-overlapping input, base,
output and scratch spans with their capacities, keep the configuration's
borrowed steps alive, and follow the residual callback ABI. An invalid
configuration or too little output or scratch capacity returns `breakdown`
before anything is written or called. Once the capacities pass, any failure
fills the output with NaN and restores the scratch trial point to the original
point. Nothing is allocated, and neither Python nor the GIL is involved, during
the numerical work.

The [compiled example](../examples/native_hessian.md) uses the installed
declarations and native callbacks without NumPy C headers. This square plan
works with [vector Chebyshev](native-vector-chebyshev.md) and the Halley family;
[rectangular Hessians](native-rectangular-hessian.md) and
[higher mixed partials](native-higher-mixed.md) cover different input and
output dimensions and higher orders. Samples within one evaluation are taken one
after another. See [Finite Differences](finite-differences.md) for every
derivative plan.
