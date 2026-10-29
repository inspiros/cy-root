# Native Vector Hessians

`cyroot.native.HessianPlan` estimates the Hessian of a square residual
\(F:\mathbb R^n\to\mathbb R^n\) using Python or compiled residual callbacks.
For different input/output lengths, use
[`RectangularHessianPlan`](native-rectangular-hessian.md), which shares its stencil.
Its numerical path uses native storage and arithmetic, without NumPy operations,
Python objects or allocations between samples. Python setup, callback bridges
and result exports still use the GIL.

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
The dimension is a positive integer fitting the existing native vector callback
ABI. Schemes are `'forward'`, `'backward'` and `'central'`. An explicit step is a
finite nonzero real scalar or a length-dimension vector. Signed anisotropic
steps are supported. Steps are copied, including from read-only, negatively
strided, zero-stride or unaligned native double buffers. Text, boolean, complex,
nonfinite and malformed steps are rejected.

`step=None` chooses
\(h_j=\max(1,|x_j|)\epsilon^{1/4}\) centrally, or
\(h_j=\max(1,|x_j|)\epsilon^{1/3}\) one-sided, with binary64 machine epsilon.
These are heuristics; they omit function-dependent conditioning constants.
An explicit scalar is copied to each axis. A negative step reverses the
sampling direction.

Properties `dimension`, `scheme`, `step`, `shape`, `sample_count`,
`storage_bytes`, `workspace_size`, `workspace_bytes`, `output_bytes`,
`evaluation_bytes` and `memory_bytes` are read-only. `sample_count` excludes the
base residual. `workspace_size` is measured in doubles; byte properties are
measured in bytes. The reported `memory_bytes` is the required native storage,
not the configured ceiling.

For dimension \(n\), the output has \(n^3\) doubles and numerical scratch has
\(2n^2+10n\). Evaluation also copies an input point and base residual, costing
\(2n\) doubles. Explicit plan steps add \(n\) persistent doubles; automatic
plans store no step vector. Construction checks the combined native requirement
against `memory_bytes`, including future single-evaluation storage, before
allocation. Checked products and native capacity limits apply even with a large
budget. Python owners, nested-tuple exports and allocator overhead use additional
memory. Concurrent evaluations each allocate their own evaluation storage.

## Rounded Stencils

The plan uses actual floating-point shifted coordinates and displacements.
Central axis samples are requested at \(x_j+h_j,x_j-h_j\); forward samples
at \(x_j+h_j,x_j+2h_j\), and backward at
\(x_j-h_j,x_j-2h_j\). Every coordinate, displacement, distinctness check and
interpolation coefficient is validated before stencil callbacks. The separately
acquired base residual may precede this preflight.

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

The weights use normalized actual displacements. Common corner weights are
applied after compensated summation to preserve small contributions. Final
scaling combines binary exponents before conversion to binary64; it avoids
premature overflow or underflow from multiplying or successively dividing
extreme axis scales. A symmetric four-corner
formula alone is biased when rounding makes opposite offsets unequal.
One-sided mixed partials use one first-offset rectangle. Compensated summation
reduces accumulation error; it does not eliminate cancellation or noise.

One-sided approximation is nominally first order. Central mixed interpolation
is second order on bounded unequal spacing; central diagonal second-order
cancellation requires symmetric spacing. Severe rounding asymmetry can reduce
accuracy. A finite `ok` estimate certifies completion, not derivative accuracy;
step-refinement studies remain necessary. Nonfinite intermediate arithmetic
reports failure even when an alternative ordering could produce a finite value.

## Counts, Results And Failures

`evaluate(function, x, f_x=None)` accepts a Python function or a residual
`VectorCallback` of the matching dimension. Jacobian-kind and uninitialized
adapters are rejected. Inputs and an optional cached `f_x` are copied under
callback reservations. Caches must describe the same point and function.

Axis samples are cached: first then second for each axis, then unordered
input-axis pairs. Central pair corners visit `(+,+)`, `(+,-)`, `(-,+)`, `(-,-)`;
these signs describe requested steps, including signed explicit steps.
A supplied base costs zero evaluations. Excluding it, central sampling costs
\(2n^2\) residual attempts, and one-sided sampling costs \(n(n+3)/2\).
Every attempt, including a failed one, counts.

The immutable result is
`HessianEstimate(hessian, f_calls, status, callback_code)`. Its nested-tuple
shape is `(residual, input, input)`. Only the two input axes are symmetric;
the plan computes one triangle and copies the mirror exactly. Status is `'ok'`,
`'nonfinite'` or `'breakdown'` for returned estimates. Duplicate/collapsed nodes
report `breakdown`; nonfinite coordinates, coefficients, samples or arithmetic
report `nonfinite`. Failures poison the entire Hessian with NaN, including
entries completed before the failing sample. Every callback output is seeded
NaN, so unwritten values cannot silently become derivatives.

Python callback exceptions retain their original identity and traceback.
A native nonzero callback code raises `NativeCallbackError` with exact
`native_status`, the failed estimate as `result`, and stage `'function'` for
base acquisition or `'hessian'` for stencil sampling. Its estimate has status
`'callback_failure'`. Reservations cover conversion, allocation, execution,
result boxing and exception transfer. An active adapter cannot be reused
concurrently or reentrantly. The immutable plan can serve independent callbacks
concurrently or in nested evaluations, with separate scratch and synchronized
native contexts where needed.

## Native Consumers And Scope

Installed `hessians.pxd` exposes `HessianConfig`, configuration copying,
`compute_hessian(...) noexcept nogil` and plan `compute`. Native output offset
is `residual + n*(input_j + n*input_k)`, extending column-major Jacobians.
Callers supply disjoint, valid input/base/output/scratch spans and capacities,
keep the config's borrowed step storage alive, and obey the residual callback
ABI. Invalid configuration or short output/scratch capacity returns `breakdown`
before span writes or callbacks. After valid-capacity preflight, failures poison
output and restore the scratch trial point to the original point. No allocation
or Python/GIL helpers occur in numerical execution.

The [compiled example](../examples/native_hessian.md) uses installed declarations
and native callbacks without NumPy C headers. This square plan integrates with [vector Chebyshev](native-vector-chebyshev.md)
and the Halley family. [Rectangular Hessians](native-rectangular-hessian.md) and
[higher mixed partials](native-higher-mixed.md) provide independent dimensions and
higher orders. Sampling is serial within an evaluation. See
[finite differences](finite-differences.md) for every derivative plan. Hessian
sampling needs no BLAS; the package still requires NumPy and SciPy for existing
algorithms and the initial native linear algebra provider.
