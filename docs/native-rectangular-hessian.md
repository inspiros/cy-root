# Native Rectangular Numerical Hessians

`RectangularHessianPlan` estimates second derivatives of
\(F:\mathbb R^n\to\mathbb R^m\), including \(m<n\), \(m>n\) and \(m=n\).
The output shape is \((m,n,n)\): residual component first, then the two input
axes, and only the input axes are symmetric. It uses the same stencil as the
square `HessianPlan`.

```python
from cyroot.native import RectangularHessianPlan

def residual(x):
    return (x[0]**2 + 3*x[0]*x[1],
            2*x[1]**2 - x[0]*x[1],
            4*x[0]**2 + x[1]**2)

plan = RectangularHessianPlan(2, 3, step=(.125, -.25))
estimate = plan.evaluate(residual, (1., 2.))
assert estimate.status == 'ok'
assert plan.shape == (3, 2, 2)
assert estimate.hessian == (((2., 3.), (3., 0.)),
                            ((0., -1.), (-1., 4.)),
                            ((8., 0.), (0., 2.)))
assert estimate.f_calls == 9
cached = plan.evaluate(residual, (1., 2.), f_x=residual((1., 2.)))
assert cached.hessian == estimate.hessian
assert cached.f_calls == 8
```

Construct with
`RectangularHessianPlan(input_dimension, output_dimension,
scheme='central', step=None, *, memory_bytes=64*1024*1024)`.
Both dimensions are positive integers that fit a C `int`. The read-only
properties are `input_dimension`, `output_dimension`, `shape`, `scheme`,
`step`, `sample_count`, `storage_bytes`, `workspace_size`, `workspace_bytes`,
`output_bytes`, `evaluation_bytes` and `memory_bytes`; the reported memory is
what is required, not the limit passed to the constructor.

`evaluate(function, x, f_x=None)` accepts a Python callable or a residual
`RectangularCallback` of matching shape, or a residual `VectorCallback` when
\(m=n\). The callback's shape and kind are checked before any input is
converted, and it stays reserved through the point and cache conversions, the
evaluation and the creation of the result or exception. Plans can be shared
concurrently by independent callbacks with their own storage. Python callbacks
take the GIL; compiled callbacks and the stencil arithmetic run without it.

## Sampling And Failures

Schemes, steps (scalar or per axis, signed), automatic steps, interpolation
on the actual rounded nodes and range handling are those of the
[square Hessian plan](native-hessian.md). Central sampling costs \(2n^2\)
residual attempts and forward or backward sampling \(n(n+3)/2\), depending
only on the input dimension, plus one for the base unless it is cached. Each
call evaluates all \(m\) components.

Results are `HessianEstimate(hessian, f_calls, status, callback_code)` with
copied tuple tensors, and mirrored input-axis entries are exact copies. An
`'ok'` result means the computation finished, not that it is accurate; noise
and cancellation still call for checks with varied steps. Collapsed nodes
report `'breakdown'`, and nonfinite inputs, coefficients, samples or
arithmetic `'nonfinite'`. Every failure fills the whole tensor with NaN,
including components already finished. Callback buffers start as NaN, so an
incompletely written output fails explicitly.

Python exceptions propagate as the original instance with its traceback.
Native callback failures raise `NativeCallbackError` with the exact integer
code, a copied estimate and the stage `'function'` for the base or `'hessian'`
for a stencil sample. Counts include failed attempts. The trial point is
restored after every sample and failure. All nodes and interpolation
coefficients are checked before any stencil callback, although an uncached
base may be evaluated first.

## Native Storage And Cython Interface

The output takes \(mn^2\) doubles. The scratch takes \(2nm+8n+2m\): a cache of
axis samples, geometry vectors, a trial point, a corner residual and a
compensation residual. An evaluation also copies the point and base, \(n+m\)
doubles. Explicit steps add \(n\) persistent doubles; automatic plans store
none. `memory_bytes` bounds the total before allocation; Python owners, tuple
exports and allocator overhead come on top. Each concurrent evaluation
allocates its own storage, and nothing is allocated between samples.

Cython callers can import `RectangularHessianPlan`, `RectangularHessianConfig`
and `compute_rectangular_hessian` from `cyroot.native.hessians`. The config
contains `input_dimension`, `output_dimension`, `matrix_size` \(=nm\),
`tensor_size` \(=mn^2\), `work`, integer `scheme` and borrowed `steps`.
Schemes zero, one and two mean forward, backward and central, as in the square
plan. `plan.copy_config(&config)` lends out the plan's immutable configuration,
which stays valid while the plan is alive. The native tensor offset is
`o + m*(j + n*k)`.

`plan.compute` and `compute_rectangular_hessian` both take non-overlapping
spans `x[n]`, `f0[m]`, `out[m*n*n]` and `workspace[2*n*m+8*n+2*m]`, the output
and work capacities, a `RectangularFunction` with its context, and non-null
count and code pointers. The function signature is
`(const double* x, size_t n, size_t m, double* out, void* context) noexcept nogil`.
The caller reserves the callback and keeps it, the configuration, the context
and the steps alive for the whole call. Input spans are read-only, and callbacks
must fill their whole output on success.

The axis cache offset is `(2*axis+side)*m+component`, with `side` zero for an
axis's first sample and one for its second, and the trial point starts at
`2*n*m+7*n`. An invalid configuration or too little capacity returns
`DIFF_BREAKDOWN`, resets `callback_code` to zero and leaves the outputs, scratch
and call count untouched. With valid spans, every failure fills the tensor with
NaN and restores the trial point. The square `HessianConfig` and
`compute_hessian` go through a stack-only callback bridge to the same kernel.

These plans compute numerical derivatives; the analytic
[`RectangularHessianCallback`](native-rectangular-tensors.md) uses the same
tensor layout. They serve the rectangular
[Chebyshev](native-rectangular-chebyshev.md),
[Halley and Super-Halley](native-rectangular-halley.md) and
[general modified Halley](native-rectangular-modified.md) methods. If you used
the removed 1.x derivative wrappers, choose a
[finite-difference plan](finite-differences.md) and recheck old `h` values
against that plan's step and stencil conventions.
