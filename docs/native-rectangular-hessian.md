# Native Rectangular Numerical Hessians

`RectangularHessianPlan` estimates second derivatives of
\(F:\mathbb R^n\to\mathbb R^m\), including \(m<n\), \(m>n\) and \(m=n\).
The output shape is \((m,n,n)\): residual component first, then the two input
axes. Only the input axes are symmetric. It shares the numerical stencil with
`HessianPlan`; the square plan's API and native configuration remain available.

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
Both dimensions are positive integers fitting C `int`. Properties are read-only:
`input_dimension`, `output_dimension`, `shape`, `scheme`, `step`, `sample_count`,
`storage_bytes`, `workspace_size`, `workspace_bytes`, `output_bytes`,
`evaluation_bytes` and `memory_bytes`. Reported memory is the requirement,
not the ceiling passed to the constructor.

`evaluate(function, x, f_x=None)` accepts a Python callable or a matching
residual-kind `RectangularCallback`. A residual-kind `VectorCallback` is also
accepted when \(m=n\). Callback shape/kind validation precedes input conversion.
Callbacks are reserved across point/cache conversions, evaluation and copied
result or exception creation. Plans can be reused concurrently with independent
callbacks and storage. Python callbacks acquire the GIL; compiled callbacks and
the numerical stencil execute without it. NumPy and SciPy remain dependencies.

## Sampling And Failures

The schemes, signed scalar/anisotropic copied steps, automatic step heuristics,
actual rounded-node interpolation and finite-range behavior are those of the
[square Hessian plan](native-hessian.md). Central sampling costs \(2n^2\)
residual attempts; forward/backward costs \(n(n+3)/2\). These counts depend on
input dimension, regardless of the number of residual components. An uncached
base adds one attempt. The callback always writes all \(m\) components.

Results reuse `HessianEstimate(hessian, f_calls, status, callback_code)` with
copied tuple tensors. Input-axis mirrors are copied exactly. A finite `'ok'`
result certifies completion, not accuracy; noise and cancellation still require
step-refinement checks. Collapsed nodes report `'breakdown'`; nonfinite inputs,
coefficients, samples or arithmetic report `'nonfinite'`. Every failure clears
the entire output tensor to NaN, including previously completed components.
Callback buffers start as NaN so partial outputs fail explicitly.

Python exceptions retain their identity and traceback. Native callback failures
raise `NativeCallbackError` with the exact integer code, copied estimate and
stage `'function'` for the base or `'hessian'` for a stencil sample. Counts
include failed attempts. Trial coordinates are restored after every sample and
failure. All axis nodes and interpolation coefficients are preflighted before
stencil callbacks; an uncached base evaluation may precede that preflight.

## Native Storage And Cython Interface

Output storage is \(mn^2\) doubles. Scratch contains \(2nm+8n+2m\) doubles:
an axis-sample cache, geometry vectors, a trial point, corner residual and
compensation residual. Evaluation also copies the point and base, costing
\(n+m\). Explicit steps add \(n\) persistent doubles; automatic plans store
none. `memory_bytes` bounds their combined requirement before allocation.
Python owners, tuple exports and allocator overhead are additional. Each
concurrent evaluation allocates its own evaluation storage; no allocation occurs
between native samples.

Cython callers can import `RectangularHessianPlan`, `RectangularHessianConfig`
and `compute_rectangular_hessian` from `cyroot.native.hessians`. The config
contains `input_dimension`, `output_dimension`, `matrix_size` \(=nm\),
`tensor_size` \(=mn^2\), `work`, integer `scheme` and borrowed `steps`.
Schemes zero/one/two mean forward/backward/central, matching the square plan.
Use `plan.copy_config(&config)` to borrow a live immutable configuration.
The native tensor offset is `o + m*(j + n*k)`.

Both `plan.compute` and `compute_rectangular_hessian` accept disjoint spans
`x[n]`, `f0[m]`, `out[m*n*n]`, `workspace[2*n*m+8*n+2*m]`, explicit output/work
capacities, a `RectangularFunction`, its context, and nonnull count/code pointers.
The function signature is `(const double* x, size_t n, size_t m, double* out,
void* context) noexcept nogil`. The caller reserves and pins the callback,
configuration, context and step storage throughout the call. Input spans are
read-only; callbacks must fill their complete output on success.

The axis cache offset is `(2*axis+side)*m+component`; `side` zero/one means the
first/second axis sample. The trial point begins at `2*n*m+7*n`. Invalid
configuration or insufficient capacities returns `DIFF_BREAKDOWN`, resets
`callback_code` to zero and leaves outputs, scratch and call count untouched.
With valid spans, every failure poisons the tensor and restores the trial point.
The old square `HessianConfig` and `compute_hessian` retain their layout and
signature, forwarding through a stack-only callback bridge to the shared kernel.

This API supplies numerical derivatives. The analytic
[`RectangularHessianCallback`](native-rectangular-tensors.md) uses the same tensor
layout. The [Chebyshev path](native-rectangular-chebyshev.md) supports rectangular
nonlinear systems; fixed [Halley/Super-Halley](native-rectangular-halley.md) also support rectangles,
and [general modified Halley](native-rectangular-modified.md) uses its separate
three-solve rectangular correction.
The 1.x general derivative wrappers were removed. Select an explicit
[finite-difference plan](finite-differences.md); old `h` values should be
rechecked against the chosen plan's step and stencil conventions.
