# Vector finite differences on the native core

`generalized_finite_difference` and `GeneralizedFiniteDifference` now use the
native derivative plans. Their established signatures and float64 NumPy output
remain available:

```python
import numpy as np
from cyroot import generalized_finite_difference, GeneralizedFiniteDifference

def residual(point):
    assert isinstance(point, np.ndarray)
    x, y = point
    return (x*x*y, x*y*y)

answer = generalized_finite_difference(residual, [.5, 1.], h=.125, order=3)
assert answer.shape == (2, 2, 2, 2)
assert answer.dtype == np.float64 and answer.flags.c_contiguous
assert abs(answer[0, 0, 0, 1]-2.) < 1e-10
wrapper = GeneralizedFiniteDifference(residual, h=.125, order=3)
assert np.array_equal(wrapper([.5, 1.]), answer)
wrapper.order = 2
hessian = wrapper([.5, 1.])
assert hessian.shape == (2, 2, 2)
assert wrapper.n_f_calls == 2
```

For \(n\) inputs and \(m\) residuals, order \(k\) returns shape
\((m,n,\ldots,n)\), with \(k\) input axes. Order one uses
`RectangularJacobianPlan`, order two uses `RectangularHessianPlan`, and later
orders use [TensorDifferencePlan](native-higher-mixed.md). The old NumPy tensor
accumulation and Python stencil cache have been removed; there is one numerical
implementation for each derivative order family.

## Callback boundaries

Ordinary Python functions and legacy `NdArrayFPtr` providers receive a private
NumPy point snapshot on every call. They can retain or mutate it and reuse an
output array; the bridge copies each finite one-dimensional residual before the
next sample. NumPy conversions occur at this compatibility boundary and for the
final result. Python and legacy ndarray callbacks require the GIL.

Explicit residual-kind `RectangularCallback` and square `VectorCallback` adapters
are also accepted. They use the native sampling path without per-sample NumPy
conversion. Compiled callbacks run without the GIL; Python functions wrapped in
these adapters receive the adapters' usual tuple inputs and acquire the GIL.
The supplied adapter is reserved before point, step or cache conversion and
remains reserved through result export and error delivery. Matching native
provider dimensions avoid an extra residual-shape discovery call. Square
adapters require equal input and output lengths; rectangular adapters declare
both. Jacobian-kind adapters and Hessian callbacks are not residual providers.

The function signature remains
`generalized_finite_difference(F, x, F_x=None, h=1., order=1, kind=0)`.
The wrapper remains `GeneralizedFiniteDifference(F, h=1., order=1, kind=0)`.
`kind` accepts `1/'forward'`, `-1/'backward'`, or `0/'central'`. `h` is a finite
nonzero scalar or a length-\(n\) vector; signed steps are supported. Public wrapper
attributes `h`, `order` and `kind` remain mutable and are validated on each call.
The wrapper retains one immutable native plan, keyed by validated dimensions,
order, normalized scheme and a snapshot of the step values. Changed options or
output lengths select a new plan; in-place step edits are checked too. Resource
limits are part of the key. Failed construction, sampling or result export leaves
the previous successful entry intact. Samples, callbacks, scratch and results
are never cached. The function entry point remains one-shot. `h=None` retains its established invalid-input behavior; use
native plans directly for automatic steps or configurable resource limits.

A supplied `F_x` skips the base callback. Otherwise the base is evaluated once
and reused, including when an ordinary function's output length must be inferred.
Shared sample reuse follows the selected native plan. Wrapper `n_f_calls` counts
derivative invocations; a supplied legacy function pointer keeps its separate
residual-attempt counter. Native plan estimates expose detailed attempt/status
information when needed.

## Numerical changes

Central first derivatives now sample `x+h` and `x-h`, instead of the old half
steps. Higher central stencils include the base and enough symmetric requested
nodes for odd multiplicities. The spacing convention now agrees with the native
Jacobian, Hessian and tensor plans. Sample coordinates, counts and results can
therefore differ from previous releases.

Weights are based on actual rounded displacements, rather than requested step
powers and binomial coefficients. This corrects unevenly rounded stencils.
Subtracting base values and the native range handling can also evaluate cases
where the previous implementation failed unnecessarily: a huge constant function
has zero derivatives, and a tiny step product need not itself cause failure.
Finite completion still gives no accuracy guarantee for noisy functions, high
orders, cancellation or ill-conditioned stencils. No convergence or speed claim
is made here.

Invalid points, shapes or steps raise `ValueError`. A collapsed stencil reports
that `h` does not produce finite, distinct sample points; nonfinite samples or
unrepresentable arithmetic raise `ValueError` stating that the derivative must
be finite. Messages referring to the removed binomial/weighted-sample accumulator
are no longer part of the failure policy. Original Python callback exceptions
retain their identity and traceback. Native callback errors retain their exact
status and the selected plan's estimate/stage through `NativeCallbackError`.

## Limits and compiled consumers

The compatibility entry points retain rank 32 including the output axis, so
`order <= 31`, a higher-order stencil-visit limit of 1,000,000, and a 256 MiB
native logical memory budget. Output is inherently exponential in order.
The retained plan keeps its native graph and step storage between calls; one
evaluation still owns private scratch. During replacement, the previous graph can
coexist with the candidate until a successful result is exported.
Python planning objects, nested-tuple exports, NumPy boundary copies, allocator
overhead and callbacks use additional memory; this is not a process peak-memory
limit. Oversized native plans reject before stencil sampling. With an ordinary
uncached function, one base call may be needed to discover its output dimension.

Existing Cython consumers may still cimport `VectorDerivativeApproximation`,
`GeneralizedFiniteDifference` and `generalized_finite_difference_kernel` from
`cyroot.vector_derivative_approximation`. The typed kernel's ndarray arguments,
`NdArrayFPtr` callback and `order/kind` defaults are retained. The derivative
base gains a private native-provider field and the wrapper retains private plan
fields, so compiled consumers must rebuild
against the current declarations. These ndarray compatibility calls require the
GIL for setup, bridging and result conversion. Use the native plans' raw compute
interfaces for caller-owned spans and entirely native numerical execution.

This migration does not make legacy derivative wrapper instances valid solver
providers. Native solvers continue accepting their documented native plans and
callbacks, keeping solver reservations and derivative configuration explicit.
NumPy and SciPy remain package dependencies.
