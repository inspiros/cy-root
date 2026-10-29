# Native Runtime-Order Scalar Derivatives

`cyroot.native.ScalarFiniteDifferencePlan` estimates a scalar derivative of any
order chosen at run time, without symbolic algebra, NumPy or a table of
formulas. It accepts Python residuals and compiled `ScalarCallback` adapters:

```python
import math
from cyroot.native import ScalarFiniteDifferencePlan

plan = ScalarFiniteDifferencePlan(3)
estimate = plan.evaluate(math.exp, .4)
assert estimate.status == 'ok'
assert abs(estimate.derivative - math.exp(.4)) < 1e-4

# High orders work too; there is no table of formulas to run out of.
estimate = ScalarFiniteDifferencePlan(12, step=1.).evaluate(lambda x: x**12, 0.)
assert math.isclose(estimate.derivative, math.factorial(12), rel_tol=1e-10)
```

The immutable constructor is
`ScalarFiniteDifferencePlan(order, scheme='central', step=None, *, max_nodes=65,
workspace_bytes=8*1024*1024)`. The order must be a positive integer.
`max_nodes` and `workspace_bytes` are resource limits, and exceeding them
raises an error during setup; any order within them works. A larger budget
allows larger stencils but does not make them accurate. Text, sequences,
booleans and complex steps are rejected; an explicit step must be a finite,
nonzero real number, and a negative step reverses the sampling direction.

The properties `order`, `scheme`, `step`, `stencil_size`, `sample_count`,
`workspace_size` and `workspace_bytes` are read-only; the last two give the
scratch actually required, in doubles and in bytes, not the configured limit.
Plans store no callback, mutable scratch or samples, so they can be shared
concurrently by independent callbacks with their own workspaces.

## Rounded Stencils And Accuracy

For order \(m\), forward/backward stencils include the base and \(m\) other
points at signed integer multiples of the step. Central stencils include the
base and pairs at \(\pm h,\ldots,\pm rh\), with \(r=\lceil m/2\rceil\).
Samples visit the base first if needed, then forward/backward multiples, or
central pairs in the order `+h, -h, +2h, -2h, ...`.

The engine computes Fornberg interpolation weights from the actual rounded
displacements, normalized by the largest one. All points, their distinctness
and the weights are validated before any stencil callback. Collapsed or
duplicate nodes report `breakdown`, and nonfinite coordinates, weights, samples
or arithmetic report `nonfinite`; stencils are never repaired, retried or
swapped. The base residual is subtracted before a compensated weighted sum, and
the result is divided by the displacement scale repeatedly instead of forming
`h**order`.

One-sided truncation error is first order. Central truncation error is second
order for symmetric spacing, but for even orders rounding asymmetry can undo
the cancellation that symmetry provides. Odd central orders use an extra node
and keep second-order truncation error for moderately unequal spacing.

Automatic steps use \(h=\max(1,|x|)\epsilon^{1/(m+p)}\), with \(p=2\) for
central and \(p=1\) for one-sided stencils. This is a rule of thumb that
ignores constants depending on the function and the stencil, while high
derivatives amplify cancellation, noise and residual offsets. An `ok` status
means the computation finished with finite values, not that any accuracy was
reached: high-order tests on polynomials can pass while a smooth
non-polynomial function of the same order is inaccurate. Check estimates by
varying the step.

## Estimates, Counters And Native Callers

`evaluate(function, x, f_x=None)` returns immutable
`ScalarDerivativeEstimate(derivative, f_calls, status, callback_code)`. Status is
`'ok'`, `'nonfinite'` or `'breakdown'` for returned estimates. `f_calls` counts
every residual attempt, failures included, and includes the base unless `f_x`
is supplied; `sample_count` excludes the base.

Python callback exceptions propagate as the original instance with its
traceback. Nonzero native codes raise `NativeCallbackError` with the failed
estimate, whose status is `'callback_failure'`, the exact code and the stage:
`'function'` for the base, `'derivative'` for a stencil sample. Each output is
set to NaN before the call, and a successful callback must write it. Numerical
failures leave the derivative NaN.

The callback adapter stays reserved through conversion, allocation, sampling,
result construction and exception transfer. Independent callbacks can share
one plan, but an active adapter cannot be used concurrently or reentrantly.
`evaluate` allocates private scratch before sampling.

The installed `higher_derivatives.pxd` exposes `compute(...) noexcept nogil`,
which works in caller-owned scratch. Native callers pass the finite base
residual, valid non-overlapping spans and the scratch capacity in doubles; too
little capacity fails before anything is written, and the scratch contents are
unspecified afterwards. The engine needs \(n(m+3)\) doubles, the Fornberg
arithmetic costs \(O(n^2m)\), and it never allocates, calls Python or takes
the GIL, so a native residual gives a fully native path. The
[Householder solver](native-householder.md) accepts these plans for any
derivative order; Newton and the Halley family use the dedicated first- and
second-derivative plans. See the
[compiled example](../examples/native_higher_derivative.md).

## Householder Without Symbolic Formulas

For derivatives through order \(d\), write
\(a_k=f^{(k)}(x)/k!\). Reciprocal Taylor coefficients satisfy

\[
b_0=\frac1{a_0},\qquad
b_n=-\frac1{a_0}\sum_{k=1}^n a_kb_{n-k}.
\]

The Householder increment is

\[
\Delta x=d\frac{(1/f)^{(d-1)}}{(1/f)^{(d)}}
        =\frac{b_{d-1}}{b_d}.
\]

The factorials cancel, so the recurrence works for any order chosen at run
time, with no symbolic formulas. The installed `householder.pxd` exposes the
allocation-free C++17 `householder_step` kernel, which takes the raw
derivatives `f, f', ..., f^(d)` and `2*(d+1)` caller-owned `ScaledCoefficient`
scratch entries. Each coefficient carries a binary64 mantissa and a checked
64-bit base-two exponent, which avoids overflow in the factorials and in the
reciprocal near a pole; only the final increment is converted to binary64. The
recurrence costs \(O(d^2)\) arithmetic and \(O(d)\) workspace.

The status is zero for success, one for nonfinite input or arithmetic, and two
for an invalid size or a zero reciprocal denominator. A residual that is
already zero gives a zero increment without looking at the derivatives. The
kernel does not need a nonzero first derivative. Scaled arithmetic extends the
range, not the precision: cancellation, inaccurate derivatives and derivatives
that already overflowed still limit the result. A final step can underflow to
zero; the solver then decides between residual success and stagnation.

The [resumable Householder solver](native-householder.md) combines this kernel
with callback providers, caches, stopping rules, history and numerical
derivatives, and `cyroot.householder` runs that solver. Python setup and result
export still need the GIL even when the kernel and the callbacks do not.
