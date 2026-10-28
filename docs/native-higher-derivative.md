# Native runtime-order scalar derivatives

`cyroot.native.ScalarFiniteDifferencePlan` estimates a selected scalar derivative
order without symbolic algebra, NumPy operations or a table of explicit formulas.
It accepts Python residuals and compiled `ScalarCallback` adapters:

```python
import math
from cyroot.native import ScalarFiniteDifferencePlan

plan = ScalarFiniteDifferencePlan(3)
estimate = plan.evaluate(math.exp, .4)
assert estimate.status == 'ok'
assert abs(estimate.derivative - math.exp(.4)) < 1e-4

# Orders beyond the old explicit reciprocal-formula table are supported.
estimate = ScalarFiniteDifferencePlan(12, step=1.).evaluate(lambda x: x**12, 0.)
assert math.isclose(estimate.derivative, math.factorial(12), rel_tol=1e-10)
```

The immutable constructor is
`ScalarFiniteDifferencePlan(order, scheme='central', step=None, *, max_nodes=65,
workspace_bytes=8*1024*1024)`. Order must be a positive integer. `max_nodes` and
`workspace_bytes` are configurable resource limits; exceeding them raises an
error during setup. There is no fixed list of implemented derivative orders.
Increasing a budget permits larger stencils but does not ensure numerical accuracy.
Text, sequences, booleans and complex steps are rejected; an explicit step must
be a finite nonzero real scalar. Negative steps reverse the sampling direction.

Properties `order`, `scheme`, `step`, `stencil_size`, `sample_count`,
`workspace_size` and `workspace_bytes` are read-only. The last two describe the
required scratch in doubles and bytes, rather than the configured ceiling.
Plans store no callback, mutable scratch or sampled values and can be reused
concurrently with independent callbacks and workspaces.

## Rounded stencils and accuracy

For order \(m\), forward/backward stencils include the base and \(m\) other
points at signed integer multiples of the step. Central stencils include the
base and pairs at \(\pm h,\ldots,\pm rh\), with \(r=\lceil m/2\rceil\).
Samples visit the base first if needed, then forward/backward multiples, or
central pairs in the order `+h, -h, +2h, -2h, ...`.

The engine computes Fornberg interpolation weights from the actual rounded
displacements, normalized by their largest magnitude. It validates all points,
distinctness and weights before stencil callbacks. Collapsed/duplicate nodes
report `breakdown`; nonfinite coordinates, weights, samples or arithmetic
report `nonfinite`. It does not repair, retry or switch stencils. The base
residual is subtracted before compensated weighted summation, and the result
is divided by the displacement scale repeatedly to avoid forming `h**order`.

One-sided truncation is first order. Central truncation is second order on
symmetric spacing; for even derivative order, rounding asymmetry can remove
the symmetry cancellation. Odd central orders use an extra interpolation node
and retain second-order truncation for bounded unequal spacing.

Automatic steps use \(h=\max(1,|x|)\epsilon^{1/(m+p)}\), with \(p=2\) for
central and \(p=1\) for one-sided stencils. This is a heuristic. High derivatives
amplify cancellation, noise and residual offsets; the step choice omits
function-dependent and stencil-conditioning constants. An `ok` status means a
finite computation completed, not that a requested accuracy was achieved.
High-order polynomial fixtures can pass while a smooth nonpolynomial at the
same order is inaccurate. Step-refinement studies remain necessary.

## Estimates, counters and native callers

`evaluate(function, x, f_x=None)` returns immutable
`ScalarDerivativeEstimate(derivative, f_calls, status, callback_code)`. Status is
`'ok'`, `'nonfinite'` or `'breakdown'` for returned estimates. `f_calls` counts
every residual attempt, including the base unless `f_x` is supplied. The cached
base is reused; `sample_count` excludes it. Failed attempts count too.

Python callback exceptions retain their original identity and traceback. Native
nonzero codes raise `NativeCallbackError` with the failed estimate, exact status
and stage `'function'` for the base or `'derivative'` for a stencil sample. Its
estimate has status `'callback_failure'`. Each output is seeded NaN; a successful
callback must write its output. Numerical failures leave the derivative NaN.

The callback adapter remains reserved through numeric conversion, allocation,
sampling, result boxing and exception transfer. Independent callbacks can reuse
the same immutable plan; an active adapter cannot be reused concurrently or
reentrantly. The convenience evaluator allocates private scratch before sampling.

The installed `higher_derivatives.pxd` exposes `compute(...) noexcept nogil`
with caller-owned scratch. Native callers supply the cached finite base, valid
disjoint spans and capacity in doubles. Insufficient capacity fails before
scratch writes. Scratch contents are unspecified on return. The engine uses
\(n(m+3)\) doubles and Fornberg arithmetic costs \(O(n^2m)\); it makes no heap,
Python or GIL calls. A native residual gives a fully native sampling path.
The [Householder solver](native-householder.md) accepts these generic plans
as matching derivative roles. Existing Newton and Halley-family dispatch still
uses the specialized first/second plans.
See the [compiled example](../examples/native_higher_derivative.md).

## Householder without symbolic formulas

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

The factorials cancel. This recurrence works for runtime order; it needs no
SymPy fallback or prewritten expression beyond order ten. Installed
`householder.pxd` exposes the allocation-free C++17 `householder_step` kernel,
with raw derivatives `f, f', ..., f^(d)` and `2*(d+1)` caller-owned
`ScaledCoefficient` scratch entries. Coefficients carry a binary64 mantissa and
a checked 64-bit base-two exponent. This avoids intermediate factorial and
reciprocal-pole overflow; only the final increment is converted to binary64.
The recurrence costs \(O(d^2)\) arithmetic and \(O(d)\) workspace.

Statuses are zero for success, one for nonfinite input/arithmetic and two for
invalid size or zero reciprocal denominator. An already zero residual returns
a zero increment without inspecting unused derivative values. The kernel does
not require a nonzero first derivative. Scaled arithmetic extends range, not
precision: cancellation, inaccurate derivatives and already overflowing user
derivatives still limit results. A final step may underflow to zero; the eventual
solver tests residual success versus stagnation.

The [resumable Householder solver](native-householder.md) integrates this kernel
with callback providers, caches, stopping, history and numerical derivatives.
The established `cyroot.householder` entry point forwards to the native solver;
its former symbolic machinery and the SymPy package dependency are removed.
Public Python setup/results still require the GIL even when the numerical
kernel and callbacks do not.
