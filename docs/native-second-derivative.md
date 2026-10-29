# Native Scalar Second Derivatives

`cyroot.native.ScalarSecondDerivativePlan(scheme='central', step=None)`
approximates a scalar residual's second derivative for native
[Halley-family methods](native-higher-order.md). Its immutable properties are
`scheme`, `step` and `sample_count`. Pass it as the second derivative argument;
pass `None` for the default central plan:

```python
from cyroot.native import Halley, ScalarSecondDerivativePlan

solver = Halley(lambda x: x*x - 2., lambda x: 2*x,
                ScalarSecondDerivativePlan(), 1.5, history=16)
assert solver.run().success

# Both derivatives can be approximated using the same residual:
assert Halley(lambda x: x*x - 2., None, None, 1.5).run().success
```

The plan holds no callback, sample cache or mutable workspace. Native evaluation
uses stack variables and can run concurrently in independent solves sharing
the plan. A compiled residual makes the complete sampling and arithmetic path
GIL-free. Python callbacks reacquire the GIL. The plan differentiates the
residual directly, including when an analytic first derivative is supplied.

## Stencil And Accuracy

All schemes use the cached residual at the current point and two additional
samples, so `sample_count` is always two. Central samples at `x+h`, then `x-h`;
forward at `x+h`, then `x+2h`; backward at `x-h`, then `x-2h`.
An explicit step is a signed finite nonzero scalar absolute displacement.
Negative steps reverse the nominal direction. Text and sequences are rejected.

Automatic steps are recomputed at each accepted point as
\(h=\eta\max(1,|x|)\). Central uses
\(\eta=\epsilon^{1/4}=2^{-13}\); one-sided schemes use
\(\eta=\sqrt[3]{\epsilon}\), for binary64 machine epsilon \(\epsilon\).
These heuristics balance smooth truncation and roundoff errors; they do not
adapt to noisy, nonsmooth or badly scaled functions.

With actual rounded displacements \(a\) and \(b\), the plan computes the
second derivative of the quadratic interpolating the three sampled points:

\[
f''(x)\approx\frac{2}{a-b}
\left(\frac{f(x+a)-f(x)}{a}-\frac{f(x+b)-f(x)}{b}\right).
\]

This preserves quadratic consistency when rounding makes the spacings unequal.
Central truncation error is second order on symmetric spacing. One-sided
truncation error is first order. Smaller steps eventually amplify cancellation
and roundoff; neither exact derivatives nor cubic solver convergence are
guaranteed with these approximations.

Every coordinate, displacement and separation is checked before sampling.
Collapsed or duplicate nodes report unsuccessful `breakdown`. Nonfinite
coordinates, samples or intermediate arithmetic report `nonfinite`. The engine
does not repair steps, retry or switch schemes. Subtraction normally precedes
division to retain cancellation accuracy. If finite secants overflow on
subtraction, each is divided by the separation first. Other unrepresentable
intermediates remain conservative failures even if another arrangement could
succeed.

## Counts And Failures

Each invoked plan adds one second derivative attempt and up to two residual
attempts to the solver's `(residual, first_derivative, second_derivative)`
counts. Failed attempts count. `d2f_x0` skips the initial plan, and `f_x0`
supplies its cached base residual. First and second derivative plans sample
independently; there is no cross-plan reuse. Stencils do not add iterations.
Initial convergence and later convergence, stagnation or budget termination
skip unused derivative work.

A failed second derivative retains the accepted point, residual and valid
first derivative, with second derivative NaN. A failed first derivative clears
both derivatives. Residual sample errors retain the exact native status or
original Python exception, with `callback_stage='second_derivative'` for this
plan. Residual adapters stay reserved throughout sampling, exports and error
transfer. An immutable plan can be shared, but an active callback adapter
cannot be concurrently reused.

The installed `cyroot/native/scalar_derivatives.pxd` exposes trusted
`ScalarSecondDerivativePlan.compute(...) noexcept nogil` with the same pointer,
callback and counter contract as the first derivative plan. Callers supply a
finite base residual and an initialized plan. Output is NaN on failure;
successful callbacks must write their output. The plan offers no standalone
Python evaluation method. See the [compiled example](../examples/native_higher_order.md).

[Runtime-order scalar derivative plans](native-higher-derivative.md) are available
separately, with generic-plan integration in [Householder](native-householder.md).
[Native vector Hessian plans](native-hessian.md) provide square and rectangular residual derivatives for higher-order vector
solvers. Sampling is serial within each evaluation. NumPy/SciPy remain package dependencies.
