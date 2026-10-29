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

The plan holds no callback, sample cache or mutable workspace. It computes
with stack variables, so independent solves can share it concurrently. With a
compiled residual, all sampling and arithmetic run without the GIL; Python
callbacks take it. The plan always differentiates the residual twice, even when
an analytic first derivative is supplied.

## Stencil And Accuracy

All schemes use the cached residual at the current point and two additional
samples, so `sample_count` is always two. Central samples at `x+h`, then `x-h`;
forward at `x+h`, then `x+2h`; backward at `x-h`, then `x-2h`.
An explicit step is a finite, nonzero absolute displacement; a negative step
reverses the direction. Text and sequences are rejected.

Automatic steps are recomputed at each accepted point as
\(h=\eta\max(1,|x|)\). Central uses
\(\eta=\epsilon^{1/4}=2^{-13}\); one-sided schemes use
\(\eta=\sqrt[3]{\epsilon}\), for binary64 machine epsilon \(\epsilon\).
These choices balance truncation and rounding error for smooth functions and
do not adapt to noisy, nonsmooth or badly scaled ones.

With actual rounded displacements \(a\) and \(b\), the plan computes the
second derivative of the quadratic interpolating the three sampled points:

\[
f''(x)\approx\frac{2}{a-b}
\left(\frac{f(x+a)-f(x)}{a}-\frac{f(x+b)-f(x)}{b}\right).
\]

The formula stays exact for quadratics when rounding makes the spacings
unequal. The central truncation error is second order for symmetric spacing,
and the one-sided error first order. Smaller steps eventually amplify
cancellation and rounding, and with these approximations neither exact
derivatives nor cubic solver convergence are guaranteed.

Every coordinate, displacement and separation is checked before sampling.
Collapsed or duplicate nodes report an unsuccessful `breakdown`, and nonfinite
coordinates, samples or intermediate arithmetic report `nonfinite`. Steps and
schemes are never changed on the fly. To keep accuracy under cancellation, the
secants are normally subtracted before dividing; if that subtraction
overflows, each secant is divided by the separation first. Any other
unrepresentable intermediate is a failure, even if another arrangement might
succeed.

## Counts And Failures

Each plan evaluation adds one second-derivative attempt and up to two
residual attempts to the solver's `(residual, first_derivative,
second_derivative)` counts, failed attempts included. `d2f_x0` skips the
initial evaluation, and `f_x0` supplies the base residual. The first- and
second-derivative plans sample independently and share no samples. Stencils add
no iterations. No derivative work is done when the starting point is a root or
after a step that ends the solve.

When the second derivative fails, the accepted point, residual and first
derivative stand and the second derivative is NaN; when the first derivative
fails, both are cleared. Errors in residual samples keep the exact native
status or the original Python exception, with
`callback_stage='second_derivative'` for this plan. The residual adapter stays
reserved through sampling, export and error transfer. The plan can be shared,
but an active adapter cannot be used concurrently.

The installed `cyroot/native/scalar_derivatives.pxd` exposes the low-level
`ScalarSecondDerivativePlan.compute(...) noexcept nogil`, with the same pointer,
callback and counter rules as the first-derivative plan. Callers pass a finite
base residual and an initialized plan. The output is NaN on failure, and
successful callbacks must write their output. The plan has no standalone Python
evaluator. See the [compiled example](../examples/native_higher_order.md).

[Runtime-order scalar derivative plans](native-higher-derivative.md) are
separate, and [Householder](native-householder.md) accepts them.
[Vector Hessian plans](native-hessian.md) provide square and rectangular
residual Hessians for the higher-order vector solvers. Samples within one
evaluation are taken one after another.
