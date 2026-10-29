# Native Scalar Finite Differences

`cyroot.native.ScalarDerivativePlan` computes forward, backward or central
first derivatives for native Newton. Pass a plan as the derivative argument, or
`None` for a default central plan:

```python
from cyroot.native import Newton, ScalarDerivativePlan

solver = Newton(lambda x: x*x - 2., ScalarDerivativePlan(), 1.5, history=16)
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

The residual can be a Python callable or a compiled `ScalarCallback`. With a
compiled residual, sampling and the derivative arithmetic run outside the GIL;
a Python residual takes it for each call and output conversion.

`ScalarDerivativePlan(scheme='central', step=None)` holds only immutable
configuration, with the read-only properties `scheme`, `step` and
`sample_count`. It owns no callback or sampling workspace, so independent
solves can share it concurrently. The computation uses stack variables, so the
solver needs no extra heap memory for finite differences. This plan has no
standalone Python evaluator; use `ScalarFiniteDifferencePlan` for that.

## Steps And Arithmetic

Schemes are the strings `'forward'`, `'backward'` and `'central'`; integer codes
are rejected. An explicit step is a finite, nonzero absolute displacement; a
negative step reverses the sampling direction. Text and sequences are
rejected.

With `step=None`, the displacement is recomputed at each accepted point as
\(h=\eta\max(1,|x|)\), where \(\eta=2^{-26}\) for one-sided schemes and
\(\eta=\sqrt[3]{\epsilon}\) for central differences, with binary64 machine
epsilon \(\epsilon\). These choices do not guarantee accuracy for noisy,
nonsmooth or poorly scaled residuals.

The current residual is reused as the base value. Forward and backward
schemes take one sample per derivative and use a first-order secant. The
central scheme takes two samples and combines the secants through the base
point with the actual rounded offsets, using the same weighted formula as
[native Jacobians](native-jacobian.md). For smooth functions this gives a
second-order truncation error, even when rounding makes the two offsets
unequal.

All required points and displacements are checked before sampling. A step that
collapses reports an unsuccessful `breakdown`; nonfinite points, samples or
arithmetic report `nonfinite`. The stencil is never repaired, and steps or
schemes are never changed on the fly. Overflow in the subtraction or division
is a numerical failure, even where another arrangement might stay finite.

## Counts And Failures

Newton's `f_calls` is `(residual_attempts, derivative_attempts)`. The residual
count includes every base, trial and stencil call, failed ones too, and
sampling adds no iterations. `f_x0` supplies a cached base value, and `df_x0`
skips the initial approximation. No derivative is approximated when the
starting point is a root, or after a step that ends the solve.

When an approximation fails, the accepted point and residual stand and
`df_root` is NaN. Python exceptions propagate as the original instance with
the callback's traceback. Native failures raise `NativeCallbackError` with
the exact code and `callback_stage='derivative'`, and the solver exposes the
same stage and code. The residual adapter stays reserved through sampling and
exception transfer; sharing an immutable plan does not allow concurrent use of
an active adapter.

The installed `scalar_derivatives.pxd` exposes the low-level `compute`, which
takes a native scalar callback, an output pointer, a residual-call counter and a
separate callback status; its pointer and no-throw requirements are documented
there. The output is NaN on failure, and successful callbacks must write their
output. See the [compiled example](../examples/native_scalar_derivative.md).

This plan also supplies the first derivative of the
[Halley-family methods](native-higher-order.md), whose second derivative takes a
separate [ScalarSecondDerivativePlan](native-second-derivative.md); the two
plans sample independently. [Runtime-order scalar plans](native-higher-derivative.md)
have their own evaluator and native interface.
