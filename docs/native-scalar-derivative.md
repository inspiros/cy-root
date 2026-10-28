# Native scalar finite differences

`cyroot.native.ScalarDerivativePlan` provides forward, backward and central
first derivatives for native Newton. Pass a plan as the second argument, or
pass `None` for a default central plan:

```python
from cyroot.native import Newton, ScalarDerivativePlan

solver = Newton(lambda x: x*x - 2., ScalarDerivativePlan(), 1.5, history=16)
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

The residual can be a Python callable or a compiled `ScalarCallback`. With a
compiled residual, sampling and derivative arithmetic remain outside the GIL.
Python residuals reacquire it for each callback and output conversion.

`ScalarDerivativePlan(scheme='central', step=None)` stores only immutable
configuration. Its read-only properties are `scheme`, `step` and `sample_count`.
It owns no callback or sampling workspace and can serve independent concurrent
solves. Native computation uses stack variables; the scalar solver needs no
additional heap workspace for finite differences. This preview provides no
standalone Python derivative evaluator.

## Steps and arithmetic

Schemes are the strings `'forward'`, `'backward'` and `'central'`; integer codes
are rejected. An explicit step must be a finite nonzero signed scalar. It is
an absolute displacement. Negative steps reverse sampling direction. Text and
step sequences are rejected.

With `step=None`, the displacement is recomputed at each accepted point as
\(h=\eta\max(1,|x|)\), where \(\eta=2^{-26}\) for one-sided schemes and
\(\eta=\sqrt[3]{\epsilon}\) for central differences, with binary64 machine
epsilon \(\epsilon\). These choices do not guarantee accuracy for noisy,
nonsmooth or poorly scaled residuals.

The current residual is reused as the base value. Forward/backward schemes
sample once per derivative and use first-order secants. Central samples twice
and combines secants through the base point using the actual rounded offsets,
with the same weighted formula as [native Jacobians](native-jacobian.md).
For smooth functions this gives second-order truncation error, including when
rounding makes the two displacements unequal.

All required points and displacements are checked before sampling. A collapsed
step reports unsuccessful `breakdown`; nonfinite points, samples or arithmetic
report `nonfinite`. The engine does not repair the stencil, retry steps or
switch schemes. Subtraction/division overflow is a numerical failure, even
where an alternative algebraic arrangement could remain finite.

## Counts and failures

Newton's `f_calls` is `(residual_attempts, derivative_attempts)`. Residual counts
include every base, Newton trial and stencil call, including failed attempts.
Sampling does not add Newton iterations. `f_x0` supplies a cached base value;
`df_x0` skips the initial derivative approximation. Initial roots and later
convergence, stagnation or iteration-budget stops skip unused derivatives.

On a failed derivative approximation, the accepted point/residual stay valid
and `df_root` is NaN. Python exceptions retain their original instance and
callback traceback. Native failures preserve their exact code in
`NativeCallbackError`, with `callback_stage='derivative'`. The solver exposes
the same stage and code. Residual adapters stay reserved throughout sampling
and exception transfer; sharing an immutable plan does not permit concurrent
reuse of an active callback adapter.

The installed `scalar_derivatives.pxd` exposes trusted `compute` with a native
scalar callback, output pointer, residual-call counter and separate callback
status. Pointer/no-throw preconditions are documented there. Output is NaN on
failure; successful callbacks must write their output. See the
[compiled example](../examples/native_scalar_derivative.md).

This first-derivative plan also supplies the first derivative of
[Halley-family methods](native-higher-order.md). Their second derivative accepts a
separate [ScalarSecondDerivativePlan](native-second-derivative.md), and the
two plans sample independently. [Runtime-order scalar plans](native-higher-derivative.md)
are available through a separate evaluator and trusted native interface.
NumPy/SciPy remain package dependencies.
