# Finite Differences

Numerical derivatives are native plans. A plan fixes the shape, scheme and
step once; `evaluate(function, x, f_x=None)` samples an ordinary Python
callable or a native callback adapter and returns a copied estimate with its
status and residual-call count. Solvers accept the same plans as derivative
providers, and use them automatically when a derivative argument is `None`.

```python
from cyroot.native import (ScalarFiniteDifferencePlan, RectangularJacobianPlan,
                           RectangularHessianPlan, TensorDifferencePlan)

first = ScalarFiniteDifferencePlan(1, 'central', .01).evaluate(lambda x: x*x, 2.)
assert abs(first.derivative - 4.) < 1e-9
jacobian = RectangularJacobianPlan(2, 1, 'central', .01).evaluate(
    lambda x: (x[0]*x[1],), (2., 3.))
assert max(abs(a-b) for a, b in zip(jacobian.jacobian[0], (3., 2.))) < 1e-9
hessian = RectangularHessianPlan(2, 1).evaluate(lambda x: (x[0]*x[0]*x[1],), (2., 3.))
third = TensorDifferencePlan(2, 1, 3).evaluate(lambda x: (x[0]*x[0]*x[1],), (2., 3.))
```

| Derivative | Plan | Result shape |
| --- | --- | --- |
| scalar, any order | `ScalarFiniteDifferencePlan(order, scheme, step)` | float |
| square Jacobian | `JacobianPlan(n, scheme, step)` | `(n, n)` |
| rectangular Jacobian | `RectangularJacobianPlan(n, m, scheme, step)` | `(m, n)` |
| square Hessian | `HessianPlan(n, scheme, step)` | `(n, n, n)` |
| rectangular Hessian | `RectangularHessianPlan(n, m, scheme, step)` | `(m, n, n)` |
| order three and above | `TensorDifferencePlan(n, m, order, ...)` | `(m,) + (n,) * order` |
| Jacobian-vector product | `JacobianVectorPlan(n, m, scheme, step)` | `(m,)` |

The first tensor axis identifies the output component. Schemes are
`'forward'`, `'backward'` and `'central'`. `step=None` chooses a relative
automatic step; an explicit step must be finite and nonzero (strictly positive for directional
`JacobianVectorPlan`, which normalizes its direction). Samples use the
actual rounded nodes, so steps that collapse in floating point report a
breakdown instead of a silently wrong estimate. Plans bound their scratch
storage and stencil size before allocating.

For estimates with an error estimate, use the adaptive Richardson providers
`AdaptiveDerivative`, `AdaptiveSecondDerivative`, `AdaptiveJacobian` and
`AdaptiveHessian`, or the complex-step `ComplexStepDerivative` and
`ComplexStepJacobian` when the residual has a holomorphic complex extension.

## Migrating From 1.X

The 1.x `finite_difference`, `FiniteDifference`, `generalized_finite_difference`
and `GeneralizedFiniteDifference` wrappers, their `DerivativeApproximation`
base classes, and the `cyroot.fptr` callback layer were removed in 2.0. Use the
plan for the derivative's order and shape from the table above. The removed
`kind=1/-1/0` selectors correspond to `'forward'`, `'backward'` and `'central'`.
