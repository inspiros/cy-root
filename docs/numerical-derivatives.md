# Numerical Derivatives

CyRoot provides fixed finite differences, adaptive Richardson refinement, and
complex-step derivatives. These are explicit providers: pass one in the ordinary
derivative or Jacobian position of a solver. They do not automatically
differentiate arbitrary Python code.

## Choose A Provider

| Provider | Derivatives | Requirement |
| --- | --- | --- |
| Fixed finite-difference plans | Scalar runtime order, vector Jacobian/Hessian/higher mixed partials | Real residual samples and an appropriate step/stencil budget. |
| `AdaptiveDerivative`, `AdaptiveSecondDerivative` | Scalar first/second | A smooth real function and a converged refinement sequence. |
| `AdaptiveJacobian`, `AdaptiveHessian` | Square/rectangular first/second | Smooth residual components; full mixed Hessian entries are included. |
| `ComplexStepDerivative`, `ComplexStepJacobian` | Scalar first / vector Jacobian | A holomorphic complex extension of the real residual. |
| `JacobianVectorPlan` | Directional first | A direction, without storing the full Jacobian. |

[Finite Differences](finite-differences.md) lists all fixed plans and output
shapes. [Jacobian Products](jacobian-vector-products.md) describes directional
steps and their different sampling costs.

## Adaptive Richardson Refinement

```python
import math
from cyroot import AdaptiveDerivative, AdaptiveSecondDerivative, newton, halley

f = lambda x: x*x - 2.
first = AdaptiveDerivative(f)
second = AdaptiveSecondDerivative(f)
estimate = first.evaluate(1.5)
assert estimate.status == 'ok'
assert abs(estimate.derivative - 3.) < 1e-8
assert estimate.error is not None
assert newton(f, first, 1.5).success
assert halley(f, first, second, 1.5).success
```

Refinement compares central estimates at progressively smaller steps and uses
Richardson extrapolation to reduce truncation error. `step`, `atol`, `rtol`,
and `max_steps` bound the initial scale and refinement. Reported `error` values
are estimates, not rigorous error bounds. Noise, discontinuities, cancellation,
or rounding can prevent convergence. `.evaluate(...)` returns the diagnostic
estimate; calling the provider as a derivative raises `DifferentiationError`
when the estimate did not converge. Solvers therefore do not silently consume
an unconverged estimate. `last_estimate` retains copied diagnostics.

```python
from cyroot import AdaptiveJacobian, AdaptiveHessian, generalized_newton

f = lambda x: (x[0]*x[0] + x[1] - 3., 2*x[0] + x[1]*x[1] - 6.)
jacobian = AdaptiveJacobian(f, 2, 2)
hessian = AdaptiveHessian(f, 2, 2)
estimate = hessian.evaluate((1., 2.))
assert estimate.status == 'ok'
assert len(estimate.hessian) == 2
assert abs(estimate.hessian[0][0][0] - 2.) < 1e-6
assert generalized_newton(f, jacobian, (1.4, 1.7)).success
```

For `n` inputs and `m` residuals the Jacobian shape is `(m,n)` and the Hessian
shape is `(m,n,n)`. Hessian refinement includes mixed partials; diagonal entries
alone would not supply the tensor expected by higher-order vector methods.
Dense storage and sampling can be substantial. Vector providers take a checked
`memory_bytes` budget; Python callback allocations are additional.

## Complex-Step

```python
import cmath
import math
from cyroot import ComplexStepDerivative, ComplexStepJacobian

derivative = ComplexStepDerivative(cmath.exp)
estimate = derivative.evaluate(.4)
assert estimate.status == 'ok'
assert abs(estimate.derivative - math.exp(.4)) < 1e-12
assert estimate.error is None

jacobian = ComplexStepJacobian(lambda x: (x[0]*x[0] + x[1],), 2, 1)
estimate = jacobian.evaluate((2., 3.))
assert estimate.status == 'ok'
assert max(abs(a-b) for a, b in zip(estimate.jacobian[0], (4., 1.))) < 1e-12
```

For an analytic real-valued function extended to complex arguments,
\(f'(x)\approx\operatorname{Im}f(x+ih)/h\). This avoids subtracting nearby
real values, but only when the function preserves the necessary complex
information. `abs`, real-part extraction, comparisons, conjugation, and
nonanalytic branches can invalidate it. Use `cmath` or complex-compatible
operations. No error estimate is supplied, and a finite answer alone does not
validate the holomorphic-extension assumption.

## Native Callbacks

These providers accept their matching native residual adapters as well as
ordinary Python functions. Numerical kernels run without the GIL when the
residual callbacks are native; Python callbacks acquire it for every sample.
Construction, input packing, and result export are Python boundaries. Sharing
an immutable plan does not make an active callback adapter reentrant; keep
independent mutable states and contexts for concurrent solves.
