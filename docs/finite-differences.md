# Finite differences

`finite_difference` approximates a scalar derivative; `generalized_finite_difference`
approximates a dense derivative tensor of a vector function. `order` specifies
the derivative order, not an independently selected accuracy order.

```python
import numpy as np
from cyroot import finite_difference, generalized_finite_difference

first = finite_difference(lambda x: x * x, 2., h=.01, kind='central')
# approximately 4
jacobian = generalized_finite_difference(
    lambda x: np.array([x[0] * x[1]]), [2., 3.], h=.01,
)
# approximately [[3, 2]]
```

Schemes accept `1`/`'forward'`, `-1`/`'backward'`, or `0`/`'central'`.
The scalar step must be finite and nonzero. Vector steps may be a scalar or a
one-dimensional sequence with exactly one finite, nonzero step per input
coordinate. Negative steps are supported. Inputs and callback outputs must be
finite; vector callbacks must return a fixed-length, nonempty vector.

For `n` inputs, `m` outputs and derivative order `p`, the tensor shape is
`(m,) + (n,) * p`. The first axis identifies the output component. A Jacobian
has shape `(m, n)` and a Hessian tensor has shape `(m, n, n)`.

For a partial derivative that differentiates coordinate `j` exactly `a[j]`
times, central samples in that coordinate are at
`x[j] + (a[j]/2 - i) * h[j]` for `i` from zero through `a[j]`.
Other coordinates stay fixed. Thus the first central derivative samples at
`x ± h/2`, while the second samples at `x-h`, `x`, and `x+h`.
Forward/backward schemes have one-sided truncation error; reducing the step
too far increases floating-point cancellation. Check accuracy against an
analytic derivative or a range of step sizes when the derivative is sensitive.

`FiniteDifference` and `GeneralizedFiniteDifference` are reusable callback
wrappers. `eval_with_f_val(x, value)` reuses a supplied function value, and the
functional APIs accept `f_x`/`F_x`. Shared stencil samples are evaluated once.
An uncached vector derivative first evaluates the function at the base point
to establish output size. An odd-order central scalar derivative does not need
that base evaluation. Function-call counts reflect actual callback evaluations.

Vector callbacks receive independent point snapshots and may retain or mutate
them, or reuse their own output arrays. Mutating those snapshots does not
change the original point or previously cached sample values.

Dense high-order tensors grow rapidly. The generic vector implementation
currently rejects tensor rank above 32, more than 1,000,000 stencil terms, or
an output/workspace estimate above 256 MiB. These are provisional guardrails,
not a process-wide memory guarantee. Unrepresentable sample points or arithmetic
are errors rather than valid derivative estimates.
