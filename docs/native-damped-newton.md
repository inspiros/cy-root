# Damped Newton

Damped Newton computes a Newton direction, then shortens it until the residual
objective decreases enough. It can recover from an overly aggressive full step;
it does not guarantee convergence from every starting point.

## Scalar And Vector Interface

```text
DampedNewton(function, derivative, x0, f_x0=None, df_x0=None, *,
             c1=1e-4, contraction=0.5, max_line_search=32, **options)
VectorDampedNewton(function, jacobian, x0, f_x0=None, jac_x0=None, *,
                   c1=1e-4, contraction=0.5, max_line_search=32, **options)
```

Both classes are exported from `cyroot` and `cyroot.native`. The functions
`damped_newton(function, derivative, x0, **options)` and
`generalized_damped_newton(function, jacobian, x0, **options)` run a solve to
completion. Pass `None` for a native numerical derivative, or any Python or
native provider accepted by [Newton](native-newton.md) and
[Vector Newton](native-vector.md).

```python
import math
from cyroot import damped_newton, generalized_damped_newton
from cyroot.native import DampedNewton

f = lambda x: x*x - 2.
solver = DampedNewton(f, lambda x: 2*x, .1, history=32)
first = solver.step()
assert solver.step_scale == .125
assert solver.backtracks == 3
answer = solver.run()
assert answer.success
assert abs(answer.root - math.sqrt(2.)) < 1e-9
assert solver.history[-1].root == answer.root

answer = generalized_damped_newton(
    lambda x: (x[0] - 1., x[1] - 2., x[0] + x[1] - 3.),
    None, (0., 0.),
)
assert answer.success
assert max(abs(a-b) for a, b in zip(answer.root, (1., 2.))) < 1e-9
```

The vector shape comes from the point and residual, or from an adapter's
declaration. Square systems take an LU step and tall and wide systems a
rank-aware least-squares step. A singular square factorization is a breakdown.
The reported rank describes the linear step; it says nothing about whether a
root exists or is unique.

## Armijo Acceptance

For \(\phi(x)=\frac12\|F(x)\|_2^2\), a step \(p\) is accepted when

\[
\phi(x+\alpha p)-\phi(x)
\leq c_1\alpha F(x)^\mathsf{T}J(x)p
\]

and the objective strictly decreases. The search starts at \(\alpha=1\) and
multiplies it by `contraction` after each rejected candidate. The test uses the
actual slope \(F^\mathsf{T}Jp\), which matters for inexact rectangular steps.
Scaled accumulation avoids needless intermediate overflow, but cannot fix an
ill-conditioned problem or inaccurate derivatives.

`c1` and `contraction` must be finite and strictly between zero and one.
`max_line_search` is the positive maximum number of candidates per iteration.
`options` passes the [shared stopping controls](stopping.md) and the base
solver's storage and history controls. All controls are keyword-only.

## Results And Failures

`step()`, `run()`, `result`, iteration and copied bounded histories work as in
the base Newton classes. The solver's `step_scale` and `backtracks` properties
describe its latest line search. Histories contain accepted states and terminal
diagnostics, not rejected trial points; residual call counts include every
evaluated trial.

Nonfinite trial points or residuals are rejected. A Python exception or a
nonzero native callback status ends the solve at once, keeping the last
accepted point and residual. When a derivative fails after an accepted step,
that accepted point is kept. Python exceptions propagate as the original
instance; native failures raise `NativeCallbackError` with their stage and
status.

Only `residual_tolerance` counts as a root. A direction that does not descend,
or a search that runs out of candidates, reports `line_search_failure`; a step
that rounds to zero reports `stagnation`. The base solver's reasons, such as
`zero_derivative`, `breakdown` and `nonfinite`, can also occur. Calling a
finished solver does not resume it.

## Native Execution

With native residual and derivative callbacks, the numerical loop and the line
search run without the GIL; Python callbacks take it for each call. Storage is
allocated before stepping, and setup and the copied result and history export
run in Python. The base classes' callback reservation, ownership and
concurrency rules apply; see [Native Core](native-core.md).
