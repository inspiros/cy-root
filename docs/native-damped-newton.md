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

Both classes are exported from `cyroot` and `cyroot.native`. The one-shot public
functions are `damped_newton(function, derivative, x0, **options)` and
`generalized_damped_newton(function, jacobian, x0, **options)`. Use `None` for a
native numerical derivative, or supply the same Python/native providers accepted
by [Newton](native-newton.md) and [Vector Newton](native-vector.md).

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

Vector shape is inferred from the point and residual, or declared by an adapter.
Square systems use an LU step; tall and wide systems use rank-aware least
squares. A singular square factorization is a breakdown. Rectangular rank
information describes the linear step, not root existence or uniqueness.

## Armijo Acceptance

For \(\phi(x)=\frac12\|F(x)\|_2^2\), a step \(p\) is accepted when

\[
\phi(x+\alpha p)-\phi(x)
\leq c_1\alpha F(x)^\mathsf{T}J(x)p
\]

and the objective strictly decreases. The search starts at \(\alpha=1\) and
multiplies it by `contraction` after each rejected candidate. It uses the actual
slope \(F^\mathsf{T}Jp\), including for inexact rectangular steps. Scaled
accumulation avoids unnecessary intermediate overflow; it cannot repair an
ill-conditioned problem or inaccurate derivatives.

`c1` and `contraction` must be finite and strictly between zero and one.
`max_line_search` is a positive bound on candidate attempts per iteration.
`options` forwards the [shared stopping defaults](stopping.md) and the base
solver's storage/history controls. All controls are keyword-only.

## Results And Failures

`step()`, `run()`, `result`, iteration, and copied bounded histories follow the
base Newton classes. `step_scale` and `backtracks` are properties of the solver
for its latest search. Histories contain accepted states and terminal diagnostics,
not rejected trial points. Residual call counts include evaluated trials.

Nonfinite trial points or residuals are rejected. A Python exception or nonzero
native callback status terminates the solve immediately, preserving the last
accepted point/residual. A failed derivative may follow an accepted step and
retains that accepted point. Exceptions preserve their original identity;
native failures raise `NativeCallbackError` with their stage and status.

Only `residual_tolerance` is root success. A non-descent direction or exhausted
search reports `line_search_failure`; a rounded zero step reports `stagnation`.
The base numerical/linear reasons, including `zero_derivative`, `breakdown`, and
`nonfinite`, remain available. Terminal calls do not resume the solve.

## Native Execution

Native residual and derivative callbacks allow the numerical loop and line
search to run without the GIL. Python callbacks acquire it per sample. Storage
is prepared before stepping; setup and copied result/history export use Python.
The base callback reservation, ownership, and concurrency contracts apply;
see [Native Core](native-core.md).
