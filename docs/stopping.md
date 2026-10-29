# Stopping Rules And Result Status

Each solver result has a `termination_reason`, a string-valued
`TerminationReason` enum. `success` and `converged` are true for residual or
bracket tolerance. `optimal` is true only when the residual tolerance is met.
A small step with an unsatisfied residual is stagnation and is not success.

```python
from cyroot import newton, TerminationReason

result = newton(lambda x: x*x - 2, lambda x: 2*x, 1.)
assert result.termination_reason == TerminationReason.RESIDUAL_TOLERANCE
assert result.success
```

The existing tolerance names have these meanings:

| Argument | Meaning |
| --- | --- |
| `etol` | Absolute residual tolerance. |
| `ertol` | Relative residual tolerance against a fixed starting residual scale. |
| `ptol` | Absolute step or bracket-width tolerance. |
| `prtol` | Relative step or bracket-width tolerance against the iterate scale. |

Residual convergence uses

\[
\|F(x_k)\|_\infty \le \mathrm{etol} + \mathrm{ertol}\,s_F.
\]

For one starting point, `s_F` is its residual norm. For multiple starting
guesses or a scalar bracket, it is the largest starting residual norm. This
reference remains fixed throughout that solve. A starting exact root is
accepted even if both residual tolerances are zero.

Open methods test step size using

\[
\|x_k-x_{k-1}\|_\infty \le \mathrm{ptol} + \mathrm{prtol}
\max(\|x_k\|_\infty,\|x_{k-1}\|_\infty).
\]

Methods that retain several iterates use the diameter of that set. A scalar
bracket uses its width and the largest endpoint magnitude. Complex methods
use complex magnitudes and geometric distances, not differences of magnitudes.
The raw step/width and residual metrics remain available as `precision` and
`error`.

A narrow valid bracket can satisfy the bracket criterion with a large residual.
Its interpretation depends on the method's assumptions, including continuity
and a valid bracket; numerical sign changes alone do not prove continuity.
Vrahatis accepts width convergence for a scalar output. For multiple outputs,
its vertex sign patterns do not certify a root, so width-only termination is
unsuccessful stagnation; residual convergence is required for success.
Rectangular systems may use least-squares steps, but their success criterion
still concerns the residual, rather than least-squares stationarity alone.

Every solver family shares the defaults
`etol=1e-10`, `ertol=4*eps`, `ptol=1e-12`, `prtol=4*eps` and `max_iter=200`
(binary64 `eps`), defined once in `cyroot/defaults.hpp`. The external-optimizer
bridge uses the same residual defaults; the external optimizer keeps its own
iteration and convergence controls. Method-specific controls,
such as optimizer `gtol` or evaluation budgets, keep their own defaults. These
controls are keyword-only in every public function and native solver class;
only the function, derivative providers, starting values and cached initial
values are positional.

The relative residual term uses the initial residual magnitude, not an estimate
of derivative accuracy. For example, with a starting residual of `1e20`, `4*eps`
contributes about `8.9e4` to the acceptance threshold. Set `ertol=0` when the
application requires an absolute residual target. Relative step tolerance also
scales with coordinate magnitude; rescale poorly balanced problems deliberately.

Tolerances must be finite and nonnegative. Relative tolerances must be smaller
than one. `max_iter` is a nonnegative integer; zero or `None` removes the
iteration limit. Disabling both the limit and every tolerance is rejected.
All methods still stop on detected nonfinite arithmetic, breakdown or a
collapsed step. Python callback exceptions propagate with their traceback.

Derivative methods evaluate only derivatives needed for another step. They
skip derivative callbacks at an already accepted starting root, and after a
step meets a stopping rule or exhausts the iteration budget. `df_root` uses
NaN placeholders, with the usual scalar or tensor shape, when a derivative
at the returned point was not evaluated. Explicitly supplied initial
derivatives are retained on an initial return. Evaluation counts reflect
these skipped calls. Quasi-Newton matrices are evolving approximations;
they are not exact derivatives at the returned point.

| Reason | Meaning |
| --- | --- |
| `residual_tolerance` | The residual criterion is satisfied. |
| `bracket_tolerance` | A valid bracket satisfies its width criterion. |
| `stagnation` | Steps or retained iterates have collapsed without residual convergence. |
| `max_iterations` | The iteration budget was exhausted. |
| `zero_derivative` | A required scalar derivative is zero. |
| `breakdown` | A denominator, factorization or other numerical precondition failed. |
| `nonfinite` | A required point, value or arithmetic result is not finite. |
| `callback_failure` | A native callback returned a nonzero status in the native core. |
| `line_search_failure` | No acceptable descent step was found by the bounded line search. |
| `linear_failure` | The required inner linear solve did not meet its acceptance criterion. |
| `max_evaluations` | The residual evaluation budget was exhausted. |
| `derivative_breakdown` | A numerical derivative/product could not form a valid estimate. |
| `stationary_nonroot` | The optimizer's stationarity test was met while the residual remained nonzero. |
| `no_acceptable_step` | A bounded optimizer trial search found no acceptable step. |
| `optimizer_stopped_without_root` | An external optimizer returned a point that failed the root tolerance. |

Interval splitting results contain `termination_reasons` per returned root and
`search_termination_reason` for the overall search. A completed search can
return no roots when interval bounds exclude them. `max_split_iterations`
means the search was incomplete, even when some returned roots converged.
Their `success` requires a completed search and successful returned roots.

Results constructed manually without explicit status use `unspecified`.
Supplying a known reason determines the corresponding success flags.
