# Stopping Rules And Result Status

Every solver result has a `termination_reason`, a string-valued
`TerminationReason` enum. `success` and `converged` are true when the residual
or bracket tolerance is met, and `optimal` only when the residual tolerance is
met. A small step with an unsatisfied residual is stagnation, not success.

```python
from cyroot import newton, TerminationReason

result = newton(lambda x: x*x - 2, lambda x: 2*x, 1.)
assert result.termination_reason == TerminationReason.RESIDUAL_TOLERANCE
assert result.success
```

The tolerances are:

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

For one starting point, `s_F` is its residual norm; for several starting
points or a scalar bracket, it is the largest starting residual norm. It stays
fixed for the whole solve. A starting point that is an exact root is accepted
even when both residual tolerances are zero.

Open methods test step size using

\[
\|x_k-x_{k-1}\|_\infty \le \mathrm{ptol} + \mathrm{prtol}
\max(\|x_k\|_\infty,\|x_{k-1}\|_\infty).
\]

Methods that keep several iterates use the diameter of that set. A scalar
bracket uses its width and the larger endpoint magnitude. Complex methods use
complex magnitudes and distances, not differences of magnitudes. The step or
width and the residual norm are reported as `precision` and `error`.

A narrow valid bracket can meet the bracket criterion while the residual is
still large. Whether it encloses a root depends on the method's assumptions,
such as continuity; sign changes observed numerically do not prove continuity.
Vrahatis accepts width convergence when there is one output. With several
outputs, its vertex signs do not certify a root, so stopping on width alone is
unsuccessful stagnation and success requires residual convergence. Rectangular
systems can take least-squares steps, but success still depends on the
residual, not on least-squares stationarity.

All solver families share the defaults
`etol=1e-10`, `ertol=4*eps`, `ptol=1e-12`, `prtol=4*eps` and `max_iter=200`
(with the binary64 `eps`), defined once in `cyroot/defaults.hpp`.
`root_via_optimizer` uses the same residual defaults, while the external
optimizer keeps its own iteration and convergence controls. Method-specific
controls, such as the optimizers' `gtol` or evaluation budgets, have their own
defaults. All of these controls are keyword-only in every public function and
native solver class; only the function, derivative providers, starting values
and cached starting values are positional.

The relative residual term scales with the starting residual, not with the
accuracy of the derivatives. For example, with a starting residual of `1e20`,
`4*eps` adds about `8.9e4` to the threshold. Set `ertol=0` when the application
needs an absolute residual target. The relative step tolerance also scales with
the size of the coordinates, so rescale badly balanced problems on purpose.

Tolerances must be finite and nonnegative, and relative tolerances less than
one. `max_iter` is a nonnegative integer; zero or `None` removes the iteration
limit. Disabling both the limit and every tolerance is rejected. Every method
still stops on nonfinite arithmetic, breakdown or a collapsed step. Python
callback exceptions propagate with their traceback.

Derivative methods evaluate only the derivatives needed for another step. They
make no derivative calls when the starting point is already a root, or after a
step that meets a stopping rule or uses up the iteration budget. When the
derivative at the returned point was not evaluated, `df_root` holds NaN
placeholders of the usual scalar or tensor shape, except that supplied initial
derivatives are kept when the starting point is returned. Evaluation counts
reflect the skipped calls. Quasi-Newton matrices are evolving approximations,
not the derivatives at the returned point.

| Reason | Meaning |
| --- | --- |
| `residual_tolerance` | The residual criterion is satisfied. |
| `bracket_tolerance` | A valid bracket satisfies its width criterion. |
| `stagnation` | Steps or retained iterates have collapsed without residual convergence. |
| `max_iterations` | The iteration budget was exhausted. |
| `zero_derivative` | A required scalar derivative is zero. |
| `breakdown` | A denominator, factorization or other numerical precondition failed. |
| `nonfinite` | A required point, value or arithmetic result is not finite. |
| `callback_failure` | A callback raised a Python exception or returned a nonzero native status; the error is also raised. |
| `line_search_failure` | No acceptable descent step was found by the bounded line search. |
| `linear_failure` | The required inner linear solve did not meet its acceptance criterion. |
| `max_evaluations` | The residual evaluation budget was exhausted. |
| `derivative_breakdown` | A numerical derivative/product could not form a valid estimate. |
| `stationary_nonroot` | The optimizer's stationarity test was met while the residual remained nonzero. |
| `no_acceptable_step` | A bounded optimizer trial search found no acceptable step. |
| `optimizer_stopped_without_root` | An external optimizer returned a point that failed the root tolerance. |

Interval-splitting results have `termination_reasons`, one per returned root,
and a `search_termination_reason` for the whole search. A completed search can
return no roots when the interval bounds exclude them. `max_split_iterations`
means the search did not complete, even if some returned roots converged.
`success` requires a completed search and successful returned roots.

A result constructed by hand without a reason reports `unspecified`; passing a
known reason sets the matching success flags.
