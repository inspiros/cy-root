# Native Secant

`cyroot.native.Secant` finds a scalar root using two starting function values.
It accepts a Python callable or the same native `ScalarCallback` used by native
bisection and Newton. It requires no derivative and no sign-changing bracket.

```python
from cyroot.native import Secant

solver = Secant(lambda x: x*x - 2., 1., 2., history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

`step()` returns an immutable `SecantStep`, including the current and previous
point/residual pairs. `run()` and the `result` property return `RootReturnType`.
`secant(function, x0, x1, **options)` is the full-run convenience function in
`cyroot.native`. Existing `cyroot.secant` remains the earlier API.

## Evaluations And Stopping

The [common stopping rules](stopping.md) apply. For two finite nonsolved starts,
the fixed residual scale is the larger initial absolute residual. The current
accepted state starts at `x1`; the initial pair distance does not satisfy a
step tolerance. Subsequent step metrics use consecutive accepted points.
A finite unchanged candidate is evaluated before reporting residual success or
unsuccessful stagnation. Equal residuals stop with `breakdown` before another
callback. The method normalizes residuals before forming their difference;
other unrepresentable step arithmetic returns `nonfinite` before a trial call.

Starting roots are checked sequentially. An accepted finite `x0` root skips
evaluation and finite-value validation of the unused second point/residual.
Otherwise the solver evaluates and checks `x1`. Inputs are still converted
before native execution, including unused scalar arguments.

`f_calls` is this solve's number of actual residual attempts. `f_x0` and `f_x1`
supply cached values and skip their corresponding starting calls. One
iteration is an attempted finite candidate residual evaluation, including
failed callback attempts. Nonfinite arithmetic before evaluation and equal
residual breakdown consume no new iteration or residual call. `max_iter`
applies to the whole solver across `step()` and `run()`; zero or `None` removes
the limit. A failed/nonfinite trial retains both prior finite pairs and their
metrics, updating only counts and the termination reason.

Python exceptions preserve their instance and callback traceback. Native
nonzero callback codes raise `NativeCallbackError`, with `native_status`, a
copied `result`, and `callback_stage='function'`. The solver exposes the same
stage and code; numerical failures return ordinary unsuccessful results.

## Inspection And Ownership

`history`, `history_every`, `history_bytes`, `history_truncated`, `done` and
iteration follow [native-core conventions](native-core.md). History is bounded
and preallocated before native work. Same-iteration terminal failures update
the matching last record. Exported snapshots and results are copied, so later
steps or solver destruction cannot change earlier exports.

The residual adapter and its context remain pinned and reserved through setup,
execution, boxing and exception transfer. Reentrant/concurrent use of an active
solver or adapter is rejected. Separate adapters sharing mutable native context
still require producer synchronization. With a native residual, initialization
and the full numerical run execute outside the GIL. Python callbacks reacquire
the GIL for calls and result conversion. See the
[compiled callback example](../examples/native_secant.md).

Secant has no bracketing safeguard, damping, or line search and can fail or
diverge for unsuitable starting pairs. NumPy and SciPy remain package dependencies.
