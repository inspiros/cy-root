# Native Secant

`cyroot.native.Secant` finds a scalar root from two starting points. The
residual can be a Python callable or the native `ScalarCallback` used by
bisection and Newton. It needs no derivative and no sign-changing bracket.

```python
from cyroot.native import Secant

solver = Secant(lambda x: x*x - 2., 1., 2., history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

`step()` returns an immutable `SecantStep` with the current and previous
point/residual pairs. `run()` and the `result` property return `RootReturnType`.
`cyroot.native.secant(function, x0, x1, **options)` runs a solver to
completion; the public `cyroot.secant(f, x0, x1, f_x0=None, f_x1=None, *, ...)`
runs the same engine.

## Evaluations And Stopping

The [common stopping rules](stopping.md) apply. When neither start is a root,
the residual scale is the larger initial absolute residual. The accepted state
starts at `x1`; the distance between the two starting points never satisfies
the step tolerance on its own. Later step metrics use consecutive accepted
points. A candidate equal to the current point is still evaluated before the
solver reports residual success or stagnation. Equal residuals stop with
`breakdown` before another callback. Residuals are normalized before their
difference is formed; any other unrepresentable step arithmetic stops with
`nonfinite` before the trial call.

The starting points are checked in order. If `x0` is already a root, `x1` is
neither evaluated nor validated, although every scalar argument is still
converted before native execution starts.

`f_calls` counts this solve's residual calls, including failed ones. `f_x0` and
`f_x1` supply cached values and skip the corresponding starting calls. An
iteration is one attempt to evaluate the residual at a finite candidate.
Nonfinite arithmetic before that call, and a breakdown from equal residuals,
use no iteration and no residual call. `max_iter` is shared by `step()` and
`run()`; zero or `None` removes the limit. A failed or nonfinite trial keeps
both previous pairs and their metrics and updates only the counts and the
reason.

A Python exception propagates as the original instance, with the callback's
traceback. A nonzero native status raises `NativeCallbackError` with
`native_status`, a copied `result` and `callback_stage='function'`; the solver
exposes the same stage and code. Numerical failures return ordinary
unsuccessful results.

## Inspection And Ownership

`history`, `history_every`, `history_bytes`, `history_truncated`, `done` and
iteration follow the [native core conventions](native-core.md). History is
bounded and allocated before native work starts. A terminal failure in the same
iteration updates that iteration's last record. Exported snapshots and results
are copies, so later steps or destroying the solver cannot change them.

The residual adapter and its context stay alive and reserved through setup,
execution, result conversion and exception transfer. Reentrant or concurrent
use of an active solver or adapter is rejected. Separate adapters that share
mutable native context still need synchronization by their owner. With a native
residual, initialization and the whole numerical run happen outside the GIL; a
Python callback takes the GIL for its call and result conversion. See the
[compiled callback example](../examples/native_secant.md).

Secant has no bracket, damping or line search, and can fail or diverge from
unsuitable starting points.
