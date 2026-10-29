# Levenberg–Marquardt root finding

`LevenbergMarquardt` minimizes the residual objective \(\phi(x)=\frac12\|F(x)\|_2^2\).
Only meeting the original residual tolerance establishes root success. A minimum
of this objective can have a nonzero residual, for example \(F(x)=x^2+1\) at zero.

```python
from cyroot.optimization import LeastSquaresObjective, LevenbergMarquardt

objective = LeastSquaresObjective(
    lambda x: (10*(x[1]-x[0]**2), 1-x[0]), 2,
    jacobian=lambda x: ((-20*x[0], 10.), (-1., 0.)),
)
solver = LevenbergMarquardt(objective, (-1.2, 1.), history=32)
first = solver.step()
result = solver.run()
assert result.success
assert result.error <= 1e-10
steps = solver.history
```

`levenberg_marquardt(objective, x0, **options)` returns the same result without
retaining an inspectable solver. Both names are available in `cyroot.optimization`
and `cyroot.native`. The new API accepts a `LeastSquaresObjective` rather than
repeating its callback configuration. Scalar residuals use a one-element vector.
Square, overdetermined and underdetermined systems use the same interface.
User callbacks return a residual vector of length `m` and an optional Jacobian
of shape `(m,n)`. If the objective has no Jacobian, the solver uses native central
finite differences and reuses the accepted residual as its base sample. Supplied
residual Hessians are unused by this first-order model.

## Algorithm

Let \(s=\|F\|_\infty\) and \(d_j=\max_i|J_{ij}|\), using one for a zero column.
Set \(A_{ij}=J_{ij}/d_j\). Each trial solves the augmented least-squares problem

\[
\min_z\left\|\begin{bmatrix} A\\\sqrt{\lambda} I\end{bmatrix}z
 -\begin{bmatrix}-F/s\\0\end{bmatrix}\right\|_2,
\qquad \Delta x_j=s z_j/d_j.
\]

SciPy's native DGELSY supplies a rank-aware solve without forming \(J^TJ\).
Scaling products and objective reductions avoids several avoidable overflow and
underflow cases. Arithmetic remains binary64; extreme conditioning and rounding
can still prevent progress. Positive damping regularizes the augmented matrix;
its reported rank describes that matrix, **not** the undamped Jacobian.

Acceptance compares actual objective reduction to the undamped linear model's
predicted reduction, using the displacement after coordinate rounding. A trial
must decrease the objective and have reduction ratio at least `1e-4`. Rejected
trials multiply damping by ten. Accepted trials multiply damping by `.3` for a
ratio above `.75`, by two below `.25`, otherwise leave it unchanged. Damping stays
within `[1e-15,1e15]`. An outer iteration samples one Jacobian and makes at most
`max_trials` trial solves. Unaccepted trial coordinates/residuals never replace
the accepted state. A nonfinite trial is rejected; a callback exception or native
callback error stops immediately and preserves the last accepted state.

## Options and diagnostics

Defaults:

```python
LevenbergMarquardt(objective, x0,
    etol=1e-10, ertol=0., ptol=1e-12, prtol=0., gtol=1e-12,
    max_iter=100, max_evaluations=10000, max_trials=16,
    damping=1e-3, rcond=1e-12, history=0, history_every=1,
    workspace_bytes=64*1024*1024)
```

Root convergence means \(\|F\|_\infty\le\mathrm{etol}+
\mathrm{ertol}\|F(x_0)\|_\infty\), with a fixed initial relative scale.
`ertol` must be smaller than one. An accepted step within the absolute/relative
coordinate tolerances stops with `stagnation` when root tolerance is unmet.
`gtol` bounds the dimensionless model metric
\(\|A^T(F/s)\|_\infty\); it does not bound the raw objective gradient.
A small metric returns `stationary_nonroot`, an unsuccessful root result.

`result` and `history` are properties returning copied immutable tuple snapshots.
`step()` performs one outer iteration; `run()` finishes the solve. Both return
`LevenbergMarquardtResult`, with root/residual, error, precision, iteration and
callback counts, rejected-trial count, damping, model-gradient norm and statuses.
`gradient` is the metric from the **last linear model**, which may precede the
returned root. `damping` is the value prepared for the next trial. `precision`
is the last accepted displacement infinity norm, initially infinity.

`iters` counts outer iterations whose Jacobian was successfully obtained,
including a final stationary or unsuccessful iteration. `f_calls` includes the
initial sample, finite-difference samples and evaluated trials. `jac_calls` counts
attempted analytic or numerical Jacobian evaluations. `max_evaluations` bounds
residual calls only; a numerical Jacobian requires room for its complete `2*n`
sample stencil. A finite residual evaluation budget and finite inner trial count
ensure rejection cannot loop indefinitely. As for the root solvers, `max_iter=0`
or `None` removes the iteration limit; the positive evaluation budget remains.

`termination_reason` is `None` while running, then one of:
`residual_tolerance`, `stationary_nonroot`, `stagnation`, `max_iterations`,
`max_evaluations`, `no_acceptable_step`, `linear_failure`, `nonfinite`,
`callback_failure`, `derivative_breakdown`. `success` is true only for
`residual_tolerance`. `linear_status` is the native linear-plan status
(0 success, 1 nonfinite, 2 backend failure, 3 invalid plan); `linear_rank` is
`None` before a solve. Python callback exceptions retain their identity; native
errors raise `NativeCallbackError` carrying the last accepted result.

`history` specifies a maximum snapshot count; zero disables recording. The initial
point, every `history_every` outer iteration and termination are eligible for
recording. Once full, history keeps its existing entries and sets
`history_truncated`. `workspace_bytes` covers solver-owned native arrays, LAPACK
scratch, pivots and requested history. It excludes the separately reusable
objective adapter, Python snapshots and callback allocations. `memory_bytes`
reports this native allocation. The augmented row count `m+n` is limited to32767
by the current LAPACK-plan policy. No Jacobian rank or unique-root guarantee is
implied for rectangular or deficient systems.

## Compiled use and the GIL

Native callbacks through the objective run without the GIL. Python callbacks
acquire it for each call. Solver construction, result/history export and `step()`/
`run()` orchestration are Python boundaries; their numerical loops release it.
No internal NumPy arrays or NumPy operations are used.

Compiled consumers can `cimport LevenbergMarquardt` from
`cyroot.native.levenberg_marquardt`, enter `_enter()` while holding the GIL,
call `_step()` or `_run()` in `nogil`, then transfer callback errors through
`_raise_callback_error()` before `_leave()` in a `finally` block with the GIL.
The initialized state, objective and callback owners must remain alive and reserved
through all execution and exports. These trusted methods allocate no native
workspace. A state/objective/shared callback adapter cannot be used reentrantly
or concurrently; independent states with independent adapters can run in parallel.
