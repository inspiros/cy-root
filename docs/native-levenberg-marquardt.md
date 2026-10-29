# Levenberg–Marquardt Root Finding

`LevenbergMarquardt` minimizes the residual objective \(\phi(x)=\frac12\|F(x)\|_2^2\).
Only meeting the residual tolerance counts as finding a root: a minimum of this
objective can have a nonzero residual, as \(F(x)=x^2+1\) does at zero.

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

`levenberg_marquardt(objective, x0, **options)` returns the same result
without keeping a solver to inspect. Both names are available from
`cyroot.optimization` and `cyroot.native`. The solver takes a
`LeastSquaresObjective`, which holds the callbacks; a scalar residual is a
one-element vector. Square, overdetermined and underdetermined systems share
the same interface. Callbacks return a residual vector of length `m` and,
optionally, a Jacobian of shape `(m,n)`. Without a Jacobian, the solver uses
native central differences around the accepted residual. Residual Hessians are
not used by this first-order model. (`scalar_levenberg_marquardt` and
`vector_levenberg_marquardt` wrap all this in the usual
`(function, jacobian, x0)` form; see [Optimization](optimization.md).)

## Algorithm

Let \(s=\|F\|_\infty\) and \(d_j=\max_i|J_{ij}|\), using one for a zero column.
Set \(A_{ij}=J_{ij}/d_j\). Each trial solves the augmented least-squares problem

\[
\min_z\left\|\begin{bmatrix} A\\\sqrt{\lambda} I\end{bmatrix}z
 -\begin{bmatrix}-F/s\\0\end{bmatrix}\right\|_2,
\qquad \Delta x_j=s z_j/d_j.
\]

SciPy's DGELSY gives a rank-aware solve without forming \(J^TJ\). Scaling the
products and the objective reductions avoids several overflow and underflow
cases, but the arithmetic is still binary64, and extreme conditioning or
rounding can still prevent progress. Positive damping regularizes the
augmented matrix, so the reported rank describes that matrix, **not** the
undamped Jacobian.

A trial is accepted by comparing the actual reduction of the objective with
the reduction predicted by the undamped linear model, using the displacement
after rounding. It must decrease the objective with a reduction ratio of at
least `1e-4`. A rejected trial multiplies the damping by ten. An accepted trial
multiplies it by `.3` when the ratio is above `.75` and by two when it is below
`.25`, and otherwise leaves it unchanged. Damping stays within `[1e-15,1e15]`.
Each outer iteration evaluates one Jacobian and tries at most `max_trials`
solves. Rejected trials never replace the accepted state. A nonfinite trial is
rejected; a callback exception or native callback error stops the solve at once
and keeps the last accepted state.

## Options And Diagnostics

Defaults:

```text
LevenbergMarquardt(objective, x0, *,
    etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, gtol=1e-12,
    max_iter=200, max_evaluations=10000, max_trials=16,
    damping=1e-3, rcond=1e-12, history=0, history_every=1,
    workspace_bytes=64*1024*1024)
```

Root convergence means \(\|F\|_\infty\le\mathrm{etol}+
\mathrm{ertol}\|F(x_0)\|_\infty\), with a fixed initial relative scale.
`ertol` must be less than one. An accepted step within the absolute and
relative step tolerances stops with `stagnation` when the root tolerance is
not met.
`gtol` bounds the residual-relative stationarity
\(\|J^TF\|_\infty/(\|F\|_\infty\max_{ij}|J_{ij}|)\), shared with BFGS and
L-BFGS. It is invariant to residual scaling and stays of order one near simple
and multiple roots, so it does not stop a solve near a root. A small value
returns `stationary_nonroot`, an unsuccessful root result. A minimum where the
whole Jacobian vanishes (for example a scalar \(f'(x)=0\)) keeps this ratio near
one; such solves usually end as `no_acceptable_step`, `line_search_failure` or
`stagnation`, which are also unsuccessful. The `gradient` field still reports
the scaled model metric \(\|A^T(F/s)\|_\infty\).

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
reports this native allocation. The augmented row count `m+n` is limited to
32767 by the least-squares plan. Nothing is implied about the Jacobian's rank
or the uniqueness of a root for rectangular or rank-deficient systems.

## Compiled Use And The GIL

Native callbacks in the objective run without the GIL; Python callbacks take
it for each call. Construction, result and history export and the Python side of
`step()` and `run()` hold the GIL, and their numerical loops release it. No
NumPy arrays or operations are used internally.

Compiled consumers can `cimport LevenbergMarquardt` from
`cyroot.native.levenberg_marquardt`, enter `_enter()` while holding the GIL,
call `_step()` or `_run()` in `nogil`, then transfer callback errors through
`_raise_callback_error()` before `_release()` in a `finally` block with the GIL.
The initialized state, objective and callback owners must remain alive and reserved
through all execution and exports. These trusted methods allocate no native
workspace. A state/objective/shared callback adapter cannot be used reentrantly
or concurrently; independent states with independent adapters can run in parallel.
