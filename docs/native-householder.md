# Native Householder

`cyroot.native.Householder` implements runtime-order scalar Householder with
Python or compiled callbacks, resumable execution and copied history. Given
derivatives through order \(d\), the method has local order \(d+1\) under the
usual smoothness and simple-root assumptions. It uses a native reciprocal-series
recurrence, without a symbolic formula generator or a cutoff at order ten.

```python
from cyroot.native import Householder

solver = Householder(
    lambda x: x*x*x - 1.,
    (lambda x: 3*x*x, lambda x: 6*x, lambda x: 6.),
    1.2, history=16,
)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

The constructor is
`Householder(function, derivatives, x0, f_x0=None, dfs_x0=None, *, order=None,
step=None, etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=8*1024*1024, max_nodes=65)`.
`householder(function, derivatives, x0, **options)` constructs and runs one solve.
Native callback ownership follows the [native core contract](native-core.md).

## Derivative Providers And Caches

`derivatives` is an exact list or tuple of positive length, in increasing
order. Each entry accepts a Python function, a native `ScalarCallback`, `None`
for numerical approximation, or a `ScalarFiniteDifferencePlan` of the matching
order. Specialized `ScalarDerivativePlan` and `ScalarSecondDerivativePlan` are
accepted in roles one and two. Native initialization and order checks apply
also to callable plan subclasses and already converged initial points.

`derivatives=None` requires a positive integer `order` and selects numerical
plans for every role. With an explicit sequence, `order`, if supplied, must
match its length. `step` sets the default numerical plans' step and is ignored
when there are no default plans; explicit plans retain their own steps.
The default is an automatic central stencil. See
[runtime-order derivatives](native-higher-derivative.md) for rounded nodes,
resource limits and accuracy limitations. No samples are shared between
separate derivative roles except the cached base residual.

`f_x0` caches the initial residual. `dfs_x0` has exactly one entry per derivative;
`None` means unevaluated. Caches must describe the supplied initial point and
functions. Cached values count as zero calls. All supplied values are converted
during guarded setup; an initially converged residual skips unused derivative
finite checks and preserves the supplied caches. Otherwise, derivatives are
evaluated in increasing order. A zero first derivative is valid for the generic
recurrence; a zero reciprocal-series denominator reports `breakdown`.

## Native Execution And Budgets

The [reciprocal recurrence](native-higher-derivative.md#householder-without-symbolic-formulas)
costs \(O(d^2)\) arithmetic and \(O(d)\) workspace per step. Binary64 mantissas
and checked 64-bit binary exponents prevent many intermediate overflows; they
do not increase precision or guarantee stable high-order derivatives.
Analytic orders have no formula-table limit. Numerical orders are constrained
by `max_nodes`, scratch requirements and representable finite arithmetic.
Increasing a resource budget does not ensure accuracy.

Setup pins all callback and plan owners, copies native dispatch descriptors,
and allocates workspace and history once. `workspace_bytes` includes values,
counters, roles, cache flags, recurrence and the largest numerical scratch.
`history_bytes` separately includes metadata, all derivatives and all counters
for every requested record. Limits are checked before native allocation. Python
owners and exported Python result objects use additional memory.

The numerical loop creates no Python objects or memoryviews, performs no NumPy
operations and makes no per-step allocation. Compiled `noexcept nogil`
callbacks allow continuous native execution. Python callbacks reacquire the GIL
for calls and scalar conversion; setup and result/history export also use it.
Adapters stay reserved through conversion, execution, boxing and exception
transfer. Separate adapters sharing a native context require synchronization
by the context owner. See the [compiled example](../examples/native_householder.md).

## Results, History And Failure State

`step()` and `run()` share one total iteration budget. `max_iter=0` is unlimited.
The [shared stopping rules](stopping.md) apply. A finite unchanged candidate is
evaluated before deciding residual success or stagnation. No fallback, damping
or symbolic recovery is attempted.

`result` is `NewtonMethodsReturnType`: `df_root` has \(d\) entries and `f_calls`
has \(d+1\), residual first. Each derivative plan invocation counts one
attempt in its role and adds every stencil residual attempt to the residual
count. Failed attempts count. Finite accepted candidates clear derivatives;
stopping and iteration limits are checked before evaluating more derivatives.
Unavailable derivatives are NaN.

`HouseholderStep` is immutable and contains `root`, `f_root`, `df_root`, `iters`,
`f_calls`, `precision`, `error`, `termination_reason`, `callback_code`,
`callback_stage` and `callback_order`. `history` returns copied immutable steps;
`history_truncated` reports exhausted capacity. The initial, sampled and terminal
states follow the [native history rules](native-core.md#step-inspect-resume).

A failed trial residual preserves the preceding finite point, derivatives and
metrics, while updating attempt counts and diagnostics. Failure in derivative
role \(k\) preserves earlier derivatives and clears roles \(k\) through \(d\).
Native nonzero callback codes raise `NativeCallbackError`, with exact status,
copied result, stage `'function'` or `'derivative'`, and `callback_order` zero
for the residual or \(k\) for a derivative/stencil failure. Original Python
exceptions retain identity and traceback. Delivered errors are not raised again
by terminal `step()` or `run()` calls.

## Established Entry Point

`cyroot.householder` and `cyroot.scalar_newton.householder` now forward to this
native engine. Their existing `dfs`, `d` and `h` names map to `derivatives`,
`order` and `step`. `h=None` now selects automatic numerical steps; a supplied
`d` must match `dfs`. The obsolete `c_code` option, explicit reciprocal formulas,
expression evaluator and symbolic factory have been removed. SymPy is no longer
a package dependency. Use `cyroot.native.Householder` for stepping and history.
Other established solvers still use their existing implementations.
