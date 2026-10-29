# Native Householder

`cyroot.native.Householder` is the scalar Householder method for any
derivative order chosen at run time, with Python or compiled callbacks,
resumable execution and copied history. Given derivatives up to order \(d\),
it converges with local order \(d+1\) near a smooth simple root. It uses a
native reciprocal-series recurrence, so there is no symbolic formula generator
and no upper limit on the order.

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
`householder(function, derivatives, x0, **options)` constructs a solver and
runs it to completion. Callback ownership follows the
[native core contract](native-core.md).

## Derivative Providers And Caches

`derivatives` is a non-empty list or tuple, starting with the first
derivative. Each entry can be a Python function, a native `ScalarCallback`,
`None` for a numerical approximation, or a `ScalarFiniteDifferencePlan` of the
matching order; `ScalarDerivativePlan` and `ScalarSecondDerivativePlan` are
also accepted for orders one and two. Order checks apply to plan subclasses
too, and are made even when the starting point is already a root.

`derivatives=None` requires a positive integer `order` and uses numerical plans
for every order. With an explicit sequence, `order` is optional and must match
its length. `step` sets the step of the default numerical plans; the native
class ignores it when there are none, while the public `cyroot.householder`
rejects it. Explicit plans keep their own steps. The default is an automatic
central stencil; see [runtime-order derivatives](native-higher-derivative.md)
for rounded nodes, resource limits and accuracy. Different derivative orders
share no samples except the cached residual at the current point.

`f_x0` caches the initial residual. `dfs_x0` has exactly one entry per
derivative, with `None` for one that has not been evaluated. Cached values must
belong to the starting point and functions, and count as zero calls. All
supplied values are converted during setup. If the starting point is already a
root, unused derivatives are not checked for finiteness and the supplied
caches are kept. Otherwise derivatives are evaluated in increasing order. A
zero first derivative is valid for the general recurrence; a zero
reciprocal-series denominator reports `breakdown`.

## Native Execution And Budgets

The [reciprocal recurrence](native-higher-derivative.md#householder-without-symbolic-formulas)
costs \(O(d^2)\) arithmetic and \(O(d)\) workspace per step. Intermediate
values are stored as binary64 mantissas with separately checked 64-bit binary
exponents, which avoids many overflows but neither increases precision nor
makes high-order derivatives stable. Analytic derivatives have no order limit.
Numerical orders are limited by `max_nodes`, scratch memory and representable
finite arithmetic; a larger budget does not make them more accurate.

Setup keeps every callback and plan owner alive, copies the native dispatch
descriptors, and allocates workspace and history once. `workspace_bytes` covers
values, counters, roles, cache flags, the recurrence and the largest numerical
scratch. `history_bytes` separately covers the metadata, all derivatives and
all counters of every requested record. Both limits are checked before native
allocation; Python owners and exported result objects use additional memory.

The numerical loop creates no Python objects or memoryviews, does no NumPy
operations and allocates nothing per step. With compiled `noexcept nogil`
callbacks, it runs entirely in native code. Python callbacks take the GIL for
their calls and scalar conversion, as do setup and result and history export.
Adapters stay reserved through conversion, execution, result conversion and
exception transfer. Separate adapters that share a native context need
synchronization by the context's owner. See the
[compiled example](../examples/native_householder.md).

## Results, History And Failure State

`step()` and `run()` share one iteration budget; `max_iter=0` or `None` is
unlimited. The [shared stopping rules](stopping.md) apply. A candidate equal to
the current point is still evaluated before success or stagnation is decided.
There is no fallback, damping or symbolic recovery.

`result` is a `NewtonMethodsReturnType`: `df_root` has \(d\) entries and
`f_calls` has \(d+1\), residual first. Each plan evaluation counts as one
attempt for its derivative order and adds every stencil sample to the residual
count; failed attempts count too. Accepting a finite candidate clears the
derivatives, and the stopping rules and iteration limit are checked before more
derivatives are evaluated. Derivatives that are not available are NaN.

`HouseholderStep` is immutable and contains `root`, `f_root`, `df_root`, `iters`,
`f_calls`, `precision`, `error`, `termination_reason`, `callback_code`,
`callback_stage` and `callback_order`. `history` returns copied immutable
steps, and `history_truncated` reports that the capacity ran out. Initial,
sampled and terminal states follow the
[native history rules](native-core.md#step-inspect-resume).

A failed trial residual keeps the previous finite point, derivatives and
metrics, and updates only the counts and diagnostics. A failure in derivative
order \(k\) keeps the lower-order derivatives and clears orders \(k\) to \(d\).
A nonzero native status raises `NativeCallbackError` with the exact status, a
copied result, the stage `'function'` or `'derivative'`, and `callback_order`:
zero for the residual, or \(k\) for a failure in derivative \(k\) or its
stencil. Python exceptions propagate as the original instance, with its
traceback. Once raised, an error is not raised again by later `step()` or
`run()` calls on the finished solver.

## Public Entry Point

`cyroot.householder(f, dfs, x0, ...)` runs this engine. Its `dfs`, `d` and `h`
arguments correspond to `derivatives`, `order` and `step`. `h=None` selects an
automatic step, and a supplied `d` must match the length of `dfs`. The 1.x
`c_code` option, explicit reciprocal formulas, expression evaluator and
symbolic factory are gone, and SymPy is no longer a dependency. Use
`cyroot.native.Householder` for stepping and history.
