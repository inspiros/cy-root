# Native Newton

`cyroot.native.Newton` uses the same scalar callback interface and stepping,
budget and history conventions as [native bisection](native-core.md). The
residual is required. An analytic derivative may be a Python callable or a
native `ScalarCallback`; the two modes can be mixed. Pass `None` or an explicit
[ScalarDerivativePlan](native-scalar-derivative.md) for numerical derivatives.

```python
from cyroot.native import Newton

solver = Newton(lambda x: x*x - 2., lambda x: 2*x, 1.5, history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

`step()` returns an immutable `NewtonStep`; `run()` and the `result` property
return `NewtonMethodsReturnType`. `newton(function, derivative, x0, **options)`
is the convenience full-run function in `cyroot.native`. Existing
`cyroot.newton` remains the older API.

## Evaluation And Termination

The tolerance arguments and formulas are in [stopping rules](stopping.md).
The residual scale is fixed at the initial absolute residual. Step size uses
the actual difference between consecutive finite points, with their maximum
absolute magnitude for relative step tolerance. A small step without a small
residual is unsuccessful stagnation.

`f_calls` is the pair `(residual_calls, derivative_calls)` for this solve.
Numerical plans include stencil samples in residual counts and count one
derivative attempt per approximation. Stencil samples do not add iterations.
Supplying `f_x0` or `df_x0` skips the corresponding initial evaluation. Cached
values must match their callbacks. Call attempts that raise or return a native
failure code are counted. One iteration is an attempted finite candidate
residual evaluation; nonfinite arithmetic before that call does not increase
`iters`.

Initial residual convergence is checked before evaluating or validating an
unused derivative. On a finite new point/residual, the solver checks residual,
step and iteration budget before evaluating the next derivative. `df_root` is
NaN when the derivative at the returned point was not evaluated. Explicitly
supplied initial derivative values are retained on an initial accepted root.
A required zero derivative stops with `zero_derivative`; nonfinite derivatives
stop with `nonfinite`.

A failed or nonfinite residual trial preserves the previous finite point,
residual, derivative and metrics, with updated attempt counts and reason. If a
derivative fails after a finite candidate was accepted, the new point/residual
remain valid and `df_root` is NaN. No stale derivative from the previous point
is presented as a derivative at the new point.

Python exceptions preserve their original instance and callback traceback.
A nonzero native code raises `NativeCallbackError`, whose `native_status`,
`callback_stage` (`'function'` or `'derivative'`) and copied `result` identify
the failure. The solver exposes the same stage through `callback_stage` and
the code through `callback_code`. Its result reason is `callback_failure`.
Numerical failures return ordinary unsuccessful results.

## History And Callback Lifetimes

`history`, `history_every`, `history_bytes`, `history_truncated`, `done` and
iteration over the solver behave as in native bisection. Each record includes
`df_root`, the pair of evaluation counts, and callback failure stage/code.
History is allocated before native work, records initial/sample/terminal
states, and exports copied immutable tuples. A failure before any new
residual attempt updates the last record for that iteration rather than
adding a duplicate. Saved earlier exports are unchanged.

Analytic callback adapters are reserved together for setup, each step, and a
full run, including exception transfer and step/run result construction. With
a numerical plan, the residual adapter stays reserved throughout sampling.
A single analytic adapter can fill both roles.
Failed acquisition of an active second adapter does not reserve the first.
Reentrant/concurrent access to the active solver or either adapter is rejected.
Separate adapters that share native context memory still require producer
synchronization. Context owners remain pinned until all adapters/solvers using
them are released.

Native callbacks use the same versioned C/Cython descriptor as bisection.
A compiled residual and compiled analytic derivative allow the full numerical
run to remain outside the GIL. A Python callback reacquires the GIL for its
call and conversion. See the [compiled example](../examples/native_scalar.md)
and its `solve_newton()` function for the installed-package build workflow.

Plain Newton takes a full step. For Armijo backtracking use
[Damped Newton](native-damped-newton.md). [Halley and Chebyshev](native-higher-order.md)
accept analytic or numerical first and second derivatives.
[Vector Newton](native-vector.md) supports square and rectangular systems.
NumPy and SciPy remain package dependencies; native cancellation is not provided.
