# Native Newton

`cyroot.native.Newton` shares the scalar callback interface and the stepping,
budget and history conventions of [native bisection](native-core.md). The
residual and an analytic derivative can each be a Python callable or a native
`ScalarCallback`, in any combination. For a numerical derivative, pass `None` or
a [ScalarDerivativePlan](native-scalar-derivative.md).

```python
from cyroot.native import Newton

solver = Newton(lambda x: x*x - 2., lambda x: 2*x, 1.5, history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

`step()` returns an immutable `NewtonStep`; `run()` and the `result` property
return `NewtonMethodsReturnType`. `cyroot.native.newton(function, derivative,
x0, **options)` runs a solver to completion. The public `cyroot.newton(f, df,
x0, f_x0=None, df_x0=None, *, h=None, ...)` runs the same engine; `h` sets the
step of the default numerical derivative.

## Evaluation And Termination

[Stopping And Results](stopping.md) defines the tolerances. The residual scale
is the initial absolute residual. The step is the actual difference between
consecutive finite points, and the relative step tolerance uses the larger of
their magnitudes. A small step without a small residual is reported as
unsuccessful stagnation.

`f_calls` is the pair `(residual_calls, derivative_calls)` for this solve.
With a numerical plan, stencil samples count as residual calls, and each
approximation counts as one derivative call; samples do not add iterations.
Supplying `f_x0` or `df_x0` skips that initial evaluation, so the cached value
must be what the callback would return. Calls that raise or return a native
failure code are counted. An iteration is one attempt to evaluate the residual
at a finite candidate; nonfinite arithmetic before that call does not increase
`iters`.

If the starting point already satisfies the residual tolerance, no derivative
is evaluated or validated. After each finite new point and residual, the solver
checks the residual, step and iteration budget before evaluating the next
derivative. `df_root` is NaN when the derivative at the returned point was not
evaluated, except that a supplied `df_x0` is kept when the starting point is
the root. A zero derivative stops with `zero_derivative`, a nonfinite one with
`nonfinite`.

A failed or nonfinite residual trial keeps the previous finite point, residual,
derivative and metrics, and updates only the call counts and the reason. If the
derivative fails after a finite candidate was accepted, the new point and
residual stand and `df_root` is NaN: a derivative from the previous point is
never reported for the new one.

A Python exception propagates as the original instance, with the callback's
traceback. A nonzero native status raises `NativeCallbackError`, whose
`native_status`, `callback_stage` (`'function'` or `'derivative'`) and copied
`result` identify the failure. The solver also exposes the stage and code as
`callback_stage` and `callback_code`, and the result reason is
`callback_failure`. Numerical failures return ordinary unsuccessful results
instead of raising.

## History And Callback Lifetimes

`history`, `history_every`, `history_bytes`, `history_truncated`, `done` and
iteration over the solver behave as in native bisection. Each record includes
`df_root`, both evaluation counts and any callback failure stage and code.
History storage is allocated before native work starts; it records the initial,
sampled and terminal states and exports copied immutable tuples. A failure
before any new residual attempt updates that iteration's last record instead of
adding a duplicate. Previously exported records never change.

The callback adapters are reserved together for setup, each step and a full
run, including exception transfer and result construction. With a numerical
plan, the residual adapter stays reserved throughout sampling. One adapter can
serve as both residual and derivative. If the second adapter is already in use,
the first is not left reserved. Reentrant or concurrent use of an active solver
or adapter is rejected. Separate adapters that share native context memory
still need synchronization by their owner. Context owners stay alive until
every adapter and solver using them is released.

Native callbacks use the same versioned C/Cython descriptor as bisection. With
a compiled residual and a compiled derivative, the whole numerical run happens
outside the GIL; a Python callback takes the GIL for its call and conversion.
The [compiled example](../examples/native_scalar.md) and its `solve_newton()`
function show how to build against the installed package.

Plain Newton always takes the full step. For Armijo backtracking, use
[Damped Newton](native-damped-newton.md). [Halley and Chebyshev](native-higher-order.md)
accept analytic or numerical first and second derivatives, and
[Vector Newton](native-vector.md) handles square and rectangular systems. A
running native solve cannot be cancelled.
