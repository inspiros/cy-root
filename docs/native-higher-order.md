# Native Halley Family

`cyroot.native.Halley`, `Chebyshev`, `ModifiedHalley`, `SuperHalley` and
`TangentHyperbolas` share a scalar native solver interface using first and
second derivatives. The residual can be a Python
callable or a compiled `ScalarCallback`. Each derivative independently accepts
an analytic callback, a native finite-difference plan, or `None` for a default
central plan. These modes can be mixed.

```python
from cyroot.native import Halley, Chebyshev, ModifiedHalley, SuperHalley, TangentHyperbolas

def function(x):
    return x*x - 2.

for method in (Halley, Chebyshev, ModifiedHalley, SuperHalley, TangentHyperbolas):
    solver = method(function, lambda x: 2*x, lambda x: 2., 1.5, history=16)
    first = solver.step()
    result = solver.run()
    assert result.success
    assert solver.history[-1].root == result.root

# Approximate both derivatives from the residual:
result = Halley(function, None, None, 1.5).run()
assert result.success

# Select another member of the modified Halley family:
result = ModifiedHalley(function, None, None, 1.5, alpha=1.).run()
assert result.success
```

The convenience functions `halley(function, derivative, second_derivative, x0,
**options)`, `chebyshev(...)`, `super_halley(...)`, `tangent_hyperbolas(...)`
and `modified_halley(..., alpha=0.5, **options)` perform a complete run.
Top-level functions use the same native engines, with keyword-only controls.
For numerical first derivatives,
pass a [ScalarDerivativePlan](native-scalar-derivative.md); for numerical second
derivatives, pass a [ScalarSecondDerivativePlan](native-second-derivative.md).
The plan types are specific to their derivative order. Derivative arguments
remain positional: pass `None` explicitly when selecting a default plan.

## Steps And Numerical Failures

For residual \(f\), first derivative \(f'\) and second derivative \(f''\)
at the current point, define

\[
t=\frac{f}{f'},\qquad q=\frac{t}{2}\frac{f''}{f'}.
\]

The parameterized modified Halley family uses

\[
\Delta x=-t\left(1+\frac{q}{1-2\alpha q}\right).
\]

| Class | Parameter | Implemented update |
| --- | --- | --- |
| `Chebyshev` | \(\alpha=0\) | \(-t(1+q)\) |
| `Halley` | \(\alpha=1/2\) | \(-t/(1-q)\) |
| `SuperHalley` | \(\alpha=1\) | \(-t(1+q/(1-2q))\) |
| `ModifiedHalley` | finite real \(\alpha\), default \(1/2\) | parameterized formula above |
| `TangentHyperbolas` | \(\alpha=1/2\) | Halley's update |

`ModifiedHalley` adds keyword-only `alpha` to the otherwise common constructor
options. Negative parameters and values outside `[0,1]` are allowed. Text,
sequences, booleans and nonfinite values are rejected. The parameter is
converted once under solver/callback reservations before any numerical work,
even at a cached initial root. Exact parameters `0` and `0.5` use the existing
Chebyshev and Halley arithmetic, respectively. `SuperHalley` fixes the parameter
at one; it has no `alpha` option.

For scalar residuals, both legacy tangent-hyperbolas formulas simplify to
Halley's update. `TangentHyperbolas` inherits the Halley initializer and engine.
The new native API has no `formula` selector. This scalar equivalence does not
establish a policy for vector tangent-hyperbolas methods.

The ratio form avoids directly squaring the derivative
or multiplying the residual by the second derivative, which can overflow
under common residual rescaling. Ratios and all following arithmetic must
still be finite. An unrepresentable intermediate reports `nonfinite`, even
where a different algebraic arrangement might work.

These methods have local cubic convergence at sufficiently smooth simple roots
under their usual nondegeneracy conditions with exact derivatives. Numerical
derivative errors can limit this convergence rate and achievable accuracy.
They have no global convergence
guarantee, bracketing safeguard, damping or line search. A required zero first
derivative reports `zero_derivative`; a zero second derivative is valid. A zero
Halley or modified-Halley denominator reports unsuccessful `breakdown` before another residual
call. There is no automatic Newton fallback. A finite unchanged candidate is
evaluated before residual success or unsuccessful stagnation is decided.
In particular, a zero Super-Halley correction at `q=1` still evaluates that
unchanged candidate; it reports stagnation if the residual remains unsatisfied.
Super-Halley may have higher local order on special functions such as
quadratics; it is generally a third-order method at simple roots.

Tolerance and total-budget options follow [native Newton](native-newton.md)
and [stopping rules](stopping.md): fixed initial absolute residual scale,
actual finite step magnitude, consecutive point scale, and `max_iter` shared
across stepping and full runs. Each attempted finite candidate residual is one
iteration. Arithmetic failure before that call consumes no iteration or call.

## Results, Caches And Lazy Derivatives

`run()` and `result` return `NewtonMethodsReturnType`. Its `df_root` is the
tuple `(first_derivative, second_derivative)` at the returned point.
`f_calls` is `(residual_attempts, first_derivative_attempts,
second_derivative_attempts)` for this solve. Failed callbacks count as attempts.
Each numerical plan invocation counts as one derivative attempt; all of its
residual samples also count as residual attempts. Sampling adds no iterations.
Plans reuse the accepted residual but do not share samples with each other.
`step()` and `history` return immutable `HigherOrderStep` records, with separate
scalar fields `df_root` and `d2f_root`, along with counts, metrics, reason and
callback stage/code. Running step records have `termination_reason=None`;
the passive running result uses `unspecified`.

`f_x0`, `df_x0` and `d2f_x0` supply independent cached initial values, with zero
new calls for their corresponding roles. They must match their callbacks.
Initial residual convergence precedes unused derivative evaluation and finite
validation. Supplied derivatives are retained at an already solved initial
point. Cache conversion and callback validation still occur during setup.

If another step is needed, the first derivative is evaluated and checked before
the second. A failed, nonfinite or zero first derivative skips the unused
second derivative callback. After a finite candidate residual, the new point
and residual are accepted, both derivatives are cleared to NaN, and convergence,
stagnation and budget are checked before further derivative work.

A failed/nonfinite candidate residual preserves the previous finite point,
residual, derivatives and metrics, with updated attempts and reason. A failed
first derivative at a newly accepted point leaves both derivatives NaN. If only
the second derivative fails, the valid new first derivative is retained and the
second is NaN. A newly evaluated zero first derivative is retained as zero with
an unevaluated second derivative of NaN. No stale derivative from another point
is labelled as a derivative at the returned point.

## Callbacks, History And Ownership

Native producers use the same scalar signature and keepalive interface as
[native Newton](native-newton.md):

```cython
cdef int evaluate(double x, double* out, void* context) noexcept nogil:
    out[0] = ...
    return 0
```

Each successful callback must write its output. Outputs are seeded NaN before
each invocation. A native nonzero status raises `NativeCallbackError` with
the exact `native_status`, copied result and `callback_stage`: `'function'`,
`'derivative'` or `'second_derivative'`. Python exceptions preserve their
original instance and callback traceback. The solver exposes stage/code too;
ordinary numerical failures return unsuccessful results.
Callback failures are delivered once by the call that performed work. Later
`run()`/`step()` on the terminal solver return its unchanged failed state
without further callbacks or repeated exceptions.

All present adapters are validated before any are reserved. They remain pinned
and reserved throughout setup coercion, execution, result/history construction
and exception transfer. One adapter may fill several roles; distinct adapters
sharing mutable native context still require producer synchronization.
Reentrant/concurrent solver or active-adapter access is rejected. With compiled
callbacks for every evaluated role, initialization and the full numerical run
stay outside the GIL, including numerical derivative sampling. Python callbacks
reacquire it for calls and conversion. A failed residual sample reports the
stage of its plan: `'derivative'` or `'second_derivative'`. See the
[compiled example](../examples/native_higher_order.md).

`history`, `history_every`, `history_bytes`, `history_truncated`, `done` and
iteration follow the native scalar conventions. History is preallocated and
byte-bounded before native work. Initial, sampled and terminal states are
recorded until capacity is exhausted; later entries are dropped and truncation
is reported. Same-iteration terminal failures update the last matching record.
Copied snapshots remain valid and unchanged after resumption or destruction.
No new native history or workspace allocations occur per iteration.

The callbacks use versioned native descriptors. NumPy and SciPy remain package
dependencies, and native cancellation/signal checks are not implemented.
