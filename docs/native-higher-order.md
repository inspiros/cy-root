# Native Halley Family

`cyroot.native.Halley`, `Chebyshev`, `ModifiedHalley`, `SuperHalley` and
`TangentHyperbolas` share one scalar solver interface that uses first and
second derivatives. The residual can be a Python callable or a compiled
`ScalarCallback`. Each derivative can independently be an analytic callback, a
native finite-difference plan, or `None` for a default central plan.

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
and `modified_halley(..., alpha=0.5, **options)` run a solver to completion,
and the top-level `cyroot` functions of the same names run the same engines.
To configure a numerical first derivative, pass a
[ScalarDerivativePlan](native-scalar-derivative.md); for the second derivative,
pass a [ScalarSecondDerivativePlan](native-second-derivative.md). Each plan type
serves only its own derivative order. Derivative arguments are positional, so
pass `None` explicitly to select a default plan.

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

`ModifiedHalley` adds a keyword-only `alpha` to the common constructor
options. Any finite real value is allowed, including negative values and values
outside `[0, 1]`; text, sequences, booleans and nonfinite values are rejected.
`alpha` is converted once, before any numerical work, even when the starting
point is already a root. `alpha=0` and `alpha=0.5` reproduce the Chebyshev and
Halley arithmetic exactly. `SuperHalley` fixes the parameter at one and has no
`alpha` option.

For scalar equations, both classical tangent-hyperbolas formulas reduce to
Halley's update, so `TangentHyperbolas` reuses the Halley engine and has no
`formula` selector. The public `cyroot.tangent_hyperbolas` still accepts
`formula=1` or `formula=2`, and both give the same result. The equivalence
holds only for scalar equations.

The ratio form avoids squaring the derivative or multiplying the residual by
the second derivative, either of which can overflow when the residual is
rescaled. The ratios and all later arithmetic must still be finite: an
unrepresentable intermediate reports `nonfinite`, even where a different
algebraic arrangement might have succeeded.

With exact derivatives, these methods converge cubically near a smooth simple
root under the usual nondegeneracy conditions. Numerical derivative errors can
lower the rate and the attainable accuracy. There is no global convergence
guarantee, bracket, damping or line search. A zero first derivative reports
`zero_derivative`; a zero second derivative is fine. A zero Halley or modified
Halley denominator reports an unsuccessful `breakdown` before another residual
call; there is no automatic fallback to Newton. A candidate equal to the
current point is still evaluated before success or stagnation is decided. In
particular, a zero Super-Halley correction at `q=1` evaluates that point and
reports stagnation if its residual is not small enough. Super-Halley can
converge faster on special functions such as quadratics, but in general it is
third order at simple roots.

Tolerances and the iteration budget work as for [native Newton](native-newton.md)
and as described in [Stopping And Results](stopping.md): the residual scale is
the initial absolute residual, steps are actual finite differences between
consecutive points, and `max_iter` is shared by `step()` and `run()`. Each
attempt to evaluate the residual at a finite candidate is one iteration; an
arithmetic failure before that call uses no iteration and no call.

## Results, Caches And Lazy Derivatives

`run()` and `result` return `NewtonMethodsReturnType`. Its `df_root` is the
tuple `(first_derivative, second_derivative)` at the returned point.
`f_calls` is `(residual_attempts, first_derivative_attempts,
second_derivative_attempts)` for this solve, counting failed calls too. Each
numerical plan evaluation counts as one derivative attempt, and all of its
residual samples count as residual attempts; sampling adds no iterations. Plans
reuse the accepted residual but do not share samples with each other. `step()`
and `history` return immutable `HigherOrderStep` records with separate scalar
`df_root` and `d2f_root` fields, the counts, the metrics, the reason and any
callback stage and code. Step records taken while running have
`termination_reason=None`; the `result` of a running solver reports
`unspecified`.

`f_x0`, `df_x0` and `d2f_x0` are optional cached starting values; each one
skips the corresponding initial call, so it must match what the callback would
return. If the starting point is already a root, unused derivatives are neither
evaluated nor validated, and supplied derivatives are kept in the result.
Cached values are still converted and callbacks still validated during setup.

When another step is needed, the first derivative is evaluated and checked
before the second; a failed, nonfinite or zero first derivative skips the
second derivative call. After a finite candidate residual, the new point and
residual are accepted, both derivatives are reset to NaN, and convergence,
stagnation and the budget are checked before any further derivative work.

A failed or nonfinite candidate residual keeps the previous finite point,
residual, derivatives and metrics, and updates only the counts and the reason.
If the first derivative fails at a newly accepted point, both derivatives are
NaN. If only the second derivative fails, the new first derivative is kept and
the second is NaN. A zero first derivative is reported as zero, with the
unevaluated second derivative as NaN. A derivative from another point is never
reported for the returned point.

## Callbacks, History And Ownership

Native callbacks use the same scalar signature and lifetime rules as
[native Newton](native-newton.md):

```cython
cdef int evaluate(double x, double* out, void* context) noexcept nogil:
    out[0] = ...
    return 0
```

A successful callback must write its output, which is set to NaN before each
call. A nonzero native status raises `NativeCallbackError` with the exact
`native_status`, a copied result and the `callback_stage`: `'function'`,
`'derivative'` or `'second_derivative'`. A failed residual sample inside a
numerical plan reports the stage of that plan. Python exceptions propagate as
the original instance, with the callback's traceback. The solver also exposes
the stage and code; ordinary numerical failures return unsuccessful results.
A callback failure is raised once, by the call that did the work. Later
`run()` or `step()` calls on the finished solver return its failed state
without calling back or raising again.

All supplied adapters are validated before any is reserved. They stay alive
and reserved through setup, execution, result and history construction and
exception transfer. One adapter can fill several roles; distinct adapters that
share mutable native context still need synchronization by their owner.
Reentrant or concurrent use of the solver or an active adapter is rejected.
When every evaluated role has a compiled callback, initialization and the whole
numerical run, including derivative sampling, happen outside the GIL; Python
callbacks take the GIL for their calls and conversions. See the
[compiled example](../examples/native_higher_order.md).

`history`, `history_every`, `history_bytes`, `history_truncated`, `done` and
iteration follow the native scalar conventions. History storage is allocated
and bounded in bytes before native work starts. Initial, sampled and terminal
states are recorded until it is full; later entries are dropped and
`history_truncated` is set. A terminal failure in the same iteration updates
that iteration's last record. Copied snapshots stay unchanged after the solver
resumes or is destroyed. Iterations allocate no new history or workspace
memory.

The callbacks use versioned native descriptors. Native solves cannot be
cancelled and do not check for signals.
