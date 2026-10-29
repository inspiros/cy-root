# Scalar Solver Migration

In 2.0, all 26 scalar solvers run on native state machines, under the same
names in `cyroot` and `find_scalar_root` as before. Python callables and native
callback descriptors drive the same numerical methods. Python callbacks take
the GIL for their calls, while native callbacks let the whole numerical run
happen without it; validation, ownership setup and result and history export
hold the GIL.

## Supported Methods

| Family | Methods | Native inspection interface |
| --- | --- | --- |
| Bracketing | bisect | `Bisection` |
| Bracketing | regula_falsi, illinois, pegasus, anderson_bjorck, dekker, brent, chandrupatla, ridders, toms748, wu, itp | `BracketSolver(method=...)` |
| Interval search | hybisect | `Hybisect` |
| Newton | newton, homeier | `Newton`, `Homeier` |
| Higher order | halley, super_halley, chebyshev, tangent_hyperbolas, householder | `Halley`, `ModifiedHalley`, `SuperHalley`, `Chebyshev`, `TangentHyperbolas`, `Householder` |
| Interpolation | secant, sidi, steffensen, inverse_quadratic_interp, hyperbolic_interp, muller | `Secant`, `Sidi`, `Steffensen`, `InverseQuadraticInterpolation`, `HyperbolicInterpolation`, `Muller` |

The inspection classes live in `cyroot.native`. They expose `step()`, `run()`,
`result`, `done`, iteration and bounded copied history. For example:

```python
from cyroot import brent, find_scalar_root
from cyroot.native import BracketSolver

answer = brent(lambda x: x*x-2, 0, 2)
assert answer.success
assert find_scalar_root('brent', lambda x: x*x-2, 0, 2).success

solver = BracketSolver(lambda x: x*x-2, 0, 2, method='brent', history=32)
first = solver.step()
answer = solver.run()
assert answer.success
assert solver.history[-1].root == answer.root
```

`step()` advances one iteration of the algorithm. Ridders can sample twice per
step and one TOMS748 iteration several times, and Steffensen needs an extra
residual evaluation to form its candidate. Hybisect steps are search events,
described below. Function counts include every callback attempt; supplied
initial values are not counted. A failed trial keeps the last accepted finite
state. To diagnose a failure, check `termination_reason`, `f_calls`,
`precision` and `error`. A small step with a residual that is not small is
stagnation, not success. Bracketing methods can also succeed by narrowing a
sign-changing bracket enough, given the method's continuity assumptions.

`max_iter` is one budget shared by `step()` and `run()`; zero or `None` removes
the limit. History is opt-in: `history=0` disables it, and `history_every` and
`history_bytes` bound the copied snapshots. A full history does not stop the
solve, and snapshots stay valid after the solver resumes or is destroyed. No
observer callback is ever called.

ITP's projection uses a fixed full-width tolerance derived from `ptol` and the
initial `prtol` scale. With both disabled, it projects on an internal spacing
scale instead, and the stopping tests are unchanged. The paper's bound on
bisection steps holds in exact arithmetic; with rounding, an extra iteration
can be needed near a width threshold. The solver always respects `max_iter` and
never relaxes the width tolerance to claim success. TOMS748 accepts `k` from 1
to 100.

## Newton Derivatives And Breaking Changes

When no analytic derivative is given, the Newton-family functions use native
derivative plans: `h=None` selects automatic steps, and an explicit `h` central
differences with that step. A step `h` that no numerical derivative would use
is rejected. `ScalarDerivativePlan` and `ScalarSecondDerivativePlan` set the
scheme directly. The 1.x `FiniteDifference` API was removed; pass a native plan
or a Python callable instead (see [Finite Differences](finite-differences.md)).

The `alpha` argument of `halley` selects native modified Halley; Super-Halley
and Chebyshev correspond to `alpha` 1 and 0. The scalar tangent-hyperbolas
formulas 1 and 2 are algebraically the same as Halley's and use the same native
update, but `formula` is still validated. Householder uses the native
runtime-order reciprocal-series recurrence, without SymPy.

Compared with the 1.x kernels, iteration and evaluation counts, floating-point
paths, failure states and numerical derivative steps can differ. Rebuild
compiled consumers against the 2.0 declarations. The removed private kernel
functions and `NewtonPolynomial` have no replacement.

## Compiled Real And Complex Callbacks

Real methods accept a `ScalarCallback`, created from Python or by
`make_scalar_callback` in `cyroot.native.scalar`. Typed `.py` and `.pyx`
functions can both provide one; see
[Compile Python Callbacks](compiling-callbacks.md).

Muller accepts Python complex callables or `ComplexScalarCallback`. It can leave
the real axis even when all starting points are real:

```python
from cyroot import muller

answer = muller(lambda z: z*z+1, 0, 1, 2)
assert answer.success
assert abs(answer.root*answer.root+1) < 1e-9
```

`cyroot.native.complex_callbacks` declares the native Cython factory
`make_complex_scalar_callback(function, context, keepalive=None)`. Its portable
C signature passes components separately:

```cython
ctypedef int (*ComplexScalarFunction)(double real, double imag,
    double* out_real, double* out_imag, void* context) noexcept nogil
```

Write both outputs and return zero on success. A nonzero status stops the
solve and raises `NativeCallbackError`; native exceptions must not escape this
ABI. The header `complex_callbacks.h` defines `cyroot_complex_scalar_descriptor`
(ABI version 1) and the capsule name `cyroot.complex_scalar_callback.v1`.
`ComplexScalarCallback.from_capsule(capsule, keepalive=owner)` validates and
copies the descriptor. The producer must keep its context alive and synchronize
shared mutable state. Reentrant or concurrent use of one adapter is rejected.

## Hybisect Interval Search

Hybisect needs both a residual and an interval-enclosure callback. The latter is
called as `interval_function(a,b)` and returns `(lower,upper)` containing **all**
residual values on the interval. Producing a rigorous enclosure, including
outward rounding where needed, is the caller's responsibility.

```python
from cyroot.native import Hybisect

def enclosure(a, b):
    # Exact for the simple representable endpoints used in this example.
    return (-1 if a <= 0 <= b else min(a*a, b*b)-1,
            max(a*a, b*b)-1)

solver = Hybisect(lambda x: x*x-1, enclosure, -2, 2, history=32)
answer = solver.run()
assert answer.success
assert answer.root == [-1, 1]
```

The search discards intervals whose enclosure excludes zero, splits intervals
whose endpoints have the same sign (depth first), and bisects sign-changing
intervals. Each sign-changing interval yields one root; neither a completed
search nor the method guarantees that every root in the domain was isolated.
Identical roots found at shared endpoints are reported once.

`max_iter` applies to each root bracket separately and `max_split_iter` to the
total number of splits; zero removes either limit. The pending stack and the
root records are allocated up front within `workspace_bytes` (8 MiB by
default), with `max_pending=4096` and `max_roots=4096`. Running out of capacity
stops the search with `breakdown` and sets `workspace_exhausted`; the roots
found so far remain available. A successful search has
`search_termination_reason='completed'` and successful per-root reasons; a
completed search can rightly return no roots.

`step()` performs one interval, split, bisection or root event. The copied
events include the current bracket, the total bisection, split and callback
counts, the roots found, the pending intervals and the search's termination.
`roots_found` and `root` events count recorded root candidates, failed brackets
included, so check the per-root reason before treating a candidate as a root.
The result keeps per-root iteration counts and `(residual_calls,
interval_calls)` separately. Callbacks first run on `step()` or `run()`;
constructing a search only validates arguments and allocates its state.

`IntervalCallback` uses `cyroot.native.hybisect.make_interval_callback` and the
native signature:

```cython
ctypedef int (*IntervalFunction)(double a, double b,
    double* lower, double* upper, void* context) noexcept nogil
```

Write both bounds on success; nonfinite or reversed bounds stop the search.
`interval_callbacks.h` defines the version-1 interval descriptor and the
capsule name `cyroot.interval_callback.v1`. Context, keepalive and exception
rules are those of the other callbacks. Both the residual and the enclosure must
be native for the whole search to run without the GIL.

## Removed Experimental Methods

The experimental Ahmadi and Mishchenko methods, which were never verified,
were removed in 2.0.
