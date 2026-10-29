# Scalar solver migration

All 26 supported scalar solver entry points now dispatch to native state machines.
Their familiar names remain available from `cyroot` and `find_scalar_root`.
Python callables and explicit native callback descriptors feed the same numerical
methods. Python callbacks acquire the GIL for their calls; native callbacks allow
the numerical run to remain outside the GIL. Validation, ownership setup and
result/history export still happen with the GIL.

## Supported methods

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

`step()` advances one algorithm iteration. Ridders can sample twice; one TOMS748
outer iteration can take several samples. Steffensen needs an auxiliary residual
evaluation to form its candidate. Hybisect steps are search events, described
below. Function counts include attempted callback calls; supplied initial values
are not new evaluations. A failed trial preserves the last accepted finite state.
Check `termination_reason`, `f_calls`, `precision` and `error` when diagnosing
failure. A small step with a non-small residual is stagnation, not root success.
Bracketing methods can also succeed by a sufficiently narrow sign bracket under
the method's continuity assumptions.

`max_iter` is the total budget across `step()` followed by `run()`; zero (or None
where accepted by the common validator) removes that iteration limit. History is
opt-in: `history=0` disables it; `history_every` and `history_bytes` bound copied
snapshots. Exhausting history storage does not stop a solve. Snapshots remain
valid after resumption or destruction. No observer callback runs implicitly.

ITP's projection uses a fixed full-width tolerance derived from `ptol` and the
initial `prtol` scale. With both disabled, an internal spacing-scale projection
is used while the requested stopping tests remain unchanged. Its paper's
bisection budget is an exact-arithmetic result; coordinate rounding can require
an extra iteration near a width threshold. The solver always respects the
caller's `max_iter` and never relaxes the requested width tolerance to claim
success. Native TOMS748 accepts `k` from 1 to 100.

## Newton derivatives and breaking changes

The familiar Newton-family APIs now use native derivative plans when an analytic
derivative is omitted. `h=None` selects automatic steps; explicit `h` selects
central differences with that step. Native `ScalarDerivativePlan` and
`ScalarSecondDerivativePlan` configure schemes directly. The 1.x
`FiniteDifference` API was removed; pass a native plan or an ordinary Python
callable instead (see [finite differences](finite-differences.md)).

The `alpha` argument of familiar `halley` selects native modified Halley.
Super-Halley and Chebyshev select alpha 1 and 0. Scalar tangent-hyperbolas formulas
1 and 2 are algebraically equivalent to Halley and select the same native ratio
formula; the selector remains validated. Householder continues to use the native
runtime-order reciprocal-series recurrence without SymPy.

Changing from legacy kernels can change iteration/evaluation counts, floating
point paths, failed-state contents and numerical derivative steps. Rebuild any
compiled consumers of changed declarations. Removed private kernel functions
and `NewtonPolynomial` are not compatibility APIs.

## Compiled real and complex callbacks

Real methods accept `ScalarCallback`, constructed from Python or exported by
`make_scalar_callback` in `cyroot.native.scalar`. Typed `.py` and `.pyx` functions
can both use it; see [compiling callbacks](compiling-callbacks.md).

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

Write both outputs and return zero on success. Nonzero status stops the solve
and raises `NativeCallbackError`; native exceptions must not escape this ABI.
The header `complex_callbacks.h` defines `cyroot_complex_scalar_descriptor`, ABI
version 1, and capsule name `cyroot.complex_scalar_callback.v1`.
`ComplexScalarCallback.from_capsule(capsule, keepalive=owner)` validates and copies
the descriptor. The producer must keep its context alive and synchronize shared
mutable state. Reentrant/concurrent use of one adapter is rejected.

## Hybisect interval search

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

The search prunes intervals whose enclosure excludes zero, splits equal-sign
endpoint intervals depth first, and bisects sign-changing intervals. One
sign-changing interval produces one root; neither a completed search nor the
method itself certifies isolation of every mathematical root in the domain.
Repeated exact endpoint roots are deduplicated by equality.

`max_iter` applies separately to each root bracket; `max_split_iter` bounds total
splits. Zero removes the corresponding iteration bound. The pending stack and
root records are preallocated under `workspace_bytes` (8 MiB by default), with
`max_pending=4096` and `max_roots=4096`. Capacity exhaustion stops the search with
`breakdown` and sets `workspace_exhausted`; partial roots remain available.
A successful search has `search_termination_reason='completed'` and successful
per-root reasons. A completed search can legitimately return no roots.

`step()` performs one interval, split, bisection or root event. Copied events
include current bracket, global bisection/split/callback counts, roots found,
pending intervals and search termination. `roots_found` and a `root` event count
recorded root candidates, including failed brackets; inspect the per-root reason
before treating a candidate as a successful root. The result retains per-root iteration
counts and `(residual_calls, interval_calls)` separately. Callbacks first run on
`step()`/`run()`; constructing a search only validates and allocates its state.

`IntervalCallback` uses `cyroot.native.hybisect.make_interval_callback` and the
native signature:

```cython
ctypedef int (*IntervalFunction)(double a, double b,
    double* lower, double* upper, void* context) noexcept nogil
```

Write both bounds on success; nonfinite or reversed bounds stop the search.
`interval_callbacks.h` defines the separate version-1 interval descriptor and
capsule `cyroot.interval_callback.v1`. Producer context/keepalive and native
exception rules match the other callbacks. Both residual and enclosure must be
native to keep the full search outside the GIL.

## Remaining project work

The unverified experimental Ahmadi method was removed in 2.0. Vector Vrahatis/polygon code still needs
migration. Native scalar coverage does not mean the full overhaul or supported
platform matrix has completed release qualification. SciPy remains the initial
native linear-algebra provider.
