# Native Wolfe–Bittner Method

`cyroot.wolfe_bittner` now uses native multipoint buffers and SciPy's Cython
LAPACK DGELSY solver. Its numerical loop performs no NumPy operations and runs
without the GIL for compiled residual callbacks. Python callbacks receive tuple
points and acquire the GIL. NumPy and SciPy remain package dependencies.

```python
from cyroot import wolfe_bittner
from cyroot.native import WolfeBittner

def residual(point):
    return (point[0]**3 - 2*point[0] + 2.,)

solver = WolfeBittner(residual, ((0.,), (1.,)), max_iter=2,
                     etol=0., ertol=0., ptol=0., prtol=0., history=3)
first = solver.step()
assert first.root == (1.,)                 # best point so far
assert abs(first.active_root[0]-2.) < 1e-12
result = solver.run()
assert abs(result.root[0]-.8) < 1e-12
assert solver.history[-1].root == result.root

# Rectangular systems and different pool sizes remain supported.
answer = wolfe_bittner(lambda p: (p[0]-1., 2*(p[0]-1.)), ((0.,), (2.,)))
assert answer.success
```

The established signature is
`wolfe_bittner(F, x0s, F_x0s=None, *, rcond=1e-12,
etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.
Options after the residual cache are now keyword-only. The module export and
`find_vector_root('wolfe_bittner', ...)` share this behavior. Native
`WolfeBittner(function, x0s, f_x0s=None, **options)` provides `step()`, `run()`,
iteration, copied `.result` and bounded copied `.history`. Native one-shot
`wolfe_bittner(function, x0s, **options)` runs that state.

## Shapes And Initialization

Supply a nonempty matrix of \(k\) points of length \(n\), with residuals of
length \(m\). The cache, when supplied, must have shape \((k,m)\). Neither
\(m=n\) nor \(k=n+1\) is required. Duplicate points and smaller/larger point
pools are accepted; the affine consistency check below determines whether a
useful step exists. A single nonroot point has zero pool width and stops with
unsuccessful `stagnation`, including when absolute and relative tolerances are
zero. No shape warning or automatic point generation is used.

Python functions infer the residual dimension from the first evaluation and
reuse it. A residual-kind `VectorCallback` declares a square shape;
`RectangularCallback` declares independent input/output dimensions. Both can
wrap compiled callbacks. Native LAPACK limits are \(1\le k\le32767\),
\(1\le m\le32766\), and a positive input dimension fitting C `int`.

All initial point coordinates are packed and checked before any evaluation.
Caches are packed before evaluations, and all initial residuals must be finite
before convergence can be reported. Uncached initial evaluations follow input
order and count once each. Caches count zero calls. The first point attaining
the smallest residual norm wins ties. On an initial evaluation failure, the
best previously evaluated finite point is retained, or the first point with a
NaN residual if no finite evaluation is available.

The fixed residual scale is the largest initial residual infinity norm.
Initial coordinate scale includes every point. Pool width `precision` is the
maximum coordinate range, independent of residual spread. Initial residual or
width stopping occurs before an unused linear solve. Otherwise the pool is
stably ordered from largest to smallest initial residual; ties retain input
order. After every accepted trial, the oldest pool point is discarded and the
trial is appended. There is no repeated residual sorting.

## Affine Step And Root Checks

For the current pool, form columns \(A_{:,j}=(1,F(x_j))\) and solve
\(Ap=b\), with \(b=(1,0,\ldots,0)\), for the minimum-norm least-squares
weights using DGELSY and `rcond`. The immutable plan and caller-owned scratch
support arbitrary rectangular affine systems. Rank deficiency alone does not
reject a consistent solution. `rcond` must be finite and strictly between zero
and one; it controls numerical rank, not root tolerance.

Before sampling the candidate \(x=\sum_jp_jx_j\), require

\[
\|Ap-b\|_\infty \le
32\epsilon\max(m+1,k)\left(\|A\|_\infty\|p\|_\infty+1\right).
\]

An inconsistent affine system reports `breakdown` without a trial callback or
iteration. Nonfinite coefficients, arithmetic, candidates or samples report
`nonfinite`. The helper checks every point, including points with zero weights.
Intermediate sums that overflow also report `nonfinite`; cancellation does not
rescue those calculations. This is a backward consistency check, not a bound
on candidate or root accuracy. A successful LAPACK call alone is insufficient.
Only nonlinear residual convergence reports root success.

A trial iteration is counted when its residual is invoked. Failed callbacks
and nonfinite samples preserve the pool, active accepted state and best point.
A finite trial becomes the active state even if it worsens the residual; it
replaces the best point only on strict improvement. Stop checks use the best
residual and current pool width, then the iteration budget. No next affine
solve occurs until another step is requested. There is no damping, random
restart, line search or global convergence guarantee.

## Results, History And Memory

The new `WolfeBittnerResult` replaces the legacy ndarray `RootReturnType`.
It contains copied tuples `root`, `f_root`, `active_root`, `active_f_root`,
plus `iters`, scalar `f_calls`, `precision`, `error`, `termination_reason`,
`callback_code`, `callback_stage`, `linear_status`, `linear_rank` and
`affine_error`. `root`/`f_root`/`error` always refer to the best point; active
fields identify the latest accepted trial, initially the best initial point.
`success`, `converged` and `optimal` all mean residual convergence.

`WolfeBittnerStep` aliases the same immutable result type. Latest linear rank,
backend status and affine error may describe a failed attempted step while
coordinates remain at accepted states. Before any solve, rank is `None` and
affine error is NaN. Native callback failures carry stage `'function'` and their
original integer code; Python exceptions preserve identity and are delivered
once. Terminal inspection performs no further evaluations. Explicit callback
reservations cover conversions, evaluations and result/error/history creation.

The native workspace contains
\(k(n+m)+3(n+m)+(m+1)k+(m+1)+k+w\) doubles and \(k\) C-int pivots,
where \(w\) is the least-squares plan's `work_size`. The bounded initial point
copy is released before allocating persistent storage. History has separate
checked storage for two point/residual pairs, diagnostics and affine error per
record. Python boundary copies and provider-internal storage are excluded from
the native budget. All loop buffers are allocated before iteration.

`wolfe_bittner_ops.pxd` exposes the allocation-free `noexcept`
`wolfe_bittner_candidate` from `wolfe_bittner.hpp`. Trusted callers supply
disjoint column-major `matrix[rows*k]`, `weights[k]`, row-major `points[k*n]`,
`candidate[n]` and `affine_error[1]` spans. Invalid dimensions return two without
writes; nonfinite inputs/arithmetic return one; inconsistent systems return
three. Valid-dimension failures poison the candidate. Affine error is retained
when the full finite residual was computed, otherwise NaN.

The class is added to the existing native vector extension. Existing callback,
record and result layouts are unchanged. Rebuild compiled consumers using the
2.0 declarations. All established vector quasi-Newton routes now delegate to
native states; the legacy Wolfe–Bittner NumPy loop is removed.
Original reference: [Wolfe–Bittner](https://doi.org/10.1145/368518.368542).
