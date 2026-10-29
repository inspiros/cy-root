# Native Wolfe–Bittner Method

`cyroot.wolfe_bittner` uses native multipoint buffers and SciPy's Cython
LAPACK DGELSY solver. Its numerical loop does no NumPy operations and, with
compiled residual callbacks, runs without the GIL. Python callbacks receive
tuple points and take the GIL.

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

# Rectangular systems and pools of any size are supported.
answer = wolfe_bittner(lambda p: (p[0]-1., 2*(p[0]-1.)), ((0.,), (2.,)))
assert answer.success
```

The public signature is
`wolfe_bittner(F, x0s, F_x0s=None, *, rcond=1e-12,
etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`, also available as
`find_vector_root('wolfe_bittner', ...)`. Options after the residual cache are
keyword-only. `cyroot.native.WolfeBittner(function, x0s, f_x0s=None,
**options)` provides `step()`, `run()`, iteration, a copied `.result` and a
bounded, copied `.history`, and `cyroot.native.wolfe_bittner(function, x0s,
**options)` runs one solver to completion.

## Shapes And Initialization

Supply a nonempty matrix of \(k\) points of length \(n\), with residuals of
length \(m\). A cache, if supplied, must have shape \((k,m)\). Neither \(m=n\)
nor \(k=n+1\) is required; duplicate points and smaller or larger pools are
accepted, and the affine consistency check below decides whether a useful step
exists. A single point that is not a root has zero pool width and stops with an
unsuccessful `stagnation`, even when all tolerances are zero. No warning is
issued about the shape and no extra points are generated.

Python functions get the residual dimension from their first evaluation, which
is reused. A residual `VectorCallback` declares a square shape and a
`RectangularCallback` separate input and output dimensions; both can wrap
compiled callbacks. LAPACK limits the sizes to \(1\le k\le32767\) and
\(1\le m\le32766\), and the input dimension must fit a C `int`.

All starting coordinates and caches are packed and checked before any
evaluation, and every initial residual must be finite before convergence can be
reported. Uncached initial residuals are evaluated in input order and count
once each; cached ones count as zero. On ties, the first point with the
smallest residual norm wins. If an initial evaluation fails, the result reports
the best finite point evaluated so far, or the first point with a NaN residual
if there is none.

The residual scale is the largest initial residual infinity norm, and the
coordinate scale includes every point. The pool width `precision` is the
largest coordinate range, regardless of the residuals. Initial residual or
width tolerances are checked before any linear solve. Otherwise the pool is
ordered from largest to smallest initial residual, keeping input order on ties.
After each accepted trial, the oldest point is dropped and the trial appended;
the pool is not re-sorted.

## Affine Step And Root Checks

For the current pool, form columns \(A_{:,j}=(1,F(x_j))\) and solve
\(Ap=b\), with \(b=(1,0,\ldots,0)\), for the minimum-norm least-squares
weights using DGELSY and `rcond`. Any rectangular affine system is supported,
and rank deficiency alone does not reject a consistent solution. `rcond` must
be finite and strictly between zero and one; it sets the numerical rank cutoff,
not a root tolerance.

Before sampling the candidate \(x=\sum_jp_jx_j\), require

\[
\|Ap-b\|_\infty \le
32\epsilon\max(m+1,k)\left(\|A\|_\infty\|p\|_\infty+1\right).
\]

An inconsistent affine system reports `breakdown` without a trial callback or
iteration. Nonfinite coefficients, arithmetic, candidates or samples report
`nonfinite`. Every point is checked, including those with zero weight, and an
intermediate sum that overflows reports `nonfinite` even if later cancellation
would have brought it back into range. The test checks backward consistency;
it does not bound the error of the candidate or the root. A successful LAPACK
call alone is not enough, and only nonlinear residual convergence counts as a
root.

An iteration is counted when the trial residual is evaluated. A failed callback
or nonfinite sample leaves the pool, the active state and the best point
unchanged. A finite trial becomes the active state even if its residual is
worse, and replaces the best point only if it is strictly better. The stopping
checks use the best residual and the current pool width, then the iteration
budget. The next affine solve happens only when another step is requested.
There is no damping, random restart or line search, and no global convergence
guarantee.

## Results, History And Memory

Results are `WolfeBittnerResult` tuples, replacing the ndarray-based
`RootReturnType` of 1.x. They contain the copied tuples `root`, `f_root`,
`active_root` and `active_f_root`, plus `iters`, the integer `f_calls`,
`precision`, `error`, `termination_reason`, `callback_code`, `callback_stage`,
`linear_status`, `linear_rank` and `affine_error`. `root`, `f_root` and `error`
always describe the best point; the active fields describe the latest accepted
trial, which starts as the best initial point. `success`, `converged` and
`optimal` all mean residual convergence.

`WolfeBittnerStep` is the same immutable type. The latest rank, LAPACK status
and affine error can describe a failed step while the coordinates stay at the
accepted states. Before the first solve, the rank is `None` and the affine error
NaN. Native callback failures report the stage `'function'` and the original
integer code; Python exceptions propagate as the original instance, once.
Inspecting a finished solver makes no further evaluations. Callback
reservations cover conversions, evaluations and result, error and history
creation.

The native workspace holds
\(k(n+m)+3(n+m)+(m+1)k+(m+1)+k+w\) doubles and \(k\) C-int pivots,
where \(w\) is the least-squares plan's `work_size`. The temporary copy of the
starting points is released before the persistent storage is allocated.
History has its own checked storage for two point/residual pairs, diagnostics
and the affine error per record. Python boundary copies and memory used inside
LAPACK are not counted. All loop buffers are allocated before the iteration.

`wolfe_bittner_ops.pxd` exposes the allocation-free, `noexcept`
`wolfe_bittner_candidate` from `wolfe_bittner.hpp`. Callers pass valid,
non-overlapping column-major `matrix[rows*k]`, `weights[k]`, row-major
`points[k*n]`, `candidate[n]` and `affine_error[1]` spans. It returns two for
invalid dimensions (nothing is written), one for nonfinite inputs or arithmetic
and three for an inconsistent system. With valid dimensions, any failure fills
the candidate with NaN. The affine error is kept when the full finite residual
was computed, and is NaN otherwise.

Original reference: [Wolfe–Bittner](https://doi.org/10.1145/368518.368542).
