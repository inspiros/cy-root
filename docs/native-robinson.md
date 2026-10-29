# Native Robinson Method

`cyroot.robinson` runs the shared native Newton state from two starting
points, with a divided-difference matrix built from orthogonal probes. Native
buffers, probe construction and SciPy's Cython LAPACK routines let the
numerical work run without the GIL when the residual is native; Python
callbacks and boundary conversions hold it.

```python
from cyroot import robinson
from cyroot.native import Robinson

def residual(point):
    return (point[0]**2-2.,)

result = robinson(residual, (1.,), (1.5,))
assert result.success
assert abs(result.root[0]-2.**.5) < 1e-9

solver = Robinson(residual, (1.,), (1.5,), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert first.iters == 1
assert solver.history[-1].root == result.root
```

The public signature is
`robinson(F, x0, x1, F_x0=None, F_x1=None, *, orth='efficient',
etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`, also available as
`find_vector_root('robinson', ...)`. Options after `F_x1` are keyword-only.

`cyroot.native.Robinson(function, x0, x1, f_x0=None, f_x1=None, **options)`
provides `step()`, `run()`, iteration and copied `.history`, and
`cyroot.native.robinson(function, x0, x1, **options)` runs one solver to
completion. The solver subclasses `VectorNewton` and reuses its iteration and
stopping rules, without a Jacobian provider or Jacobian cache. Only nonempty
square systems are accepted. The two starting points must have the same length
and differ, even when a cached residual shows one is a root. A residual
`VectorCallback` or a square `RectangularCallback` can wrap a compiled
function.

## Directions And Corrected Secant Equations

Let the previous point be \(p\), the active point \(x\), the actual separation
\(s=x-p\) and radius \(r=\|s\|_2\). `orth='identity'` or the exact built-in
integer `1` uses \(Q=I\). `orth='efficient'` (the default) or the exact built-in integer `2`
uses a deterministic Householder reflection. With \(u=s/r\), set
\(v=u+\sigma e_0\), where \(\sigma=-1\) when \(u_0<0\) and \(+1\) otherwise,
and form

\[
Q=I-\frac{2vv^T}{v^Tv}.
\]

This sign choice avoids cancellation in the reflection vector and handles
steps with leading zero coordinates, which the 1.x construction treated as
special cases. The radius is accumulated with `hypot`. The 1.x `'random'`/`3`
option is gone, and no random numbers are drawn. Booleans, subclasses and
objects that merely convert to these values are not accepted.

Each column defines the full probe point obtained by rounding \(x+rQ_{:,j}\).
Write its actual displacement as \(S_{:,j}=\operatorname{probe}_j-x\), and
\(D_{:,j}=F(\operatorname{probe}_j)-F(x)\). The divided-difference matrix satisfies

\[
JS=D.
\]

The solver factors \(S^T\) once, solves for each row of \(J\), and then takes
the ordinary Newton step \(J\delta=-F(x)\); no inverse is formed. Using the
actual rounded displacements, rather than assuming an exactly orthogonal
matrix, also fixes the row/column orientation error of 1.x. Both direction
modes use the same reconstruction and the same square LU solve.

All starting coordinates and probe points are checked before sampling. A zero
separation, a fully collapsed probe or a singular rounded displacement matrix
reports an unsuccessful `breakdown`; overflowing coordinates, separations,
factors, differences or results report `nonfinite`. There is no pseudoinverse,
random retry, damping or line search, and no global convergence guarantee.
Only residual convergence counts as a root.

## Initial Pair, Counts And Snapshots

Both initial residuals are evaluated in order, `x0` then `x1`, unless they are
cached; cached values count as zero calls. Both points and residuals must be
finite for initialization to proceed. A nonfinite starting point, or two
identical points, fail validation before any callback.

The residual scale is the larger initial residual norm, the coordinate scale
uses both points, and the initial `precision` is their separation in the
infinity norm. If the starting pair already meets a residual or width
tolerance, the point with the smaller residual is returned (`x1` on a tie).
Otherwise `x1` becomes the active point, even if `x0` has the smaller residual,
and early failures report this active state's point, residual and matrix. A
separation that overflows is harmless at an initial root but reports
`nonfinite` once probes are needed.

Python callbacks receive immutable tuples. Results are
`NewtonMethodsReturnType` with copied tuple point, residual and matrix rows,
and `f_calls=(residual_calls, matrix_attempts)`. Building a matrix costs \(n\)
residual probes, and the uncached starting pair adds two calls. A matrix
attempt that fails its node or factor checks counts as an attempt without
probes. Only trial residual calls count as iterations; probes and row solves do
not. Each accepted trial makes the old active point the new previous point.

Convergence, stagnation and the iteration budget are checked before a matrix
rebuild that would go unused, so a final `df_root` is then all NaN. A failed
trial keeps the previous active state. A failed rebuild keeps the accepted
point and residual and clears the whole matrix. Native errors in probe samples
report the stage `'jacobian'`, and errors in the initial or trial residuals
`'function'`. Python exceptions propagate as the original instance; once one
has been raised, further calls only return the solver's state. If a starting
callback fails, the result reports the `x1` state with its cached residual, or
NaN if it was not evaluated.

`step()` and `.history` return `VectorNewtonStep` records, including the linear
status and rank (`None` for square LU). `linear_status` is the latest LAPACK
status, including that of the probe-displacement factorization and row solves.
Exported history is copied. Callback reservations cover setup conversions,
probe sampling, and result, error and history construction.

## Memory And Native Helper

The native workspace is \(8n+4n^2\) doubles and \(n\) C-int pivots: the square
Newton workspace plus the previous point and residual, two probe and
orientation matrices and a scratch row. No finite-difference plan is involved.
Storage is checked and allocated before the iteration starts, and history has
its own bounded allocation. Python boundary objects and memory used inside
LAPACK are not counted.

`robinson_ops.pxd` exposes the allocation-free, `noexcept` `robinson_nodes`
helper from `robinson.hpp`. Callers pass valid, non-overlapping `previous[n]`,
`current[n]`, `nodes[n*n]`, `transpose[n*n]` and `scratch[n]` spans. `nodes`
holds the full probe points column by column, and `transpose[j+n*i]` the actual
displacement in coordinate `i` for probe `j`. Status zero is success, one a
nonfinite input or result, two an invalid dimension (nothing is written), and
three a zero separation or a collapsed probe. Other failures fill both outputs
with NaN. The solver validates the rounded orientation matrix before calling
back.

The original reference is
[Robinson's paper](https://epubs.siam.org/doi/abs/10.1137/0703057).
