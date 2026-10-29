# Native Robinson Method

`cyroot.robinson` now uses the shared native Newton state with two initial points
and a divided-difference matrix built from orthogonal probes. Native buffers,
probe construction and SciPy's Cython LAPACK routines allow numerical work
without the GIL when the residual is native. Python callbacks and boundary
conversions use the GIL. NumPy and SciPy remain package dependencies.

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

The established signature is
`robinson(F, x0, x1, F_x0=None, F_x1=None, *, orth='efficient',
etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200,
history=0, history_every=1, history_bytes=8*1024*1024,
workspace_bytes=64*1024*1024)`.
The module export and `find_vector_root('robinson', ...)` share this behavior.
Options after `F_x1` are now keyword-only.

Retain `cyroot.native.Robinson(function, x0, x1, f_x0=None, f_x1=None,
**options)` for `step()`, `run()`, iteration and copied `.history`. Native
`robinson(function, x0, x1, **options)` runs one state. The state subclasses
`VectorNewton` and reuses its iteration and stopping rules, with no analytic
Jacobian provider or Jacobian cache. Only nonempty square systems are accepted.
The initial points must have matching lengths and be distinct, even at a
cached root. A residual-kind `VectorCallback` or square `RectangularCallback`
can wrap a compiled function.

## Directions And Corrected Secant Equations

Let the previous point be \(p\), the active point \(x\), the actual separation
\(s=x-p\) and radius \(r=\|s\|_2\). `orth='identity'` or the exact built-in
integer `1` uses \(Q=I\). `orth='efficient'` or the exact built-in integer `2`
uses a deterministic Householder reflection. With \(u=s/r\), set
\(v=u+\sigma e_0\), where \(\sigma=-1\) when \(u_0<0\) and \(+1\) otherwise,
and form

\[
Q=I-\frac{2vv^T}{v^Tv}.
\]

This choice avoids cancellation in the reflection vector and replaces the
legacy efficient construction's zero-prefix cases. Radius accumulation uses
`hypot`. The old `'random'`/`3` option is removed; no random state is consumed.
Booleans, subclasses and conversion-protocol objects are not selector values.

Each column defines the full probe point obtained by rounding \(x+rQ_{:,j}\).
Write its actual displacement as \(S_{:,j}=\operatorname{probe}_j-x\), and
\(D_{:,j}=F(\operatorname{probe}_j)-F(x)\). The divided-difference matrix satisfies

\[
JS=D.
\]

The native code factors \(S^T\) once and solves for each row of \(J\), then
uses the ordinary Newton step \(J\delta=-F(x)\). It constructs no inverse.
This corrects the old row/column orientation mismatch and accounts for the
actual rounded displacements instead of assuming an ideal orthogonal matrix.
Both direction modes use the same reconstruction and strict square LU policy.

All initial coordinates and complete probe nodes are checked before sampling.
A zero separation, wholly collapsed probe or singular rounded displacement
matrix reports unsuccessful `breakdown`; overflowing coordinates, separation,
factors, differences or final entries report `nonfinite`. No pseudoinverse,
random retry, damping, line search or global convergence guarantee is supplied.
Only residual convergence reports root success.

## Initial Pair, Counts And Snapshots

Both initial residuals are evaluated in order, `x0` then `x1`, unless their
matching caches are supplied. Caches count zero calls. Both initial points and
residuals must be finite for initialization to proceed. A nonfinite initial
point skips callbacks. Identical points fail validation before callbacks.

The fixed initial residual scale is the larger initial residual norm; the
initial coordinate scale uses both points, and initial `precision` measures
pair separation in the infinity norm. Initial convergence or width stopping
considers the smaller residual and returns its matching point (`x1` wins ties).
Otherwise `x1` is active, even if `x0` has a smaller residual. This replaces the
legacy best-initial placeholder for early failures: result coordinates, residual
and matrix now describe the active state. An overflowing separation can remain
unused at an initial root; it reports `nonfinite` if probes are required.

Python callbacks now receive immutable tuple points, and results use
`NewtonMethodsReturnType` with copied tuple point/residual/matrix rows and
`f_calls=(residual_calls, matrix_attempts)`. This replaces the old `RootReturnType`
and scalar call count. A matrix attempt costs \(n\) residual probes; the uncached
initial pair adds two calls. Failed node/factor preflight counts an attempt but
no probe. Trial residual calls count iterations; probes and row solves do not.
An accepted trial makes the old active point the next previous point.

Terminal root, stagnation and iteration-budget checks precede unused matrix
rebuilds; terminal `df_root` is then all NaNs. Failed trials preserve the previous
active state. A failed rebuild preserves the accepted point/residual and clears
the whole matrix. Native sample errors report stage `'jacobian'`, while initial
base and trial errors report `'function'`. Original Python exceptions retain
identity; terminal inspection after delivery is passive. Initial callback
failures retain the `x1` state with its cache or NaN residual if not yet evaluated.

`step()` and `.history` return shared `VectorNewtonStep` records, including
linear status and rank (`None` for square LU). `linear_status` is the latest
LAPACK status, including a required probe-displacement factorization/row solve.
History exports are copies; all callback reservations span setup conversions,
probe sampling and result/error/history construction.

## Memory And Native Helper

Native workspace is \(8n+4n^2\) doubles and \(n\) C-int pivots: the square
Newton workspace plus previous point/residual, two probe/orientation matrices
and one row scratch vector. There is no immutable finite-difference plan.
Storage is checked and allocated before iteration, with a separate bounded
result-history allocation. Python boundary objects and provider-internal
memory are excluded from this native budget.

`robinson_ops.pxd` exposes the allocation-free, `noexcept` `robinson_nodes`
helper from `robinson.hpp`. Trusted callers supply disjoint `previous[n]`,
`current[n]`, `nodes[n*n]`, `transpose[n*n]` and `scratch[n]` spans. Nodes are
column-major full probe points; `transpose[j+n*i]` stores the actual displacement
in coordinate `i` for probe `j`. Status zero means success, one means nonfinite
input/result, two means invalid dimension (no writes), and three means zero
separation or a collapsed probe. Other failures poison both complete outputs.
The solver validates the rounded orientation matrix before invoking callbacks.

The 2.0 solver object/vtable adds private setup/state fields; compiled
consumers must rebuild. Callback descriptors, record and result layouts are
unchanged. The legacy NumPy Robinson loop and random/efficient direction
helpers are removed. [Wolfe–Bittner](native-wolfe-bittner.md) also uses a native state.
The original reference is [Robinson's paper](https://epubs.siam.org/doi/abs/10.1137/0703057).
