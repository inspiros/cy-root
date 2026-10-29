# Native Barnes Method

`cyroot.barnes` runs the native direct quasi-Newton engine with a
deterministic Barnes secant update. It shares initialization, the square LU
solve, stopping rules, callback guards and copied history with
[good Broyden](native-broyden.md). Numerical work uses native buffers and
SciPy's Cython LAPACK interface. Native residual callbacks run without the GIL;
Python callbacks and boundary conversions hold it.

```python
from cyroot import barnes
from cyroot.native import Barnes, JacobianPlan

def residual(point):
    return (point[0]**2-2.,)

result = barnes(residual, (1.5,), J_x0=JacobianPlan(1, step=1e-4))
assert result.success
assert abs(result.root[0]-2.**.5) < 1e-9

solver = Barnes(residual, ((3.,),), (1.5,), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
assert solver.secant_history_size == 0  # At most n-1 directions are retained.
```

The public signature is
`barnes(F, x0, F_x0=None, J_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`, also available as
`find_vector_root('barnes', ...)`. `J_x0` is an initial Jacobian matrix, a
`JacobianPlan` of matching size, or `None` for central differences. A cached
`F_x0` counts as zero calls and is reused by the plan. If the starting point is
already a root, no derivative is computed and no update is made.

Compared with 1.x, options after `J_x0` are keyword-only, and `h` and `formula`
are gone. Use `JacobianPlan(..., step=...)` for the step. The update always uses
the explicit secant residual described below (the former `formula=2`); the
other formula used the new residual alone, which assumes an exact linear solve
and ignores rounding of the accepted step. Python callbacks receive immutable
tuples, so array operations on points need indexed arithmetic or an explicit
conversion inside the callback. Results are `BroydenResult` tuples with an
integer residual-call count; code that unpacked the old tuple or used ndarray
methods needs updating.

`cyroot.native.Barnes(function, jacobian0, x0, f_x0=None, **options)` provides
`step()`, `run()`, iteration and copied `.history`, and
`cyroot.native.barnes(function, jacobian0, x0, **options)` runs one solver to
completion. Constructor options and result fields are those of `BroydenGood`.
`solver.method` is `'barnes'`, and `solver.secant_history_size` is the number of
step directions currently retained for the update. That memory is separate from
the optional exported history.

## Deterministic Retained-Secant Update

Write \(s=x_{k+1}-x_k\), using actual rounded accepted points, and
\(y=F(x_{k+1})-F(x_k)\). Retain the most recent \(n-1\) step directions from
completed updates, in oldest-first order. Normalize each historical vector by
its largest absolute entry and build a basis of its numerical span with two
passes of modified Gram–Schmidt. Zero or numerically dependent history rows
are skipped.

Project the max-scaled current step onto the orthogonal complement of that
basis and normalize the result to a unit vector \(z\). The relative dependence
threshold is \(64n\epsilon\), compared with the original max-scaled vector's
Euclidean norm, for both history rank and the current projection. A current
projection at or below that threshold reports unsuccessful `breakdown`.

The matrix update is

\[
B^+=B+\frac{(y-Bs)z^T}{z^Ts}.
\]

In exact arithmetic it satisfies \(B^+s=y\) and preserves the old matrix's
action on the retained numerical basis. History components omitted by the
dependence threshold are preserved only approximately. Among unit vectors in that
complement, the projected direction maximizes \(|z^Ts|\). This deterministic
choice replaces the random direction of 1.x, so no random numbers are drawn.
The first update has no history and reduces to good Broyden, and so does every
update in one dimension, where the direction history has zero capacity.

After a complete, finite update, the accepted displacement is appended to the
history, dropping the oldest entry when it is full. A failed update changes
neither the matrix, its update iteration nor this history, but the newly
accepted point and residual are kept. Convergence, stagnation and the iteration
budget are checked before an update that would go unused, so a final result
can carry the previous matrix and direction history, as
`approximation_iteration` shows.

The projection uses scaled, compensated dot products, and the rank-one update
keeps binary exponents and both compensated residual components through the
final addition. This widens the intermediate range and reduces some
cancellation, but does not raise binary64 precision or make the multisecant
conditions hold exactly. Near-dependence and ill-conditioning can still limit
accuracy. A dependent current direction does not trigger a restart or a
fallback to Broyden. There is no damping, line search or global convergence
guarantee.

Only nonempty square systems are accepted. A singular LU solve reports
`breakdown`, with no least-squares fallback. Nonfinite factors, residual
changes or updated entries report `nonfinite`. A small step with a nonzero
residual is unsuccessful stagnation; only residual convergence counts as a
root. Native callback codes, Python exceptions and the kept last accepted state
behave as in the shared [Broyden engine](native-broyden.md).

## Memory And Native Use

Persistent native storage is \(6n+5n^2\) doubles and \(n\) C-int pivots.
Compared with good Broyden, that adds \(n(n-1)\) stored directions,
\(n(n-1)\) basis entries, an \(n\)-double scratch row and the \(n\)-double
projected direction. Numerical initialization adds \(5n\) scratch doubles and
copies of the plan's explicit steps. Budgets are checked before allocation, and
nothing is allocated during a step. Exported history has its own byte budget.
Python boundary objects and memory used inside LAPACK are not counted.

`barnes_ops.pxd` exposes the allocation-free `barnes_direction` and
`barnes_update` from `barnes.hpp`. Callers must pass valid, non-overlapping
spans. The direction inputs are the oldest-first `history[count*n]` and
`step[n]` with `count<=n-1`, the scratch is `basis[n*(n-1)]` and `row[n]`, and
the output is `z[n]`. The update inputs are the column-major `matrix[n*n]`,
`step[n]`, `target[n]` and `z[n]`, and the output is `out[n*n]`. Status zero is
success, one a nonfinite input or result, two an invalid shape or count (nothing
is written), and three a dependent or zero direction or a zero update
denominator. Other failures fill the whole output with NaN. Callers are
responsible for valid pointers and for keeping callback code and context alive.

The original reference is
[Barnes, An algorithm for solving non-linear equations based on the secant
method](https://academic.oup.com/comjnl/article/8/1/66/489886).
