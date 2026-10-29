# Native Barnes Method

`cyroot.barnes` now uses the native direct quasi-Newton engine with a deterministic
Barnes secant update. Initialization, square LU, stopping, callback guards and
copied history are shared with [good Broyden](native-broyden.md). Numerical work
uses native buffers and SciPy's Cython LAPACK provider; native residual callbacks
run without the GIL. Python callbacks and boundary conversions use the GIL.
NumPy and SciPy remain package dependencies.

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

The established signature is
`barnes(F, x0, F_x0=None, J_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`.
The module export and `find_vector_root('barnes', ...)` share this behavior.
`J_x0` is an initial Jacobian matrix, matching `JacobianPlan`, or `None` for
central initialization. Cached `F_x0` counts zero calls and is reused by the
plan. Initial residual convergence skips unused derivatives and updates.

Options after `J_x0` are now keyword-only. `h` and `formula` are removed:
configure `JacobianPlan(..., step=...)`, and use the explicit secant residual
update described below (the former `formula=2`). The shortcut using the new
residual alone assumes an exact linear solve and ignores rounding of the
accepted step. Python callbacks now receive immutable tuples. Replace array
operations on points with sequence arithmetic or explicit conversion inside
the callback. Results now use `BroydenResult`, with tuple snapshots and an
integer residual-call count; old tuple unpacking and ndarray methods need updating.

`cyroot.native.Barnes(function, jacobian0, x0, f_x0=None, **options)` retains
`step()`, `run()`, iteration and copied `.history`; native `barnes(function,
jacobian0, x0, **options)` runs one state. Constructor options and result fields
match `BroydenGood`. `solver.method` is `'barnes'` and
`solver.secant_history_size` is the number of private retained step directions.
This direction memory is separate from the optional exported result history.

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
complement, the projected direction maximizes \(|z^Ts|\). It replaces the
legacy random direction and neither consumes Python/NumPy randomness nor
retries random draws. The first update has no history and reduces to good
Broyden. In one dimension the method continues to reduce to good Broyden,
without attempting to access a zero-capacity direction history.

After a complete finite update commits, append the accepted displacement and
discard the oldest if the capacity is full. Failed updates do not change the
matrix, its update iteration or this FIFO. A newly accepted point/residual is
retained on update failure. Root convergence, stagnation and iteration budget
are checked before an unused update, so terminal results can contain the
previous matrix and direction history. `approximation_iteration` makes this
explicit.

The projection uses scaled, compensated dot products; the rank-one update
retains binary exponents and both compensated residual components through the
final addition. These extend intermediate range and reduce some cancellation;
they do not increase binary64 precision or certify exact multisecant constraints.
Near dependence and ill-conditioning can still limit accuracy. A dependent
current direction does not trigger an automatic restart or Broyden fallback.
There is no damping, line search or global convergence guarantee.

Only nonempty square systems are accepted. Required singular LU solves report
`breakdown`, with no least-squares fallback. Nonfinite factors, residual changes
or final update entries report `nonfinite`. Small steps with nonzero residuals
are unsuccessful stagnation. Only residual convergence reports root success.
Native callback codes, original Python exceptions and last-accepted-state
semantics follow the shared [Broyden engine](native-broyden.md).

## Memory And Native Use

Persistent native storage is \(6n+5n^2\) doubles and \(n\) C-int pivots. Relative
to direct good Broyden, it adds \(n(n-1)\) historical directions, \(n(n-1)\)
basis entries, one \(n\)-double row scratch and one \(n\)-double projected
direction. Numerical initialization adds \(5n\) scratch doubles and copied
explicit plan steps. Budgets are checked before allocation; no cy-root storage
is allocated during a step. Result history has a separate byte budget. Python
boundary objects and provider-internal allocations are excluded.

`barnes_ops.pxd` exposes allocation-free `barnes_direction` and `barnes_update`
from `barnes.hpp`. Trusted callers supply disjoint live spans. Direction inputs
are oldest-first `history[count*n]` and `step[n]`, with `count<=n-1`; scratch is
`basis[n*(n-1)]`, `row[n]`, and output `z[n]`. Update inputs are column-major
`matrix[n*n]`, `step[n]`, `target[n]`, `z[n]`, and output `out[n*n]`. Status zero
means success, one means nonfinite input/final result, two means invalid
shape/count (no writes), and three means a dependent/zero direction or zero
update denominator. Other failures poison the whole output. Producers are
responsible for valid pointers and callback code/context lifetime.

Compiled consumers must rebuild for the 2.0 solver's added private fields.
Callback descriptors, native history records and result layouts are unchanged.
The legacy Barnes NumPy loop and random orthogonal helper are removed. [Robinson](native-robinson.md)
and [Wolfe–Bittner](native-wolfe-bittner.md) also use native states. The original reference is
[Barnes, An algorithm for solving non-linear equations based on the secant
method](https://academic.oup.com/comjnl/article/8/1/66/489886).
