# Native Klement Method

`cyroot.klement` runs the native Broyden state engine with Klement's
row-weighted Jacobian update. Its numerical loop, initial derivative sampling
and matrix operations use native buffers and SciPy's Cython LAPACK interface.
With a native residual callback the numerical work runs without the GIL; Python
callbacks and input and result conversions hold it.

```python
from cyroot import klement
from cyroot.native import JacobianPlan, Klement

def residual(point):
    x, y = point
    return (x*x+y-2., x+2*y*y-3.)

result = klement(residual, (1.4, .7),
                 J_x0=JacobianPlan(2, step=1e-4))
assert result.success
assert max(abs(value-1.) for value in result.root) < 1e-8

solver = Klement(residual, ((2.8, 1.), (1., 2.8)), (1.4, .7), history=32)
first = solver.step()
result = solver.run()
assert result.success
assert first.approximation_iteration == 1
assert solver.history[-1].root == result.root
```

The public signature is
`klement(F, x0, F_x0=None, J_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`, also available as
`find_vector_root('klement', ...)`.

`J_x0` is an initial Jacobian matrix, a `JacobianPlan` of matching size, or
`None` for automatic central differences; a callable Jacobian is not accepted.
A cached `F_x0` skips the initial residual call and is reused by the plan.
Central initialization takes `2*n` residual samples and forward or backward
initialization `n`; nothing is sampled when the starting point is a root. After
initialization, the matrix changes only through secant updates.

Compared with 1.x, options after `J_x0` are keyword-only and `h` is gone; use
`JacobianPlan(..., step=...)`. The removed 1.x derivative providers are
rejected. Python residuals receive immutable tuples and must return sized
sequences or buffers, so array arithmetic on the point needs indexed arithmetic
or an explicit conversion inside the callback.

The result is a `BroydenResult`, with tuple point, residual and matrix, an
integer `f_calls`, the approximation's validity and the iteration of its last
update. The matrix is a secant approximation, not the derivative at the
returned point. Code that unpacked the old tuple or used NumPy methods on the
result needs updating. `success`, `converged` and `optimal` are true only for
residual convergence.

`cyroot.native.Klement(function, jacobian0, x0, f_x0=None, **options)` provides
`step()`, `run()`, iteration and copied `.history`, and
`cyroot.native.klement(function, jacobian0, x0, **options)` runs one solver to
completion. Constructor options and result and history fields are those of
[`BroydenGood`](native-broyden.md). `solver.method` is `'klement'`.

## Update And Failure Policy

Write \(s=x_{k+1}-x_k\) using the actual rounded accepted points and
\(y=F(x_{k+1})-F(x_k)\). Each row of the old approximation \(B\) defines
\(u_j=B_{ij}s_j\) and \(d_i=\sum_j u_j^2\). Klement updates

\[
B^+_{ij}=B_{ij}+\frac{\left(y_i-\sum_j u_j\right)u_jB_{ij}}{d_i}.
\]

In exact arithmetic this enforces \(B^+s=y\). Unlike good Broyden's shared
\(s^Ts\) denominator, the weighting differs from row to row. Entries that start
at zero stay zero, so the method cannot discover a sparsity pattern that the
initial matrix lacks. A zero denominator in any row reports an unsuccessful
`breakdown` when an update is needed; there is no fallback to Broyden, restart
or random direction.

The update uses binary-exponent scaling and compensated sums to avoid some
premature overflow and underflow in products, squares and corrections. It does
not raise precision or make the secant error small under severe cancellation.
In one dimension a nonzero row uses the secant quotient directly. Nonfinite
inputs or updated entries report `nonfinite`.

Only nonempty square systems are supported. A singular linear solve reports
`breakdown`, with no least-squares fallback. There is no damping, line search
or global convergence guarantee, and a small step with a nonzero residual is
unsuccessful stagnation. The engine accepts a finite trial before checking
convergence, stagnation and the budget, and skips updates that would go
unused. A failed update keeps the accepted point and residual and the previous
complete matrix, with `phase='update'` and an unchanged
`approximation_iteration`.

Memory budgets, history sampling, callback guards and error transfer are those
of good Broyden: `6*n+3*n*n` doubles and `n` C-int pivots, plus `5*n` doubles
and copied steps for numerical initialization. Native storage is allocated
before the loop; Python objects and memory used inside LAPACK are not counted.
For compiled calls, pass a residual `VectorCallback`.

`broyden_ops.pxd` also exposes the low-level `klement_update` from
`broyden.hpp`. It takes non-overlapping column-major `matrix[n*n]`,
`direction[n]`, `target[n]` and `out[n*n]` spans. Status zero is success, one a
nonfinite input or result, two an invalid dimension (nothing is written), and
three a zero row denominator. Other failures fill the whole output with NaN.

The original reference is [On Using Quasi-Newton Algorithms of the Broyden
Class for Model-to-Test Correlation](https://jatm.com.br/jatm/article/view/373).
