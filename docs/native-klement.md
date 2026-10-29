# Native Klement Method

`cyroot.klement` now uses the native Broyden state engine with Klement's
row-weighted Jacobian update. Its numerical loop, initial derivative sampling
and matrix operations use native buffers and SciPy's Cython LAPACK provider.
With a native residual callback, the numerical work runs without the GIL.
Python callbacks and input/result conversions use the GIL. NumPy and SciPy
remain package dependencies.

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

The established signature is
`klement(F, x0, F_x0=None, J_x0=None, *, etol=1e-10, ertol=4*eps,
ptol=1e-12, prtol=4*eps, max_iter=200, history=0, history_every=1,
history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`.
The module export and `find_vector_root('klement', ...)` use the same engine.

`J_x0` is an initial Jacobian matrix, an initialized matching `JacobianPlan`,
or `None` for automatic central differences. There is no callable Jacobian
role. A cached `F_x0` skips the initial residual and counts zero calls; the
native plan reuses that base. Central initialization uses `2*n` residual samples,
forward/backward uses `n`, and no derivatives are sampled at an initial root.
After initialization, all matrix changes are secant updates.

This migration makes options after `J_x0` keyword-only and removes `h`; configure
`JacobianPlan(..., step=...)` instead. Legacy derivative providers are rejected.
Python residuals receive immutable tuples, and outputs must be sized sequences
or native-format double buffers. Replace array arithmetic on the point with
indexed sequence arithmetic or convert explicitly inside the callback.

The return type is now `BroydenResult`, with tuple point/residual/matrix snapshots,
integer `f_calls`, approximation validity and last-update iteration. Its matrix
is a secant approximation, not an analytic derivative at the returned point.
Old tuple unpacking and NumPy-specific result methods need updating. `success`,
`converged` and `optimal` are true only for residual convergence.

Use `cyroot.native.Klement(function, jacobian0, x0, f_x0=None, **options)` for
retained `step()`, `run()`, iteration and copied `.history`. The native
`klement(function, jacobian0, x0, **options)` convenience function constructs
and runs one state. Constructor options and result/history fields match
[`BroydenGood`](native-broyden.md). `solver.method` is `'klement'`.

## Update And Failure Policy

Write \(s=x_{k+1}-x_k\) using the actual rounded accepted points and
\(y=F(x_{k+1})-F(x_k)\). Each row of the old approximation \(B\) defines
\(u_j=B_{ij}s_j\) and \(d_i=\sum_j u_j^2\). Klement updates

\[
B^+_{ij}=B_{ij}+\frac{\left(y_i-\sum_j u_j\right)u_jB_{ij}}{d_i}.
\]

In exact arithmetic this enforces \(B^+s=y\). The row-dependent weighting
differs from good Broyden's shared \(s^Ts\) denominator. Initial zero matrix
entries remain zero; the method cannot discover an arbitrary missing sparsity
pattern. A zero denominator in any row reports unsuccessful `breakdown` when
an update is needed. No Broyden fallback, restart or random direction is used.

The helper uses binary-exponent scaling and compensated sums to avoid some
premature product, square and correction range failures. It does not increase
precision or certify a small secant error under severe cancellation. In one
dimension a nonzero row uses the secant quotient directly. Nonfinite inputs
or final entries report `nonfinite`.

Only nonempty square systems are supported by this port. Required singular
linear solves report `breakdown`, with no least-squares fallback. No damping,
line search or global convergence guarantee is supplied. A small step with a
nonzero residual is unsuccessful stagnation. The shared engine commits a finite
trial before checking convergence, stagnation and budget, then skips unused
updates. An update failure preserves the accepted point/residual and previous
complete matrix, with `phase='update'` and an unchanged
`approximation_iteration`.

Memory budgets, history sampling, native callback guards and error transfer
are those of direct good Broyden: `6*n+3*n*n` doubles and `n` C-int pivots,
plus `5*n` doubles and copied steps for numerical initialization. Native work
is allocated before the loop. Python and provider-internal allocations are
excluded from the native budget. Retain a residual-kind `VectorCallback` for
compiled calls; no new descriptor or native module is required.

`broyden_ops.pxd` also exposes trusted `klement_update` from `broyden.hpp`.
It accepts disjoint column-major `matrix[n*n]`, `direction[n]`, `target[n]`,
`out[n*n]` spans. Status zero means success, one means nonfinite input/final
output, two means invalid dimension (no writes), and three means a zero row
denominator. Other failures poison the whole output. The native solver
object layout gains a private flag, so compiled consumers must rebuild;
callback descriptors and result/history record layouts are unchanged.

The numerical engine uses native storage. Other supported quasi-Newton methods
also use native states. The original reference is [On Using Quasi-Newton Algorithms of
the Broyden Class for Model-to-Test Correlation](https://jatm.com.br/jatm/article/view/373).
