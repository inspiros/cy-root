# Migrating Higher-Order Vector Methods

In 2.0, `cyroot.generalized_halley`, `generalized_super_halley`,
`generalized_chebyshev` and `generalized_tangent_hyperbolas` run on
[a shared native engine](native-vector-chebyshev.md), and Python callbacks
receive tuples. Besides square systems, they handle rectangular ones: the
Chebyshev path, including `alpha=0`, with
[rank-aware least-squares corrections](native-rectangular-chebyshev.md),
[matrix Halley and Super-Halley](native-rectangular-halley.md) likewise, and
other values of `alpha` with a
[three-solve rectangular correction](native-rectangular-modified.md), so
rectangular steps can jump at the exact values `.5` and `1`. Singular square
systems use LU and never fall back to a pseudoinverse. `generalized_newton`
also handles different point and residual lengths; see
[Vector Newton](native-vector.md). The vector quasi-Newton methods run on
native states too, each with its own shape rules. The 1.x derivative wrappers
were removed; use the [finite-difference plans](finite-differences.md)
instead.

## Methods And Matrix Semantics

| Entry point | Native method |
| --- | --- |
| `generalized_halley(..., alpha=None)` (default) | `VectorHalley` |
| `generalized_halley(..., alpha=value)` | `VectorModifiedHalley` |
| `generalized_super_halley(...)` | `VectorSuperHalley` |
| `generalized_chebyshev(...)` | `VectorChebyshev` |
| `generalized_tangent_hyperbolas(...)` | `VectorHalley` |

For a square residual map, let \(Js=F\) and
\(A_{o,k}=\sum_j H_{o,j,k}s_j\). Matrix Halley solves
\((J-A/2)v=F\), then takes \(\Delta x=-v\). The 1.x default divided
componentwise after a different contraction. The two agree in one dimension but
can give different steps for coupled systems.

For example, at \((0,0)\),
\[
F(x,y)=(1+x+xy,\;2+y+x^2)
\]
has \(J=I\) and \(A=\begin{pmatrix}2&1\\2&0\end{pmatrix}\).
The matrix Halley step is \((4,2)\), while the 1.x componentwise step was
\((1,-4)\). This shows a numerical difference, not which converges. The
componentwise implementation is gone. Vector tangent hyperbolas uses matrix
Halley directly, and its redundant `formula` option is gone.

[Modified Halley](native-vector-modified.md) keeps the order of matrix products
and reuses the Jacobian's factors and pivots for its final correction. `alpha`
exactly `0`, `.5` and `1` uses the Chebyshev, Halley and Super-Halley
arithmetic, and other finite values the general correction. There is no
automatic parameter choice, damping or alternative solve on failure.

## Callback Inputs, Results And Derivatives

Python callbacks receive immutable tuples and can return numeric sequences or
buffers, NumPy arrays included. Expressions that index the point work
unchanged:

```python
from cyroot import generalized_halley

def residual(point):
    x, y = point
    return (x*x + x*y + y - 5., 2*x + y*y - 6.)

def jacobian(point):
    x, y = point
    return ((2*x + y, x + 1.), (2., 2*y))

def hessian(point):
    return (((2., 1.), (1., 0.)), ((0., 0.), (0., 2.)))

result = generalized_halley(residual, jacobian, hessian, (1.25, 1.75))
assert result.success
assert max(abs(a-b) for a, b in zip(result.root, (1., 2.))) < 1e-9
```

Expressions such as `point - 1.` or `scale * point` need elementwise
arithmetic or an explicit conversion inside the callback. Existing NumPy code
can be wrapped, as in `lambda point: old_residual(np.asarray(point))`; the
wrapper runs in Python under the GIL. The solver itself never creates a NumPy
array around a callback.

`J=None` and `H=None` select independent automatic central-difference plans.
To configure them, pass the native plan classes:

```python
from cyroot.native import JacobianPlan, HessianPlan

result = generalized_halley(residual, JacobianPlan(2, step=1e-4),
                            HessianPlan(2, step=1e-3), (1.25, 1.75))
assert result.success
```

These four functions no longer take `h`; use `JacobianPlan` and `HessianPlan`
for numerical derivatives, since the 1.x derivative approximation classes are
gone. `generalized_newton` likewise takes `None`, an analytic provider or a
native plan for its Jacobian.

`NewtonMethodsReturnType` holds tuple copies of the point, residual, Jacobian
and Hessian. `df_root=(jacobian, hessian)` has NaNs for derivatives that were not
evaluated at the returned point. `f_calls=(F, J, H)` counts each plan
evaluation for its derivative and every sampled residual in `F`. Stopping,
cached values and failures work as in the native engine.

## Options, Shapes And Failure Policy

The shared signature is
`(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, *, etol=1e-10,
ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200, history=0,
history_every=1, history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`.
`generalized_halley` also takes a keyword-only `alpha=None`. Everything after
the three cache arguments is keyword-only. The default `max_iter` is 200; zero
or `None` is unlimited. Removed options raise `TypeError` before any callback
runs. Caches and options are passed unchanged to the native constructor.

With n inputs and m residuals, Jacobians have shape \((m,n)\) and Hessians
\((m,n,n)\), indexed `[residual][input][input]`. The Python residual, native
providers of matching shape or cached residuals determine m, and the default
numerical plans follow. Inputs are never reshaped silently. Every finite
`alpha` works for rectangular systems, as the rectangular pages specify. Only
residual convergence counts as success. A singular square Jacobian or
secondary system reports `breakdown`, with no least-squares fallback, while
rectangular solves allow rank deficiency and take minimum-norm least-squares
corrections. Nonfinite data or intermediates report `nonfinite`, and small
steps with an unsatisfied residual report unsuccessful stagnation.

To inspect intermediate steps or the linear stage and status, use
`VectorHalley`, `VectorSuperHalley`, `VectorChebyshev` or `VectorModifiedHalley`
from `cyroot.native`. They provide `step()`, `run()` and bounded copied
history, while the functions only return the final result. For example:

```python
from cyroot.native import VectorHalley

solver = VectorHalley(residual, jacobian, hessian, (1.25, 1.75), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

[Native vector and Hessian callbacks](../examples/native_vector_modified.md)
can be passed straight to these functions. With native callbacks for every
evaluated role, the numerical engine runs without the GIL; Python callbacks,
setup and result and history export take it. Compiled consumers must be
rebuilt against the 2.0 callback and solver declarations.
