# Square higher-order vector methods

The established `cyroot.generalized_halley`, `generalized_super_halley`,
`generalized_chebyshev` and `generalized_tangent_hyperbolas` use
[the shared native engine](native-vector-chebyshev.md). Python callbacks receive
tuples. The Chebyshev path, including explicit `alpha=0`, now supports
[rectangular systems](native-rectangular-chebyshev.md) with rank-aware
least-squares corrections. Fixed [matrix Halley and Super-Halley](native-rectangular-halley.md) also
support rectangles. Other modified-Halley alpha values use a specified
[three-solve rectangular correction](native-rectangular-modified.md). Exact alpha
method selectors can produce discontinuous rectangular steps. Singular
square systems use strict LU and do not fall back to a pseudoinverse.
`generalized_newton` also supports independent point/residual lengths; see
[its contract](native-vector.md). All established vector quasi-Newton routes
now use native states; their individual shape policies are documented separately.
Legacy standalone derivative approximations retain their existing interfaces.

## Methods and matrix semantics

| Entry point | Native method |
| --- | --- |
| `generalized_halley(..., alpha=None)` (default) | `VectorHalley` |
| `generalized_halley(..., alpha=value)` | `VectorModifiedHalley` |
| `generalized_super_halley(...)` | `VectorSuperHalley` |
| `generalized_chebyshev(...)` | `VectorChebyshev` |
| `generalized_tangent_hyperbolas(...)` | `VectorHalley` |

For a square residual map, let \(Js=F\) and
\(A_{o,k}=\sum_j H_{o,j,k}s_j\). Matrix Halley solves
\((J-A/2)v=F\), then takes \(\Delta x=-v\). The previous default applied
componentwise division after a different contraction. These vector extensions
agree in one dimension but can give different steps in coupled systems.

For example, at \((0,0)\),
\[
F(x,y)=(1+x+xy,\;2+y+x^2)
\]
has \(J=I\) and \(A=\begin{pmatrix}2&1\\2&0\end{pmatrix}\).
The new matrix Halley step is \((4,2)\); the old componentwise step was
\((1,-4)\). This illustrates a numerical change, not a convergence guarantee.
The old componentwise implementation is removed. Tangent hyperbolas now uses
matrix Halley directly; its redundant `formula` option is removed.

[Modified Halley](native-vector-modified.md) retains matrix multiplication order
and the original Jacobian factors/pivots for its final correction. Explicit
`alpha=0`, `.5` and `1` select the existing Chebyshev, Halley and Super-Halley
arithmetic paths. Other finite parameters use the general correction equation.
No automatic parameter choice, damping or alternate solve is applied on failure.

## Callback inputs, results and derivatives

Python callbacks receive immutable tuple snapshots. Return numeric sequences or
supported double buffers; NumPy arrays are still accepted at the boundary.
Index-based expressions work directly:

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

Expressions such as `point - 1.` or floating-point `scale * point` need sequence
arithmetic or an explicit conversion inside the callback. For existing NumPy
code, an explicit user bridge can call `old_residual(np.asarray(point))`.
That bridge uses Python and the GIL. The solver does not insert a NumPy allocation
around every callback.

`J=None` and `H=None` select independent automatic central finite-difference
plans. Configure them explicitly with the native plan classes:

```python
from cyroot.native import JacobianPlan, HessianPlan

result = generalized_halley(residual, JacobianPlan(2, step=1e-4),
                            HessianPlan(2, step=1e-3), (1.25, 1.75))
assert result.success
```

The `h` option is removed from these four entry points. Use `JacobianPlan` and
`HessianPlan` for numerical derivatives; the 1.x derivative approximation classes
were removed. `generalized_newton` now also requires
an analytic provider or a native `JacobianPlan`.

The common `NewtonMethodsReturnType` now contains tuple snapshots for the point,
residual, Jacobian and Hessian. `df_root=(jacobian,hessian)` uses NaNs for
unevaluated derivatives at the returned point. `f_calls=(F,J,H)` counts native
plan calls in their derivative roles and every sampled residual in `F`.
Stopping, cached-value reuse and failure behavior follow the native engine.

## Options, shapes and failure policy

The shared signature is
`(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, *, etol=1e-10,
ertol=4*eps, ptol=1e-12, prtol=4*eps, max_iter=200, history=0,
history_every=1, history_bytes=8*1024*1024, workspace_bytes=64*1024*1024)`.
`generalized_halley` additionally accepts keyword-only `alpha=None`.
Options after the three cache arguments are keyword-only. The default `max_iter`
is 200; zero or `None` is unlimited. Removed options raise `TypeError` before callbacks run.
Caches and options pass unchanged into the guarded native constructor.

With n inputs and m residuals, Jacobians have shape \((m,n)\) and Hessians
\((m,n,n)\), indexed `[residual][input][input]`. Ordinary Python residuals,
matching native providers or cached residuals determine m; default numerical
plans follow those dimensions. Inputs are never silently reshaped. All finite
alpha paths accept rectangles, with equations and exact special dispatch
specified in the rectangular manuals. Only residual convergence reports success.
Required singular square Jacobian/secondary systems report `breakdown`, with no
least-squares fallback. Rectangular solves allow rank deficiency and use
minimum-norm least-squares corrections. Nonfinite data or intermediates report
`nonfinite`; small steps with unsatisfied residuals report unsuccessful stagnation.

To inspect intermediate steps or native linear-stage/status diagnostics, use
`VectorHalley`, `VectorSuperHalley`, `VectorChebyshev` or `VectorModifiedHalley`
from `cyroot.native`. These classes share `step()`/`run()` and bounded copied
history; the established functions remain one-shot calls. For example:

```python
from cyroot.native import VectorHalley

solver = VectorHalley(residual, jacobian, hessian, (1.25, 1.75), history=16)
first = solver.step()
result = solver.run()
assert result.success
assert solver.history[-1].root == result.root
```

[Native vector/Hessian callbacks](../examples/native_vector_modified.md) can be
passed directly through the established entry points. With native callbacks in
all evaluated roles, the numerical engine executes outside the GIL; Python
callbacks, setup and result/history export use it. NumPy and SciPy remain package
dependencies, and the callback ABI remains a preview. This migration makes no
performance claim or change to rectangular algorithms outside these four methods.
