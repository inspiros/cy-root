# Rectangular Matrix Halley And Super-Halley

`VectorHalley` and `VectorSuperHalley` accept different input and residual
lengths through the [shared higher-order engine](native-rectangular-chebyshev.md),
with the same signatures as for square systems. Declared providers, the
residual cache or the first Python residual, which is reused, determine the
shapes, and `None` derivatives select numerical plans of the right shape.
Compiled callbacks and the numerical loop run without the GIL; Python callbacks
take it.

```python
from cyroot import generalized_halley, generalized_super_halley
from cyroot.native import VectorHalley

def residual(x):
    value = x[0]**2 - 2.
    return (value, 2*value)

def jacobian(x):
    return ((2*x[0],), (4*x[0],))

def hessian(x):
    return (((2.,),), ((4.,),))

for method in (generalized_halley, generalized_super_halley):
    answer = method(residual, jacobian, hessian, (1.5,))
    assert answer.success
    assert abs(answer.root[0]-2.**.5) < 1e-9

solver = VectorHalley(residual, None, None, (1.5,), history=8)
first = solver.step()
answer = solver.run()
assert answer.success
assert first.linear_stage == 'secondary'
assert first.linear_rank == 1
assert solver.history[-1].root == answer.root
```

## Rectangular Equations

For \(J\in\mathbb R^{m\times n}\) and \(H\) of shape \((m,n,n)\), first
compute the minimum-norm least-squares correction and directional tensor matrix:

\[
s=J^+F,\qquad A_{o,k}=\sum_jH_{o,j,k}s_j.
\]

Matrix Halley then uses

\[
B=J-\tfrac12A,\qquad v=B^+F,\qquad \Delta x=-v.
\]

Super-Halley uses

\[
B=J-A,\qquad v=B^+F,\qquad \Delta x=-\tfrac12(s+v).
\]

The pseudoinverse stands for the minimum-norm least-squares solution computed
by the DGELSY plan; no inverse is formed. The rank threshold is fixed at
`rcond=1e-12`, as for rectangular Chebyshev and Newton. Numerical rank
deficiency is allowed in both the primary and the secondary system, and their
ranks can differ. A secondary system of rank zero gives a zero solution, which
can end in unsuccessful stagnation. There is no restart, damping or switch to
another method.

These formulas extend the square updates to rectangular systems. They are not
claimed to converge cubically on rectangular, rank-deficient or rank-changing
problems in general, nor to match every possible generalized-inverse
formulation. A least-squares residual alone neither rejects a step nor makes
the point a root. **Only nonlinear residual convergence counts as success**,
and an inconsistent stationary point can stagnate without one.

Every ordered pair of Hessian inputs contributes, so asymmetric analytic
tensors are used as given. The shared contraction uses scaled, compensated
arithmetic and forms secondary entries with fused multiply-add, and
Super-Halley's averaging handles very large and subnormal values as in the
square case. None of this removes derivative noise, cancellation or
conditioning limits.

Square systems still use LU. A singular square primary or secondary matrix
reports `breakdown`, with no least-squares fallback.

## Routes, Diagnostics And Storage

These entry points share the same state and equations:

| Entry point | Method |
| --- | --- |
| `VectorHalley`, `vector_halley`, `generalized_halley(alpha=None)` | Halley |
| `VectorSuperHalley`, `vector_super_halley`, `generalized_super_halley` | Super-Halley |
| `generalized_tangent_hyperbolas` | Halley |
| `VectorModifiedHalley(alpha=.5)`, `generalized_halley(alpha=.5)` | Halley |
| `VectorModifiedHalley(alpha=1)`, `generalized_halley(alpha=1)` | Super-Halley |

`alpha=0` gives rectangular Chebyshev. Other finite values of `alpha` use the
[general rectangular correction](native-rectangular-modified.md), so steps can
jump near exactly `.5` and `1`. The `method` property of `VectorModifiedHalley`
is `'modified_halley'` even when a special path is taken.

Callbacks, cache shapes, copied result and history tuples, handling of the
accepted state, counting and bounded storage work as for rectangular
Chebyshev. `linear_stage` is `'jacobian'` for the primary solve and
`'secondary'` while forming or solving the secondary system. `linear_rank`
belongs to the latest DGELSY solve, so if the secondary contraction fails
before its solve, it is still the primary rank. `linear_status` is the latest
LAPACK status. Neither the rank nor a successful solve implies a root. For
square LU the rank is `None`.

Each rectangular step makes two DGELSY solves with the same immutable plan,
work array and pivots. The secondary right-hand side reuses the \(m\)-entry
contraction buffer, and the solution has \(n\) entries. Nothing is allocated per
step. Compared with rectangular Chebyshev, the workspace adds an \(mn\)-double
secondary matrix. The plans' steps and scratch count against the workspace
budget, and copied history has its own checked budget.

[Higher mixed partials](native-higher-mixed.md) are available as separate
tensor plans; these second-order solvers use only Jacobian and Hessian
providers.
