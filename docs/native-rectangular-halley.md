# Rectangular Matrix Halley And Super-Halley

`VectorHalley` and `VectorSuperHalley` now accept independent input and residual
lengths through the [shared higher-order engine](native-rectangular-chebyshev.md).
The established Python function signatures are unchanged: use ordinary callbacks
and point vectors as before. Declared providers, the residual cache or the first
reused Python residual determine the shapes. `None` derivatives select matching
native numerical plans. Compiled callbacks and numerical loops execute without
the GIL; Python callbacks acquire it.

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

## Specified Rectangular Equations

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

The pseudoinverse notation denotes the minimum-norm least-squares solution
computed by the existing DGELSY plan, not an explicitly constructed inverse.
The fixed default rank threshold is `rcond=1e-12`, matching rectangular
Chebyshev and Newton. Numerical rank deficiency is allowed in both primary and
secondary systems; their ranks may differ. A rank-zero secondary system returns
its zero minimum-norm solution, which can lead to unsuccessful stagnation.
There is no automatic restart, damping or switch to another method.

These are explicit rectangular extensions of the existing square update
formulas. They do not establish cubic convergence for arbitrary rectangular,
rank-deficient or rank-changing problems, nor assert equivalence to every
possible generalized-inverse formulation. Least-squares residuals do not by
themselves reject a step or establish a root. **Only nonlinear residual
convergence reports success.** Inconsistent stationary points can stagnate
without root success.

All ordered Hessian input pairs contribute, preserving asymmetric analytic
tensors. The shared contraction uses scaled compensated arithmetic and forms
secondary entries with fused multiply-add. Super-Halley's averaging retains its
existing range handling for very large or subnormal finite values. These
choices do not remove derivative noise, cancellation or conditioning limits.

Square systems keep strict LU and their existing factor/solve arithmetic.
A required singular square primary or secondary matrix reports `breakdown`,
with no least-squares fallback. Rectangular least-squares behavior does not alter
that square policy.

## Routes, Diagnostics And Storage

These routes share the same state and equations:

| Route | Selected path |
| --- | --- |
| `VectorHalley`, `vector_halley`, `generalized_halley(alpha=None)` | Halley |
| `VectorSuperHalley`, `vector_super_halley`, `generalized_super_halley` | Super-Halley |
| `generalized_tangent_hyperbolas` | Halley |
| `VectorModifiedHalley(alpha=.5)`, `generalized_halley(alpha=.5)` | Halley |
| `VectorModifiedHalley(alpha=1)`, `generalized_halley(alpha=1)` | Super-Halley |

The `alpha=0` special case remains rectangular Chebyshev. Other finite modified
Halley parameters use the separately specified [general rectangular correction](native-rectangular-modified.md).
Steps can be discontinuous near exact `.5` and `1` method selectors.
The existing `method` label remains `'modified_halley'` for that class even when
it normalizes to a special path.

Callbacks, cache shapes, copied result/history tuples, accepted-state handling,
counting and bounded storage follow the rectangular Chebyshev contract.
`linear_stage` is `'jacobian'` for primary work and `'secondary'` while forming
or solving the secondary system. `linear_rank` belongs to the latest DGELSY
solve: if secondary contraction fails before a solve, it can still be the
primary rank. `linear_status` is the latest backend status. Neither rank nor
backend success certifies a root. Square LU rank remains `None`.

Each rectangular step performs two DGELSY solves using the same immutable plan,
native work array and pivots. The secondary right-hand side reuses the existing
\(m\)-entry contraction buffer; the solution has \(n\) entries. There is no
per-step native allocation or separate rectangular solver loop. Compared with
rectangular Chebyshev, the persistent workspace adds one \(mn\)-double
secondary matrix. Provider-plan steps and scratch remain included in the
workspace budget, and copied history uses its separate checked budget.

SciPy and NumPy remain package dependencies. [Higher mixed partials](native-higher-mixed.md)
are available through separate tensor plans; these second-order solvers consume
Jacobian and Hessian providers.
