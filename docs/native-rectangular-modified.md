# Rectangular parameterized modified Halley

`VectorModifiedHalley` and `generalized_halley(..., alpha=...)` accept rectangular
systems for every finite real `alpha`. Their signatures, callback shapes and
numerical plans follow the [shared rectangular higher-order engine](native-rectangular-chebyshev.md).
Native callbacks and numerical loops run without the GIL; Python callbacks use it.

```python
from cyroot import generalized_halley
from cyroot.native import VectorModifiedHalley

def residual(x):
    value = x[0]**2-2.
    return (value, 2*value)

def jacobian(x):
    return ((2*x[0],), (4*x[0],))

def hessian(x):
    return (((2.,),), ((4.,),))

answer = generalized_halley(residual, jacobian, hessian, (1.5,), alpha=.25)
assert answer.success
solver = VectorModifiedHalley(residual, None, None, (1.5,), alpha=.25, history=8)
first = solver.step()
assert first.linear_stage == 'correction'
assert first.linear_rank == 1
assert solver.run().success
assert solver.history[-1].root == solver.result.root
```

## General correction

For \(n\) inputs, \(m\) residuals and \(\alpha\notin\{0,1/2,1\}\), the
specified rectangular update is

\[
s=J^+F,\qquad A_{o,k}=\sum_j H_{o,j,k}s_j,\qquad
v=(J-\alpha A)^+F,\qquad t=J^+(Av),\qquad
\Delta x=-s-\tfrac12t.
\]

Each pseudoinverse action is a minimum-norm least-squares solve using DGELSY,
with the same fixed `rcond=1e-12` as rectangular Newton and Chebyshev. No inverse
is constructed. Rank deficiency is allowed, and the Jacobian and secondary
matrix can have different numerical ranks. A rank-zero solve returns a zero
minimum-norm solution. Neither rank nor backend status certifies a root;
**only nonlinear residual convergence reports success**. Inconsistent stationary
points can stop unsuccessfully. There is no damping, restart or automatic fallback.
No general convergence order is promised for rectangular or rank-changing systems.

Square systems retain their existing strict LU factors and correction solve,
including `breakdown` on a required singular matrix and no least-squares fallback.

## Exact special parameters

Exactly `alpha=0`, `.5` and `1` select Chebyshev, Halley and Super-Halley,
respectively, preserving the [fixed update formulas](native-rectangular-halley.md)
and their smaller workspace. This is deliberate method selection. The general
pseudoinverse correction need not simplify to the fixed Halley or Super-Halley
formula. Consequently, rectangular steps can be discontinuous as `alpha`
approaches `.5` or `1`, even when both matrices have full rank.

For example, at \(x=0\), take
\(F(x)=(1+x,1+x^2/2)\). Then \(J=(1,0)^T\), \(A=(0,1)^T\), and the
general formula takes \(\Delta x=-1\). Exactly `.5` selects Halley and takes
\(-2/5\); exactly `1` selects Super-Halley and takes \(-1/2\). Do not treat
`alpha` as a continuous interpolation across these method selectors.
The `alpha=0` general formula does agree with Chebyshev by linearity.

## Diagnostics and storage

The existing `'correction'` linear stage now covers rectangular multiplication
\(Av\) and its Jacobian solve. `linear_rank` always describes the latest solve,
so a successful correction reports the Jacobian rank, which may differ from the
secondary rank. A contraction failure before that solve can retain the secondary
rank. `linear_status` is reset before each phase and then retains its backend
status. Existing callback errors, finite checks, accepted-state retention,
iteration counts and copied history rules apply unchanged.

The general rectangular path uses three solves with the same plan, scratch and
\(n\) pivots. It adds one \(mn\)-double directional matrix to fixed rectangular
Halley workspace, keeping \(A\) alive while solving with the secondary matrix.
With \(w\) backend scratch and \(d\) the largest numerical derivative scratch,
total native storage is \(5n+3m+3mn+mn^2+w+d\) doubles and \(n\) C-int pivots,
plus explicit derivative-plan steps. History has its separate checked byte budget.
There is no per-step native workspace allocation, new callback ABI or result schema.
SciPy and NumPy remain package dependencies.
