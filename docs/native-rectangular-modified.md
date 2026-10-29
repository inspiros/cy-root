# Rectangular Parameterized Modified Halley

`VectorModifiedHalley` and `generalized_halley(..., alpha=...)` accept
rectangular systems for every finite real `alpha`. Signatures, callback shapes
and numerical plans are those of the
[shared rectangular higher-order engine](native-rectangular-chebyshev.md).
Native callbacks and the numerical loop run without the GIL; Python callbacks
take it.

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

## General Correction

For \(n\) inputs, \(m\) residuals and \(\alpha\notin\{0,1/2,1\}\), the
rectangular update is

\[
s=J^+F,\qquad A_{o,k}=\sum_j H_{o,j,k}s_j,\qquad
v=(J-\alpha A)^+F,\qquad t=J^+(Av),\qquad
\Delta x=-s-\tfrac12t.
\]

Each pseudoinverse is a minimum-norm least-squares solve with DGELSY, using
the same fixed `rcond=1e-12` as rectangular Newton and Chebyshev; no inverse is
formed. Rank deficiency is allowed, and the Jacobian and the secondary matrix
can have different numerical ranks. A solve of rank zero returns zero. Neither
the rank nor the LAPACK status implies a root: **only nonlinear residual
convergence counts as success**, and an inconsistent stationary point can stop
unsuccessfully. There is no damping, restart or automatic fallback, and no
convergence order is promised for rectangular or rank-changing systems.

Square systems still use LU factors for the correction solve, report
`breakdown` on a singular matrix and have no least-squares fallback.

## Exact Special Parameters

`alpha` exactly `0`, `.5` and `1` selects Chebyshev, Halley and Super-Halley,
with their [fixed update formulas](native-rectangular-halley.md) and smaller
workspace. This is a deliberate choice of method: the general pseudoinverse
correction does not reduce to the fixed Halley or Super-Halley formula, so
rectangular steps can jump as `alpha` approaches `.5` or `1`, even when both
matrices have full rank.

For example, at \(x=0\), take
\(F(x)=(1+x,1+x^2/2)\). Then \(J=(1,0)^T\), \(A=(0,1)^T\), and the
general formula takes \(\Delta x=-1\). Exactly `.5` selects Halley and takes
\(-2/5\); exactly `1` selects Super-Halley and takes \(-1/2\). So do not treat
`alpha` as interpolating continuously across these values. At `alpha=0` the
general formula does agree with Chebyshev, by linearity.

## Diagnostics And Storage

The `'correction'` linear stage covers the product \(Av\) and its Jacobian
solve, for rectangular systems too. `linear_rank` always describes the latest
solve, so a successful correction reports the Jacobian's rank, which can differ
from the secondary rank; a contraction failure before that solve leaves the
secondary rank. `linear_status` is reset at the start of each phase and then
keeps the LAPACK status. Callback errors, finiteness checks, the kept accepted
state, iteration counts and copied history work as for the other higher-order
methods.

The general rectangular path makes three solves with the same plan, scratch and
\(n\) pivots. It adds an \(mn\)-double directional matrix to the fixed
rectangular Halley workspace, to keep \(A\) while solving with the secondary
matrix. With \(w\) the LAPACK scratch and \(d\) the largest numerical
derivative scratch, native storage is \(5n+3m+3mn+mn^2+w+d\) doubles and \(n\)
C-int pivots, plus the plans' explicit steps. History has its own checked byte
budget, and nothing is allocated per step.
