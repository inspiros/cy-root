# Rectangular Chebyshev

`VectorChebyshev` and `generalized_chebyshev` accept
\(F:\mathbb R^n\to\mathbb R^m\) with \(m<n\) or \(m>n\), using the same
higher-order state engine as square systems: buffers, callbacks, derivative
plans and linear algebra adapt to the shape, and there is no separate
rectangular loop. Python callbacks receive tuples and take the GIL. Compiled
callbacks and the numerical loop run without it, reusing the workspace between
steps.

```python
from cyroot import generalized_chebyshev
from cyroot.native import VectorChebyshev

def residual(x):
    value = x[0]**2 - 2.
    return (value, 2*value)

def jacobian(x):
    return ((2*x[0],), (4*x[0],))

def hessian(x):
    return (((2.,),), ((4.,),))

result = generalized_chebyshev(residual, jacobian, hessian, (1.5,))
assert result.success
assert abs(result.root[0]-2.**.5) < 1e-9

solver = VectorChebyshev(residual, None, None, (1.5,), history=8)
first = solver.step()
result = solver.run()
assert result.success
assert first.linear_rank == 1
assert solver.history[-1].root == result.root
assert (solver.input_dimension, solver.output_dimension) == (1, 2)
```

## Equations And Rank Policy

With \(J\) of shape \((m,n)\) and \(H\) of shape \((m,n,n)\), the
rectangular step is defined as

\[
s=J^+F,\qquad q=H[s,s],\qquad t=J^+q,\qquad
\Delta x=-s-\tfrac12t.
\]

Each solve takes the minimum-norm least-squares solution from the DGELSY
`LeastSquaresPlan`, with its default `rcond=1e-12`. As for rectangular Newton,
numerical rank deficiency is allowed. The two solves share native scratch and
pivots, but each calls DGELSY; the factorization is not reused. Every ordered
pair of Hessian inputs contributes, so analytic tensors need not be symmetric.

A rectangular linear solve can leave a nonzero least-squares residual. That
alone neither rejects the correction nor makes the point a root; only nonlinear
residual convergence counts as success. At an inconsistent stationary point the
solve can stop with an unsuccessful `stagnation`, and a Jacobian of rank zero
gives a zero correction with the same outcome. `rcond` is a numerical rank
threshold, not a root tolerance. Cubic convergence is not claimed for
rectangular or rank-deficient problems in general: the behavior depends on the
rank, conditioning, derivative accuracy and the residual map. There is no
damping, line search or automatic switch to another method.

Square systems still use LU, with one factorization and two triangular solves.
A singular square system reports `breakdown` instead of falling back to least
squares, and nonfinite intermediates report `nonfinite`.

`VectorModifiedHalley(alpha=0)` and `generalized_halley(..., alpha=0)` take this
same Chebyshev path, rectangular systems included. Fixed
[Halley and Super-Halley](native-rectangular-halley.md), also reached with
`alpha=.5` and `alpha=1`, handle rectangular systems too. Other values of
`alpha` use the [general rectangular correction](native-rectangular-modified.md).

## Shapes, Callbacks And Derivative Plans

The signatures are the same as for square systems:

- `generalized_chebyshev(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, **options)`.
- `VectorChebyshev(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
  hess_x0=None, **options)` for `step()`, `run()`, iteration and copied history.

Options are keyword-only, as described in the
[square manual](native-vector-chebyshev.md). The length of the point gives \(n\).
Declared providers and the residual cache fix \(m\); if none of them does, one
Python residual evaluation determines it and is reused as the first residual.
Every provider must agree with the point and residual dimensions. Caches have
shapes \((m)\), \((m,n)\) and \((m,n,n)\), and a residual cache sets the
output length rather than implying a square system. Inputs are never reshaped
silently.

Python callables can be passed directly. The native providers are
`RectangularCallback` for residuals and Jacobians, `RectangularHessianCallback`
for analytic Hessians, and `RectangularJacobianPlan` and `RectangularHessianPlan`
for numerical derivatives; `None` selects automatic central plans. Square
callback and plan types require \(m=n\), and rectangular adapters with a square
shape also work. A Hessian provider passed as the residual or Jacobian is
rejected before any input is converted. Callback reservations cover option,
point and cache conversion, evaluation, snapshots and exception construction.

An invalid starting point calls no residual callback; if the output shape is
unknown, the terminal state uses \(m=n\) placeholders. Exceptions from the
first call, which discovers the shape, or from later callbacks propagate as the
original instance. Construction records an inspectable terminal state before
raising a callback error, and inspecting it afterwards makes no further
evaluations. If the cached starting point is a root, no derivatives are
evaluated and no linear solve is made.

## Results, History And Storage

Results are `NewtonMethodsReturnType` with copied tuple point and residual and
`df_root=(Jacobian, Hessian)`. Call counts are `(residual, Jacobian, Hessian)`,
and numerical derivative samples count as residual calls. Native failures in
the residual, Jacobian or Hessian report their own callback stage and code. A
failed trial callback keeps the last accepted point and residual. An accepted
finite trial clears the old derivatives before the stopping checks and
recomputes them only if another iteration is needed.

The solver exposes `input_dimension`, `output_dimension` and `linear_rank`, and
`VectorChebyshevStep` (or `VectorHigherOrderStep`) records include
`linear_rank`. It is `None` for square LU and before the first rectangular
solve, and otherwise the latest DGELSY rank. `linear_status` is the LAPACK
status, and Chebyshev labels both solves `linear_stage='jacobian'`. Neither the
rank nor a successful solve implies a root. Result and history shapes follow
the two dimensions, and exported snapshots are immutable copies.

For rectangular systems, persistent storage holds \(5n+3m+mn+mn^2+w+d\)
doubles and \(n\) C-int pivots, where \(w\) is the least-squares plan's
`work_size` and \(d\) the largest numerical Jacobian or Hessian scratch in use.
The contraction vector doubles as the first solve's right-hand side, and there
is no LU buffer or secondary matrix. The plans' explicit steps count against
the workspace budget.

A temporary copy of the starting point is released before the persistent
storage is allocated. History has its own checked budget: one record plus
\(n+m+mn+mn^2\) doubles per snapshot. Python boundary copies and memory used
inside LAPACK come on top, and the plans' own budget checks can be
conservative. Nothing is allocated per iteration. Numerical derivatives are
configured through the [derivative plans](finite-differences.md).
