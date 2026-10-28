# Rectangular Chebyshev

`VectorChebyshev` and the established `generalized_chebyshev` now accept
\(F:\mathbb R^n\to\mathbb R^m\), including \(m<n\) and \(m>n\), through
the existing higher-order state engine. Shape-dependent buffers, callbacks,
derivative plans and linear algebra are shared; there is no separate rectangular
iteration loop. Python callbacks use tuples and acquire the GIL. Compiled
callbacks and the numerical loop run without it, with workspace reused between
steps. NumPy and SciPy remain dependencies.

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

## Equations and rank policy

With \(J\) of shape \((m,n)\) and \(H\) of shape \((m,n,n)\), the
rectangular extension is explicitly defined by

\[
s=J^+F,\qquad q=H[s,s],\qquad t=J^+q,\qquad
\Delta x=-s-\tfrac12t.
\]

Each solve uses the minimum-norm least-squares solution returned by the existing
DGELSY `LeastSquaresPlan`, with its default `rcond=1e-12`. Numerical rank
deficiency is allowed, matching rectangular Newton's policy. The two solves
reuse the same native scratch and pivots; the current implementation invokes
DGELSY twice, without introducing a separate factorization API. All ordered
Hessian input pairs contribute; analytic tensors need not be symmetric.

A rectangular linear solve may have a nonzero least-squares residual. That does
not by itself reject the correction or establish a root. Only nonlinear residual
convergence reports success. An inconsistent stationary point can stop with
unsuccessful `stagnation`; a zero-rank Jacobian can similarly produce a zero
correction. `rcond` is a numerical rank threshold, not a root tolerance.
This extension does not assert cubic convergence for general rectangular or
rank-deficient problems. Its behavior depends on rank, conditioning, derivative
accuracy and the residual map. There is no damping, line search or automatic
alternative method.

Square systems retain strict LU: one factorization and two triangular solves.
A required singular square system reports `breakdown`; it does not fall back to
least squares. Nonfinite native intermediates report `nonfinite`. The existing
square update arithmetic, matrix Halley branches, stopping and history logic
are retained.

`VectorModifiedHalley(alpha=0)` and `generalized_halley(..., alpha=0)` select
this same Chebyshev path and accept rectangles. Fixed [Halley and Super-Halley](native-rectangular-halley.md), including
`alpha=.5/1` aliases, also support rectangles. The general modified-Halley path
uses its specified [rectangular correction](native-rectangular-modified.md),
with exact alpha values selecting the fixed methods.

## Shapes, callbacks and derivative plans

The familiar signatures remain:

- `generalized_chebyshev(F, J, H, x0, F_x0=None, J_x0=None, H_x0=None, **options)`.
- `VectorChebyshev(function, jacobian, hessian, x0, f_x0=None, jac_x0=None,
  hess_x0=None, **options)` for `step()`, `run()`, iteration and copied history.

Options remain keyword-only as documented in the [square manual](native-vector-chebyshev.md).
Input length determines \(n\). Declared providers and the residual cache constrain
\(m\). If no provider/cache declares it, one Python residual evaluation infers
\(m\) and is reused as the base. Every provider must agree with the point and
residual dimensions. Matching caches have shapes \((m)\), \((m,n)\) and
\((m,n,n)\); a supplied residual cache now defines its own output length rather
than implying a square system. Inputs are never silently reshaped.

Use ordinary Python callables directly. Explicit compiled/native providers are
`RectangularCallback` for residuals/Jacobians, `RectangularHessianCallback` for
analytic Hessians, `RectangularJacobianPlan` and `RectangularHessianPlan` for
numerical derivatives. `None` selects automatic native central plans.
Square callback/plan types constrain \(m=n\); square-shaped rectangular adapters
also work. Native role checks reject Hessian providers used as residuals or
Jacobians before input conversion. Explicit callback reservations span option,
point and cache conversion, evaluation, snapshots and exception construction.

An invalid initial point skips residual callbacks. If no output shape was
available, the terminal diagnostic uses \(m=n\) as a fallback. Original exceptions
from first-call dimension discovery or subsequent callbacks retain identity.
Construction leaves an inspectable terminal state before delivering callback
errors; inspection afterward performs no further evaluations. A cached initial
root skips unused derivative evaluations and linear solves.

## Results, history and storage

Results remain `NewtonMethodsReturnType` with copied tuple point/residual and
`df_root=(Jacobian, Hessian)`. Call counts are `(residual, Jacobian, Hessian)`;
numerical derivative sampling increments residual calls. Native base/Jacobian/
Hessian failures retain their respective callback stages and codes. Failed trial
callbacks preserve the last accepted point/residual. A finite accepted trial
clears stale derivatives before stopping checks and rebuilds them only if another
iteration is needed.

`input_dimension`, `output_dimension` and `linear_rank` are available on the
solver. `VectorChebyshevStep`/`VectorHigherOrderStep` append `linear_rank` to their
existing fields. It is `None` before a rectangular solve and for square LU;
otherwise it is the latest DGELSY rank. `linear_status` is the backend status;
Chebyshev's existing `linear_stage='jacobian'` labels both solves. Rank and linear
success do not imply root success. Result/history tensor shapes reflect the
independent dimensions, and exported snapshots remain immutable copies.

For rectangles, persistent storage contains
\(5n+3m+mn+mn^2+w+d\) doubles and \(n\) C-int pivots, where \(w\) is the
least-squares plan's `work_size` and \(d\) is the largest active numerical
Jacobian/Hessian scratch requirement. The contraction vector doubles as the
first solve's right-hand side. There is no square LU buffer or secondary matrix.
Explicit derivative-plan steps are included in the solver's workspace budget.
Square native buffer accounting remains unchanged.

A bounded bootstrap point copy is released before persistent allocation.
History has a separate checked budget: one native record plus
\(n+m+mn+mn^2\) doubles per stored snapshot. Python boundary copies and
provider-internal memory are additional; plan-level budget guards can be
conservative. There is no per-iteration native allocation.

The draft solver object and record layout gain fields, and copied step records
append rank. Rebuild compiled consumers. Callback descriptors and the established
result schema are unchanged. This increment does not change legacy general
finite-difference step conventions or introduce a generic solver framework.
