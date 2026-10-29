# Newton–Krylov

Newton–Krylov computes Newton steps from products with the Jacobian instead of
the Jacobian itself. Since no dense matrix is stored, it suits problems with
many unknowns and cheap directional products. Square systems use restarted
GMRES and tall and wide systems LSQR. Every accepted nonlinear step passes an
Armijo line search.

## Interface

```text
NewtonKrylov(function, jvp, x0, f_x0=None, *, vjp=None,
             restart=20, max_linear_iter=100, linear_rtol=1e-3,
             c1=1e-4, contraction=0.5, max_line_search=32,
             etol=1e-10, ertol=4*eps, ptol=1e-12, prtol=4*eps,
             max_iter=200, max_evaluations=10000,
             history=0, history_every=1, history_bytes=8*1024*1024,
             workspace_bytes=64*1024*1024)
newton_krylov(function, jvp, x0, f_x0=None, *, vjp=None, **options)
```

The class, result type, and function are exported from `cyroot` and
`cyroot.native`. The dispatcher accepts
`find_vector_root('newton_krylov', function, jvp, x0, **options)`.
Shared tolerances use binary64 `eps`; see [Stopping And Results](stopping.md).

```python
from cyroot import NewtonKrylov, newton_krylov

f = lambda x: (x[0]*x[0] - 2., x[1]*x[1] - 3.)
jvp = lambda x, v: (2*x[0]*v[0], 2*x[1]*v[1])
solver = NewtonKrylov(f, jvp, (.1, .2), history=32)
first = solver.step()
answer = solver.run()
assert answer.success and answer.error < 1e-9
assert solver.linear_solver == 'gmres'
assert solver.history[-1].root == answer.root

# None selects native numerical products.
numerical = newton_krylov(f, None, (1.5, 1.8))
assert numerical.success
```

`jvp` is a Python function of `(x, v)`, a `JacobianVectorCallback`, a
`JacobianVectorPlan`, or `None`. For a square problem, `None` selects a forward
directional finite difference that reuses the current `F(x)`. The residual is a
Python callable, a `VectorCallback` or a `RectangularCallback`. The dimensions
come from the adapter, from a cached `f_x0`, or from one Python residual
evaluation, which is counted. Vector inputs are copied, and callbacks receive
tuples.

## Tall And Wide Systems

```python
from cyroot import newton_krylov

f = lambda x: (x[0] - 1., x[1] - 2., x[0] + x[1] - 3.)
jvp = lambda x, v: (v[0], v[1], v[0] + v[1])
vjp = lambda x, w: (w[0] + w[2], w[1] + w[2])
answer = newton_krylov(f, jvp, (0., 0.), vjp=vjp)
assert answer.success
assert max(abs(a-b) for a, b in zip(answer.root, (1., 2.))) < 1e-9

# Underdetermined: a minimum-norm linear step from (0, 0).
wide = newton_krylov(lambda x: (x[0] + x[1] - 3.,), None, (0., 0.))
assert wide.success
assert max(abs(value - 1.5) for value in wide.root) < 1e-9
```

For an `m`-by-`n` rectangular Jacobian, LSQR needs both `Jv` and `J^T w`.
Supply both products, or pass `jvp=None` and omit `vjp`. Without supplied
products, each operation recomputes the same finite-difference columns, so the
two stay consistent; a `JacobianVectorPlan(n, m, ...)` sets the scheme and step.
Each product then costs `n` shifted samples for forward or backward
differences and `2*n` for central ones. No dense Jacobian is stored, but this
can be expensive, so large problems should supply products that exploit their
structure.

A supplied transpose cannot be combined with a numerical JVP in a rectangular
solve. Supplied products must be linear in the direction and describe the same
Jacobian and its transpose at the current point; the solver cannot verify this.
Numerical products assume the residual is deterministic. A transpose supplied
for a square problem is accepted but GMRES does not use it. See
[Jacobian Products](jacobian-vector-products.md) for the adapters and native
ABIs.

LSQR solves for a least-squares step without forming `J^T J`. Because the inner
iteration starts from zero, the step has minimum norm in exact arithmetic. That
does not make the nonlinear root found the one of minimum norm, and LSQR is no
substitute for a dense rank-revealing solve on severely ill-conditioned
systems.

## Verification And Globalization

The inner system is \((J/s)p=-F/s\), where \(s=\|F\|_\infty\). GMRES uses
twice-modified Gram–Schmidt and Givens rotations. LSQR uses Golub–Kahan
bidiagonalization. The recurrences' own estimates decide when convergence is
worth checking, and freshly computed operator products decide whether the inner
solution is actually acceptable. Every normal exit, including the linear
iteration limit and breakdown, recomputes the reported diagnostics. Callback
and nonfinite failures can stop earlier, without them.

Let \(A=J/s\), \(b=-F/s\), and \(r=b-Ap\). `linear_residual` is
\(\|r\|_2/\|b\|_2\). For LSQR, `linear_stationarity` is
\(\|A^\mathsf{T}r\|_2/\|A^\mathsf{T}b\|_2\); a zero denominator gives
zero only when the normal residual is zero. Either verified quantity meeting
`linear_rtol` is sufficient for an LSQR step. GMRES requires the verified
linear residual criterion. `linear_stationarity` is infinity for GMRES.

An acceptable linear step must still decrease the nonlinear residual. The
Armijo search uses the actual slope \(F^\mathsf{T}Jp\), starts with the full
step and shrinks it by `contraction`. Nonfinite trials are rejected, and a
callback exception or nonzero status ends the solve at once. Only accepted
point/residual pairs replace the current state. There is no preconditioning,
adaptive forcing or dense fallback.

## Diagnostics And Termination

`NewtonKrylovResult` is a copied immutable record containing:

| Fields | Meaning |
| --- | --- |
| `root`, `f_root`, `error`, `precision`, `iters` | Accepted point/residual, infinity-norm residual and last accepted step, outer iterations. |
| `f_calls` | `(residual samples, Jv applications)`, including failed attempts and numerical samples. |
| `vjp_calls` | Transpose applications. |
| `linear_iterations` | Cumulative completed Arnoldi or LSQR updates. |
| `linear_residual`, `linear_stationarity`, `linear_status` | Latest verified inner diagnostics. |
| `step_scale`, `backtracks` | Latest Armijo scale and rejection count. |
| `callback_code`, `callback_stage`, `termination_reason` | Explicit failure and stopping information. |

`linear_status` is `-1` before a solve, `0` converged, `1` iteration limit,
`2` breakdown, `3` nonfinite arithmetic/output, or `4` operator failure.
Only `termination_reason='residual_tolerance'` makes `success`, `converged`,
and `optimal` true. LSQR stationarity with a nonzero nonlinear residual never
establishes a root. Other reasons are `stagnation`, `max_iterations`,
`max_evaluations`, `line_search_failure`, `linear_failure`, `nonfinite`,
`callback_failure`, and `derivative_breakdown`.

`max_linear_iter` limits each inner solve. `max_evaluations` counts the
initial residual, finite-difference samples and rejected line-search trials;
running out keeps the accepted state and reports no callback error code.
Callback stages are `function`, `jacobian_vector_product` and
`jacobian_transpose_product`. Python exceptions propagate as the original
instance, and nonzero native codes raise `NativeCallbackError` with a snapshot.
Further `step()` or `run()` calls on a finished solver return its final state
without raising again.

## Storage And Native Execution

The solver exposes `dimension`, `output_dimension`, `linear_solver`, `restart`,
and `workspace_bytes`. GMRES clamps restart to the smaller of the requested
value, input dimension, and inner budget; LSQR reports `restart=0`.
Analytic LSQR storage is `7*(n+m)` doubles. Its numerical fallback adds
`3*n+4*m` doubles and `m` compensated accumulators. GMRES storage is
`O(n*restart + restart**2)`; none of these paths stores a dense Jacobian.
Native budgets exclude Python objects and callback-owned allocations.

`step()`, `run()`, iteration, `result` and the bounded `history` share one
engine. History stores the initial and accepted states and the terminal
diagnostics, not inner Krylov vectors or rejected trial points, and exported
tuples are independent copies. Setup, export and Python callbacks hold the
GIL; with native callbacks the numerical loop runs without it and allocates no
workspace while stepping. An active solver and its explicit adapters are
protected against reentry, and independent solvers with independent contexts
can run concurrently.
