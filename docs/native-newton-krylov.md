# Newton–Krylov

Newton–Krylov computes Newton steps using products with the Jacobian. It avoids
storing a dense matrix, making it useful when the residual has a large input
space and efficient directional products. Square systems use restarted GMRES;
tall and wide systems use LSQR. All accepted nonlinear steps pass an Armijo
line search.

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

`jvp` accepts a Python `(x,v)` function, `JacobianVectorCallback`,
`JacobianVectorPlan`, or `None`. For a square problem `None` selects a forward
directional finite difference that reuses the current `F(x)` cache. The residual
accepts an ordinary callable, `VectorCallback`, or `RectangularCallback`.
Dimensions are inferred from the adapter, cached `f_x0`, or one counted Python
residual sample. Vector inputs are copied; callbacks receive tuples.

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

For an `m`-by-`n` rectangular Jacobian, LSQR uses both `Jv` and `J.Tw`. Supply
both matching products or pass `jvp=None` and omit `vjp`. The numerical fallback
regenerates matching finite-difference coordinate columns for both operations;
a `JacobianVectorPlan(n,m,...)` controls its scheme and step. Each product costs
`n` shifted samples for forward/backward differences, or `2*n` for central
ones. It stores no dense Jacobian, but can be expensive. Large problems should
supply products that exploit their structure.

An independently supplied transpose cannot be combined with a numerical JVP in
a rectangular solve. Supplied products must be linear in their direction and
represent the same Jacobian/transpose at the current point; the solver does not
prove this identity. The finite-difference fallback assumes a deterministic
residual. A supplied transpose is allowed but unused for square GMRES solves.
See [Jacobian Products](jacobian-vector-products.md) for adapter and native ABIs.

LSQR solves a least-squares step without forming `J.T @ J`. Starting the inner
iteration from zero selects a minimum-norm step in exact arithmetic. This is
not a guarantee of a globally minimum-norm nonlinear root, nor a replacement
for a dense rank-revealing solve on severely ill-conditioned systems.

## Verification And Globalization

The inner system is \((J/s)p=-F/s\), where \(s=\|F\|_\infty\). GMRES uses
twice-modified Gram–Schmidt and Givens rotations. LSQR uses Golub–Kahan
bidiagonalization. Recurrence estimates decide when to check convergence;
fresh operator products determine whether the proposed inner solution is
actually acceptable. Normal exits, including a linear iteration cap or
breakdown, refresh the returned diagnostics. Callback and nonfinite failures
may stop earlier with unavailable diagnostics.

Let \(A=J/s\), \(b=-F/s\), and \(r=b-Ap\). `linear_residual` is
\(\|r\|_2/\|b\|_2\). For LSQR, `linear_stationarity` is
\(\|A^\mathsf{T}r\|_2/\|A^\mathsf{T}b\|_2\); a zero denominator gives
zero only when the normal residual is zero. Either verified quantity meeting
`linear_rtol` is sufficient for an LSQR step. GMRES requires the verified
linear residual criterion. `linear_stationarity` is infinity for GMRES.

An acceptable linear step still needs nonlinear decrease. Armijo uses the
actual slope \(F^\mathsf{T}Jp\), starts with the full step, and contracts by
`contraction`. Nonfinite trials are rejected. A callback exception/nonzero
status terminates immediately. Only accepted point/residual pairs replace the
current state. There is no preconditioning, adaptive forcing, or dense fallback.

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

`max_linear_iter` limits each inner solve. `max_evaluations` includes the initial
residual, finite-difference samples, and rejected line-search trials. A budget
failure preserves the accepted state without inventing a callback error code.
Callback stages are `function`, `jacobian_vector_product`, and
`jacobian_transpose_product`. Python exceptions preserve their identity;
native nonzero codes raise `NativeCallbackError` with a snapshot. Repeated
terminal `step()` or `run()` calls return the frozen state without raising it again.

## Storage And Native Execution

The solver exposes `dimension`, `output_dimension`, `linear_solver`, `restart`,
and `workspace_bytes`. GMRES clamps restart to the smaller of the requested
value, input dimension, and inner budget; LSQR reports `restart=0`.
Analytic LSQR storage is `7*(n+m)` doubles. Its numerical fallback adds
`3*n+4*m` doubles and `m` compensated accumulators. GMRES storage is
`O(n*restart + restart**2)`; none of these paths stores a dense Jacobian.
Native budgets exclude Python objects and callback-owned allocations.

`step()`, `run()`, iteration, `result`, and bounded `history` share one engine.
History stores the initial/accepted states and terminal diagnostics, not inner
Krylov vectors or rejected trial points. Exported tuples remain independent.
Setup, export, and ordinary Python callbacks use the GIL; numerical loops with
native callbacks run without it and allocate no solver workspace while stepping.
Each active solver and its explicit adapters stay reserved against reentry;
independent states with independent contexts can run concurrently.
