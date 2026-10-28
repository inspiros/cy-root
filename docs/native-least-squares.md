# Native rank-aware least squares

`cyroot.native.LeastSquaresPlan` solves a dense real system with positive
row and column counts up to 32767 each. It minimizes `||A*x-b||₂` and selects a minimum-norm
solution at the numerical rank chosen by LAPACK. It is a standalone linear
primitive; existing square root solvers keep their singular-system behavior.

```python
from cyroot.native import LeastSquaresPlan

plan = LeastSquaresPlan(3, 2, rcond=1e-12)
result = plan.solve(((1., 0.), (0., 1.), (1., 1.)), (1., 2., 4.))
assert result.status == 'ok'
assert result.rank == 2
# solution is approximately (4/3, 7/3); the residual is nonzero.
```

The SciPy LAPACK provider supplies `DGELSY`, which uses pivoted QR followed by
a complete orthogonal factorization. It does not form normal equations or
materialize a pseudoinverse. `rcond` must be finite and strictly between zero
and one; the default is explicitly `1e-12`. LAPACK chooses rank from estimated
condition numbers of triangular factors. This threshold is not an SVD
singular-value cutoff, so rank choices can differ from SVD-based routines.
See the [Netlib DGELSY reference](https://netlib.org/lapack/explore-html/dc/d8b/group__gelsy_ga6d1d46ead18df76e993cd4eda6dc1bbb.html).

## Python interface and results

`LeastSquaresPlan(rows, columns, *, rcond=1e-12, workspace_bytes=64*1024*1024)`
is immutable. `solve(matrix, rhs)` accepts nested numeric sequences or native
double buffers, including strided and unaligned buffers, with exact shapes
`(rows, columns)` and `(rows,)`. Text inputs and incompatible buffer formats
are rejected. Python setup packs inputs and allocates private output and
scratch on every call; only the numerical computation releases the GIL.

`LeastSquaresResult` contains `solution` (a copied tuple), `rank`,
`linear_status` (LAPACK `info`), and `status`:

| `status` | Meaning |
| --- | --- |
| `ok` | Finite factors and solution, with a valid selected rank. |
| `nonfinite` | Nonfinite input or numerical output. |
| `backend_failure` | LAPACK failure or invalid returned rank. |
| `invalid_plan` | Uninitialized plan or invalid raw pointer arguments. |

On numerical failure, `rank` is `None` and the Python solution remains NaN.
Invalid Python shapes and constructor options raise exceptions before the
computation. A finite rank-deficient solution can have `status == 'ok'`.
The finite-factor check is conservative: a representable solution can still be
rejected if rescaling a factor overflows (for example two rows of `1.3e308`
with the same right-hand side). No partial solution is published.
Neither that status nor full rank certifies accuracy or nonlinear root
convergence. This primitive returns no residual diagnostic; a future root
solver must evaluate its own residual and stopping conditions.

## Reusable Cython workspaces

The declaration in `cyroot.native.least_squares` exposes:

```cython
cdef int compute(self, const double* A, const double* b, double* out,
                 double* scratch, int* pivots, int* rank, int* info) noexcept nogil
```

Construct a plan and allocate native storage once, then call `compute` without
Python objects, allocation or workspace queries in the numerical path.
The [compiled example](../examples/native_least_squares.md) demonstrates both
a tall and a wide system with repeated workspace reuse.

`A` is column-major with `rows*columns` doubles; `b` has `rows` doubles and
`out` has `columns`. Allocate `work_size` scratch doubles and `pivot_size`
(`columns`) C integers. The plan copies inputs, pads its scratch right-hand
side as required and resets pivots each call. Inputs are preserved. It checks
all factors and solution entries before copying the complete result into
`out`; failure leaves `out` unchanged. Raw `rank` is `-1` on failure and `info`
is zero unless the backend reports otherwise. Return codes are `0` for `ok`,
`1` for `nonfinite`, `2` for `backend_failure` and `3` for `invalid_plan`.

All spans, including scalar outputs, must be valid, correctly sized and
disjoint. Null-pointer checks cannot validate arbitrary pointer provenance,
length, overlap or lifetime; these are the compiled caller's responsibility.
An uninitialized plan created through `__new__` returns `invalid_plan`.
The immutable initialized plan can be shared between concurrent calls with
distinct output and scratch allocations and synchronized input lifetimes.

## Memory limits

`rows`, `columns`, `rcond`, `lwork`, `work_size`, `pivot_size` and
`memory_bytes` are read-only properties. Construction validates positive
dimensions and native capacities. Each dimension is capped at 32767 before
calling LAPACK, leaving headroom for its blocked workspace arithmetic. This is
a conservative implementation limit, stricter than the C-int ABI; increasing
it needs a separate provider arithmetic audit. Construction queries optimal
workspace once and checks the returned size against the LAPACK minimum.

The budget includes all plan-managed solve storage: packed matrix and right-hand
side, solution, scratch matrix, padded right-hand side, LAPACK work doubles
and pivot integers. For `m` rows and `n` columns this is
`(2*m*n + m + n + max(m,n) + lwork)*sizeof(double) + n*sizeof(int)`.
Python result/object storage and provider-internal allocations are excluded.
The raw Cython interface still requires the caller to allocate its buffers;
the plan's budget does not manage their lifetime or limit concurrent calls.
