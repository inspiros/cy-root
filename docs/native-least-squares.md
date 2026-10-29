# Native Rank-Aware Least Squares

`cyroot.native.LeastSquaresPlan` solves a dense real linear system with
between 1 and 32767 rows and columns. It minimizes `||A*x-b||₂` and returns the
minimum-norm solution at the numerical rank chosen by LAPACK. The rectangular
solvers use it for their steps; square solvers still use LU and report a
singular system as a breakdown.

```python
from cyroot.native import LeastSquaresPlan

plan = LeastSquaresPlan(3, 2, rcond=1e-12)
result = plan.solve(((1., 0.), (0., 1.), (1., 1.)), (1., 2., 4.))
assert result.status == 'ok'
assert result.rank == 2
# solution is approximately (4/3, 7/3); the residual is nonzero.
```

SciPy's LAPACK provides `DGELSY`, which uses pivoted QR followed by a complete
orthogonal factorization; it forms neither the normal equations nor a
pseudoinverse. `rcond` must be finite and strictly between zero and one, and
defaults to `1e-12`. LAPACK chooses the rank from estimated condition numbers
of triangular factors, not from a singular-value cutoff, so the rank can differ
from that of SVD-based routines.
See the [Netlib DGELSY reference](https://netlib.org/lapack/explore-html/dc/d8b/group__gelsy_ga6d1d46ead18df76e993cd4eda6dc1bbb.html).

## Python Interface And Results

`LeastSquaresPlan(rows, columns, *, rcond=1e-12, workspace_bytes=64*1024*1024)`
is immutable. `solve(matrix, rhs)` accepts nested numeric sequences or
buffers, including strided and unaligned ones, of exactly shape
`(rows, columns)` and `(rows,)`. Text and incompatible buffer formats are
rejected. Each call packs the inputs and allocates private output and scratch
in Python; only the numerical computation releases the GIL.

`LeastSquaresResult` contains `solution` (a copied tuple), `rank`,
`linear_status` (LAPACK `info`), and `status`:

| `status` | Meaning |
| --- | --- |
| `ok` | Finite factors and solution, with a valid selected rank. |
| `nonfinite` | Nonfinite input or numerical output. |
| `backend_failure` | LAPACK failure or invalid returned rank. |
| `invalid_plan` | Uninitialized plan or invalid raw pointer arguments. |

On numerical failure, `rank` is `None` and the solution is all NaN. Invalid
shapes and constructor options raise exceptions before any computation. A
finite rank-deficient solution can have `status == 'ok'`. The finiteness check
on the factors is conservative: a representable solution can still be rejected
if rescaling a factor overflows (for example two rows of `1.3e308` with the
same right-hand side). A partial solution is never returned. Neither an `ok`
status nor full rank guarantees accuracy, and the plan reports no residual;
root solvers built on it check their own residual and stopping conditions.

## Reusable Cython Workspaces

The declaration in `cyroot.native.least_squares` exposes:

```cython
cdef int compute(self, const double* A, const double* b, double* out,
                 double* scratch, int* pivots, int* rank, int* info) noexcept nogil
```

Construct a plan and allocate native storage once; `compute` then runs with no
Python objects, allocations or workspace queries. The
[compiled example](../examples/native_least_squares.md) solves a tall and a wide
system, reusing the workspace.

`A` is column-major with `rows*columns` doubles; `b` has `rows` doubles and
`out` has `columns`. Allocate `work_size` scratch doubles and `pivot_size`
(`columns`) C integers. Each call copies the inputs, which are left unchanged,
pads the scratch right-hand side as needed and resets the pivots. All factors
and solution entries are checked before the complete result is copied into
`out`, so a failure leaves `out` unchanged. `rank` is `-1` on failure, and
`info` is zero unless LAPACK reports otherwise. The return code is `0` for
`ok`, `1` for `nonfinite`, `2` for `backend_failure` and `3` for
`invalid_plan`.

Every span, including the scalar outputs, must be valid, correctly sized and
non-overlapping. Checking for null pointers cannot catch a wrong length,
overlap or lifetime; those are the caller's responsibility. A plan created
through `__new__` without initialization returns `invalid_plan`. An
initialized plan is immutable and can be shared between concurrent calls, as
long as each call has its own output and scratch and the inputs stay alive.

## Memory Limits

`rows`, `columns`, `rcond`, `lwork`, `work_size`, `pivot_size` and
`memory_bytes` are read-only properties. Construction validates positive
dimensions and native capacities. Each dimension is capped at 32767, well
below the C `int` limit, to leave headroom for LAPACK's blocked workspace
arithmetic. Construction queries the optimal workspace once and checks it
against the LAPACK minimum.

The budget covers all the storage the plan manages for a solve: the packed
matrix and right-hand side, the solution, a scratch matrix, the padded
right-hand side, LAPACK's work array and the pivots. For `m` rows and `n`
columns this is
`(2*m*n + m + n + max(m,n) + lwork)*sizeof(double) + n*sizeof(int)`.
Python result objects and memory used inside LAPACK are not counted. With the
raw Cython interface the caller allocates the buffers, and the plan's budget
neither manages their lifetime nor limits concurrent calls.
