# Linear algebra operators

`cyroot.ops.matrix_ops` exposes the current float32/float64 operators. These
are useful for inspecting the numerical backend; root solvers do not yet all
use these low-level interfaces.

```python
import numpy as np
from cyroot.ops import matrix_ops

a = np.array([[3., 1.], [1., 2.]])
b = np.array([[1.], [3.]])
solution = np.asarray(matrix_ops.solve(a, b))
# [[-0.2], [1.6]]
```

`solve` requires a nonempty square matrix and a two-dimensional right-hand side
with the same number of rows. `transposed=True` solves the transposed system.
`inv` requires a square matrix; `lu_factor`, `svd`, and `svdvals` also accept
rectangular matrices. LU pivots retain LAPACK's **one-based** indexing.

Inputs may be C-contiguous, Fortran-contiguous, or strided, including negative
strides. The current memoryview signatures require writable inputs even when
an operation preserves them. Allocating operations preserve input contents;
forms ending in `_` intentionally modify their operand. Allocated output
matrices preserve Fortran order when the relevant input has that layout and
otherwise use C order.

Forms ending in `_out` write into supplied output buffers with exact required
shapes. Each output's elements must occupy disjoint writable storage. Separate
SVD outputs must not share storage. Unsupported output layouts raise an error;
use an ordinary contiguous output if a more unusual layout is rejected.
For elementwise operations, partial overlap between inputs and outputs is
unsupported; use separate output storage. Explicit in-place forms support
writing back to the same operand.

`svd` produces full square `U` and `Vt` matrices. Validate a decomposition by
reconstruction, singular values, and orthogonality: singular vector signs and
bases within degenerate subspaces are not unique. All allocating SVD forms
preserve input contents regardless of layout.

Empty/nonfinite inputs, incompatible output dimensions and unrepresentable
sizes raise errors. Singular LU/inverse/solve operations, non-positive-definite
Cholesky input and LAPACK factorization failures raise `numpy.linalg.LinAlgError`.
Factorization failure leaves supplied outputs unchanged. These wrappers still
allocate private workspaces per operation; allocation reuse and uninterrupted
native solver execution are separate work.
