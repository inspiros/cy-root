# Native Linear Algebra

The 1.x Python matrix/vector operator helpers were removed in 2.0. Solvers use
preallocated native operations and SciPy's Cython BLAS/LAPACK interfaces.

Use [Native Least Squares](native-least-squares.md) for the public rank-aware
`LeastSquaresPlan` and its compiled workspace API. For large problems that should
avoid a dense Jacobian, see [Newton–Krylov](native-newton-krylov.md).
