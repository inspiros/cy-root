"""Private packed square-solve backend; pointers are validated by its caller."""
cdef int square_solve(int n, double* matrix, double* rhs, int* pivots) noexcept nogil

# Trusted positive C-int dimension and private column-major matrix/pivot spans.
# factor overwrites matrix/pivots; solve_factored reads them and overwrites rhs.
# The caller checks finite data and every returned LAPACK info value.
cdef int square_factor(int n, double* matrix, int* pivots) noexcept nogil
cdef int square_solve_factored(int n, const double* matrix, double* rhs,
                              const int* pivots) noexcept nogil
