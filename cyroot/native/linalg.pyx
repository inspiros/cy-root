# distutils: language=c++
"""SciPy LAPACK provider, resolved at import and called directly without Python.

matrix is a private n-by-n column-major allocation, rhs is a private length-n
vector and pivots has n C ints. LAPACK overwrites all three. The caller checks
dimensions and finite input and handles info; no least-squares fallback occurs.
"""
from scipy.linalg.cython_lapack cimport dgesv, dgetrf, dgetrs


cdef int square_solve(int n, double* matrix, double* rhs, int* pivots) noexcept nogil:
    cdef int one = 1, info = 0
    dgesv(&n, &one, matrix, &n, pivots, rhs, &n, &info)
    return info


cdef int square_factor(int n, double* matrix, int* pivots) noexcept nogil:
    cdef int info = 0
    dgetrf(&n, &n, matrix, &n, pivots, &info)
    return info


cdef int square_solve_factored(int n, const double* matrix, double* rhs,
                              const int* pivots) noexcept nogil:
    cdef int one = 1, info = 0
    cdef char operation = 'N'
    # LAPACK's dgetrs treats LU and pivots as inputs. Its Cython declaration
    # predates const qualifiers, hence the casts at this narrow boundary.
    dgetrs(&operation, &n, &one, <double*>matrix, &n, <int*>pivots,
           rhs, &n, &info)
    return info
