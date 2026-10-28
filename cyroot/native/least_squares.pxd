"""Draft immutable DGELSY configuration; caller supplies disjoint native spans."""
from libc.stddef cimport size_t

cdef class LeastSquaresPlan:
    cdef int _m, _n, _ldb, _lwork
    cdef size_t _matrix_size, _work_size, _memory_bytes
    cdef double _rcond
    cdef bint _initialized, _constructing
    cdef void _ready(self) except *
    # Trusted A[m*n], b[m], out[n], scratch[work_size], pivots[n], rank/info.
    # Matrix is column-major. All spans must be valid, live and disjoint.
    # Distinct work/output spans allow concurrent use of this immutable plan.
    # 0=ok, 1=nonfinite, 2=backend_failure, 3=invalid_plan/null pointer.
    # Failure preserves out, reports rank=-1, and retains LAPACK info if called.
    cdef int compute(self, const double* matrix, const double* rhs, double* out,
                     double* scratch, int* pivots, int* rank,
                     int* info) noexcept nogil
