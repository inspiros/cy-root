# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Rank-aware rectangular linear solves, separate from root solver policies."""
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport isfinite, NAN
from libc.stddef cimport size_t
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from cpython.buffer cimport PyObject_CheckBuffer
from scipy.linalg.cython_lapack cimport dgelsy
from .buffers cimport OwnedDoubleBuffer, VectorView, MatrixView
from ._buffers cimport DoubleBuffer
from .higher_derivatives import _checked_integer

# Bound the supported domain before entering LP64 LAPACK workspace heuristics.
# This conservative preview limit leaves integer headroom for blocked queries.
MAX_DIMENSION = 32767


class LeastSquaresResult(NamedTuple):
    """Finite linear solution/rank status, not a nonlinear convergence result."""
    solution: tuple
    rank: object
    linear_status: int
    status: str


def _validate_workspace_query(value, info, minimum):
    """Validate the LAPACK query before converting its double to a C int."""
    if info != 0:
        raise RuntimeError('DGELSY workspace query failed.')
    # The backend produces a binary64 scalar, not an arbitrary user object.
    if not isfinite(value) or value < minimum or value > INT_MAX:
        raise ValueError('DGELSY workspace query is outside native limits.')
    result = int(value)
    if value != result:
        raise ValueError('DGELSY workspace query must be an integer.')
    return result


cdef double _query(int m, int n, double rcond, int* info) noexcept nogil:
    cdef int one = 1, ldb = m if m > n else n, query = -1, rank = -1, pivot = 0
    cdef double a = 0., b = 0., work = NAN
    # The documented LWORK=-1 path calculates size only and returns before
    # referencing A/B/JPVT. No full matrix allocation is needed for a query.
    dgelsy(&m, &n, &one, &a, &m, &b, &ldb, &pivot, &rcond,
           &rank, &work, &query, info)
    return work


cdef bint _finite(const double* values, size_t count) noexcept nogil:
    cdef size_t i
    for i in range(count):
        if not isfinite(values[i]):return False
    return True


cdef void _pack_matrix(object source, double* out, size_t m, size_t n) except *:
    cdef DoubleBuffer guard
    cdef MatrixView view
    cdef size_t i, j
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric matrix.')
    if PyObject_CheckBuffer(source):
        guard = DoubleBuffer(source)
        view = guard.matrix_view()
        if view.rows() != m or view.columns() != n:
            raise ValueError('Matrix shape must match plan rows and columns.')
        for j in range(n):
            for i in range(m):out[i+j*m] = view.get(i, j)
    else:
        if len(source) != <Py_ssize_t>m:
            raise ValueError('Matrix shape must match plan rows and columns.')
        for i in range(m):
            row = source[i]
            if isinstance(row, (str, bytes, bytearray)) or len(row) != <Py_ssize_t>n:
                raise ValueError('Matrix shape must match plan rows and columns.')
            for j in range(n):out[i+j*m] = row[j]


cdef void _pack_rhs(object source, double* out, size_t m) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric right-hand side vector.')
    if PyObject_CheckBuffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != m:
            raise ValueError('Right-hand side length must match plan rows.')
        for i in range(m):out[i] = view.get(i)
    else:
        if len(source) != <Py_ssize_t>m:
            raise ValueError('Right-hand side length must match plan rows.')
        for i in range(m):out[i] = source[i]


cdef class LeastSquaresPlan:
    """Immutable shape/rank policy; compute reuses caller-owned native workspaces."""
    def __init__(self, rows, columns, *, rcond=1e-12, workspace_bytes=64*1024*1024):
        cdef int m, n, info = 0
        cdef double threshold, query
        cdef vector[int] capacity
        if self._constructing or self._initialized:
            raise RuntimeError('Least-squares plans cannot be reinitialized.')
        self._constructing = True
        try:
            m_value = _checked_integer(rows, 'rows')
            n_value = _checked_integer(columns, 'columns')
            limit = _checked_integer(workspace_bytes, 'workspace_bytes')
            if m_value == 0 or n_value == 0 or m_value > 32767 or n_value > 32767:
                raise ValueError('Rows and columns must be between 1 and 32767 in this preview.')
            # Avoid overflow inside LAPACK's workspace arithmetic before query.
            minimum = max(min(m_value, n_value)+3*n_value+1, 2*min(m_value, n_value)+1)
            matrix_size = m_value*n_value
            padded_rhs = max(m_value, n_value)
            fixed_count = 2*matrix_size+m_value+n_value+padded_rhs
            if (minimum > INT_MAX or n_value > capacity.max_size() or
                    fixed_count+minimum > OwnedDoubleBuffer.max_size() or
                    (fixed_count+minimum)*sizeof(double)+n_value*sizeof(int) > limit):
                raise ValueError('Requested least-squares workspace exceeds native limits or workspace_bytes.')
            if isinstance(rcond, (str, bytes, bytearray, bool)):
                raise TypeError('rcond must be a real scalar strictly between zero and one.')
            if getattr(getattr(rcond, 'dtype', None), 'kind', None) in ('b', 'c'):
                raise TypeError('rcond must be a real scalar strictly between zero and one.')
            try:
                len(rcond)
            except TypeError:
                pass
            else:
                raise TypeError('rcond must be a real scalar strictly between zero and one.')
            threshold = rcond
            if not isfinite(threshold) or threshold <= 0. or threshold >= 1.:
                raise ValueError('rcond must be finite and strictly between zero and one.')
            m, n = m_value, n_value
            with nogil:query = _query(m, n, threshold, &info)
            lwork = _validate_workspace_query(query, info, minimum)
            count = fixed_count+lwork
            memory_bytes = count*sizeof(double)+n_value*sizeof(int)
            if count > OwnedDoubleBuffer.max_size() or memory_bytes > limit:
                raise ValueError('Queried least-squares workspace exceeds native limits or workspace_bytes.')
            self._m, self._n, self._ldb, self._lwork = m, n, padded_rhs, lwork
            self._matrix_size = matrix_size
            self._work_size = matrix_size+padded_rhs+lwork
            self._memory_bytes = memory_bytes
            self._rcond = threshold
            self._initialized = True
        finally:
            self._constructing = False

    cdef void _ready(self) except *:
        if self._constructing:raise RuntimeError('This plan is being initialized.')
        if not self._initialized:raise RuntimeError('This plan is not initialized.')

    cdef int compute(self, const double* matrix, const double* rhs, double* out,
                     double* scratch, int* pivots, int* rank, int* info) noexcept nogil:
        cdef int m = self._m, n = self._n, one = 1, computed_rank = -1
        cdef int ldb = self._ldb, lwork = self._lwork, backend_info = 0
        cdef double rcond = self._rcond
        cdef double* a
        cdef double* b
        cdef double* work
        if rank != NULL:rank[0] = -1
        if info != NULL:info[0] = 0
        if (not self._initialized or matrix == NULL or rhs == NULL or out == NULL or
                scratch == NULL or pivots == NULL or rank == NULL or info == NULL):
            return 3
        if not _finite(matrix, self._matrix_size) or not _finite(rhs, <size_t>m):return 1
        a = scratch
        b = a+self._matrix_size
        work = b+ldb
        memcpy(a, matrix, self._matrix_size*sizeof(double))
        memset(b, 0, <size_t>ldb*sizeof(double))
        memcpy(b, rhs, <size_t>m*sizeof(double))
        memset(pivots, 0, <size_t>n*sizeof(int))
        dgelsy(&m, &n, &one, a, &m, b, &ldb, pivots, &rcond,
               &computed_rank, work, &lwork, &backend_info)
        info[0] = backend_info
        if backend_info != 0 or computed_rank < 0 or computed_rank > m or computed_rank > n:
            return 2
        if not _finite(a, self._matrix_size) or not _finite(b, <size_t>n):return 1
        memcpy(out, b, <size_t>n*sizeof(double))
        rank[0] = computed_rank
        return 0

    def solve(self, matrix, rhs):
        """Allocate a private solve; native loops should call compute with reused scratch."""
        cdef OwnedDoubleBuffer* storage = NULL
        cdef vector[int] pivots
        cdef double* a
        cdef double* b
        cdef double* out
        cdef double* scratch
        cdef int rank = -1, info = 0, status
        cdef size_t i
        self._ready()
        try:
            storage = new OwnedDoubleBuffer(self._matrix_size+self._m+self._n+self._work_size)
            pivots.resize(self._n)
            a = storage.data()
            b = a+self._matrix_size
            out = b+self._m
            scratch = out+self._n
            for i in range(self._n):out[i] = NAN
            _pack_matrix(matrix, a, self._m, self._n)
            _pack_rhs(rhs, b, self._m)
            with nogil:status = self.compute(a, b, out, scratch, pivots.data(), &rank, &info)
            return LeastSquaresResult(tuple([out[i] for i in range(self._n)]),
                rank if status == 0 else None, info,
                'ok' if status == 0 else 'nonfinite' if status == 1 else
                'backend_failure' if status == 2 else 'invalid_plan')
        finally:
            if storage != NULL:del storage

    @property
    def rows(self):self._ready();return self._m
    @property
    def columns(self):self._ready();return self._n
    @property
    def rcond(self):self._ready();return self._rcond
    @property
    def lwork(self):self._ready();return self._lwork
    @property
    def work_size(self):self._ready();return self._work_size
    @property
    def pivot_size(self):self._ready();return self._n
    @property
    def memory_bytes(self):self._ready();return self._memory_bytes
