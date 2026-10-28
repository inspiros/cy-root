"""Reusable no-GIL least-squares workspaces for tall and wide systems."""
from libcpp.vector cimport vector
from cyroot.native.buffers cimport OwnedDoubleBuffer
from cyroot.native.least_squares cimport LeastSquaresPlan

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


def solve(bint wide=False):
    """Solve twice with one allocation set; return copied results and diagnostics."""
    cdef int m = 2 if wide else 3
    cdef int n = 3 if wide else 2
    cdef LeastSquaresPlan plan = LeastSquaresPlan(m, n, rcond=1e-12)
    cdef OwnedDoubleBuffer matrix = OwnedDoubleBuffer(m*n)
    cdef OwnedDoubleBuffer rhs = OwnedDoubleBuffer(m)
    cdef OwnedDoubleBuffer output = OwnedDoubleBuffer(n)
    cdef OwnedDoubleBuffer scratch = OwnedDoubleBuffer(plan.work_size)
    cdef vector[int] pivots
    pivots.resize(plan.pivot_size)
    cdef int rank = -1
    cdef int info = 0
    cdef int status = -1
    cdef int gil_before, gil_after, repeat, i

    if wide:
        # A = [[1, 0, 1], [0, 1, 1]], stored by columns.
        matrix.set(0, 1.)
        matrix.set(3, 1.)
        matrix.set(4, 1.)
        matrix.set(5, 1.)
        rhs.set(0, 1.)
        rhs.set(1, 2.)
    else:
        # A = [[1, 0], [0, 1], [1, 1]], stored by columns.
        matrix.set(0, 1.)
        matrix.set(2, 1.)
        matrix.set(4, 1.)
        matrix.set(5, 1.)
        rhs.set(0, 1.)
        rhs.set(1, 2.)
        rhs.set(2, 4.)

    with nogil:
        gil_before = PyGILState_Check()
        for repeat in range(2):
            status = plan.compute(matrix.data(), rhs.data(), output.data(),
                                  scratch.data(), pivots.data(), &rank, &info)
            if status != 0:
                break
        gil_after = PyGILState_Check()

    copied = []
    for i in range(n):
        copied.append(output.get(i))
    return {
        'solution': tuple(copied),
        'rank': rank,
        'info': info,
        'status': status,
        'gil_held': bool(gil_before or gil_after),
    }
