# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from cython cimport view
from cython.parallel import prange
from libc cimport math
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.vector cimport vector
import numpy as np

from .scalar_types cimport format_character
from . cimport scalar_ops as sops

cdef bint equal(real_t[:] a, real_t[:] b) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef size_t i
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True

cdef bint cequal(double complex[:] a, double complex[:] b) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef size_t i
    for i in range(a.shape[0]):
        if a[i].real != b[i].real or a[i].imag != b[i].imag:
            return False
    return True

cdef bint allclose(double[:] a, double[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef size_t i
    for i in range(a.shape[0]):
        if not sops.isclose(a[i], b[i], rtol, atol):
            return False
    return True

cdef bint callclose(double complex[:] a, double complex[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef size_t i
    for i in range(a.shape[0]):
        if not sops.cisclose(a[i], b[i], rtol, atol):
            return False
    return True

cdef bint anyclose(double[:] a, double[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef size_t i
    for i in range(a.shape[0]):
        if sops.isclose(a[i], b[i], rtol, atol):
            return True
    return False

cdef bint canyclose(double complex[:] a, double complex[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef size_t i
    for i in range(a.shape[0]):
        if sops.cisclose(a[i], b[i], rtol, atol):
            return True
    return False

cdef int[:] sign(double[:] xs) nogil:
    cdef size_t i
    cdef int[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(int),
                         format=format_character[int]())
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = sops.sign(xs[i])
    return res

cdef double complex[:] csign(double complex[:] xs) nogil:
    cdef size_t i
    cdef double complex[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double complex),
                         format='C')
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = sops.csign(xs[i])
    return res

cdef double[:] fabs(double[:] xs) nogil:
    cdef size_t i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double),
                         format=format_character[double]())
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = math.fabs(xs[i])
    return res

cdef double[:] cabs(double complex[:] xs) nogil:
    cdef size_t i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double),
                         format=format_character[double]())
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = sops.cabs(xs[i])
    return res

cdef double width(double[:] xs) nogil:
    cdef size_t argmin_i, argmax_i
    argmin_i, argmax_i = argminmax(xs)
    return xs[argmax_i] - xs[argmin_i]

cdef double cwidth(double complex[:] xs) nogil:
    cdef size_t argmin_i, argmax_i
    cdef double[:] xs_abs = cabs(xs)
    argmin_i, argmax_i = argminmax(xs_abs)
    return xs_abs[argmax_i] - xs_abs[argmin_i]

cdef double[:] sqrt(double[:] xs) nogil:
    cdef size_t i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double),
                         format=format_character[double]())
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = sops.sqrt(xs[i])
    return res

cdef double complex[:] csqrt(double complex[:] xs) nogil:
    cdef size_t i
    cdef double complex[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double complex),
                         format='C')
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = sops.csqrt(xs[i])
    return res

cdef double norm(double[:] xs, double order=2) nogil:
    if order == math.INFINITY:
        return max(fabs(xs))
    if order == -math.INFINITY:
        return min(fabs(xs))
    cdef size_t i
    cdef double res = 0
    if order == 0:
        for i in range(xs.shape[0]):
            res += xs[i] != 0
        return res
    for i in range(xs.shape[0]):
        res += sops.fabs(xs[i]) ** order
    return res ** (1 / order)

cdef double cnorm(double complex[:] xs, double order=2) nogil:
    if order == math.INFINITY:
        return max(cabs(xs))
    if order == -math.INFINITY:
        return min(cabs(xs))
    cdef size_t i
    cdef double res = 0
    if order == 0:
        for i in range(xs.shape[0]):
            res += xs[i] != 0
        return res
    for i in range(xs.shape[0]):
        res += sops.cabs(xs[i]) ** order
    return res ** (1 / order)

cdef double[:] permute(double[:] xs, size_t[:] inds) nogil:
    cdef size_t i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double),
                         format=format_character[double]())
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        res[i] = xs[inds[i]]
    return res

cdef numeric_t sum(numeric_t[:] xs) nogil:
    cdef size_t i
    cdef numeric_t res = 0
    for i in range(xs.shape[0]):
        res += xs[i]
    return res

cdef numeric_t prod(numeric_t[:] xs) nogil:
    cdef numeric_t res = 1
    cdef size_t i
    for i in range(xs.shape[0]):
        res *= xs[i]
    return res

cdef double complex cprod(double complex[:] xs) nogil:
    cdef double complex res = 1
    cdef size_t i
    for i in range(xs.shape[0]):
        res *= xs[i]
    return res

cdef double mean(double[:] xs) nogil:
    return sum[double](xs) / xs.shape[0]

cdef double min(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef size_t i
    cdef double minimum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
    return minimum

cdef double max(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef size_t i
    cdef double maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] > maximum:
            maximum = xs[i]
    return maximum

cdef size_t argmin(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef size_t i, argmin_i = 0
    cdef double minimum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
            argmin_i = i
    return argmin_i

cdef size_t argmax(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef size_t i, argmax_i = 0
    cdef double maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] > maximum:
            maximum = xs[i]
            argmax_i = i
    return argmax_i

cdef (size_t, size_t) argminmax(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef size_t i, argmin_i = 0, argmax_i = 0
    cdef double minimum = xs[0], maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
            argmin_i = i
        if xs[i] > maximum:
            maximum = xs[i]
            argmax_i = i
    return argmin_i, argmax_i

cdef inline void sort(double[::1] xs) nogil:
    cpp_sort(&xs[0], (&xs[0]) + xs.shape[0])

cdef struct _IndexedDouble:
    size_t id
    double val

cdef inline bool_t _ascending_cmp(_IndexedDouble &lhs, _IndexedDouble &rhs) noexcept nogil:
    return lhs.val < rhs.val

cdef inline bool_t _descending_cmp(_IndexedDouble &lhs, _IndexedDouble &rhs) noexcept nogil:
    return lhs.val > rhs.val

cdef size_t[:] argsort(double[:] xs, bool_t reverse=False) nogil:
    cdef size_t i
    cdef vector[_IndexedDouble] indexed_xs = vector[_IndexedDouble](xs.shape[0])
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        indexed_xs[i].id = i
        indexed_xs[i].val = xs[i]
    if not reverse:
        cpp_sort(indexed_xs.begin(), indexed_xs.end(), &_ascending_cmp)
    else:
        cpp_sort(indexed_xs.begin(), indexed_xs.end(), &_descending_cmp)

    cdef size_t[:] inds
    with gil:
        inds = np.empty(shape=(xs.shape[0],), dtype=np.uintp)
    for i in prange(xs.shape[0], nogil=True, schedule='static'):
        inds[i] = indexed_xs[i].id
    return inds
