# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from cython cimport view
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.vector cimport vector
from libc cimport math

from . cimport scalar_ops as sops

cdef inline bint equal(real[:] a, real[:] b) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef unsigned long i
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True

cdef inline bint cequal(double complex[:] a, double complex[:] b) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef unsigned long i
    for i in range(a.shape[0]):
        if a[i].real != b[i].real or a[i].imag != b[i].imag:
            return False
    return True

cdef inline bint allclose(double[:] a, double[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef unsigned long i
    for i in range(a.shape[0]):
        if not sops.isclose(a[i], b[i], rtol, atol):
            return False
    return True

cdef inline bint callclose(double complex[:] a, double complex[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef unsigned long i
    for i in range(a.shape[0]):
        if not sops.cisclose(a[i], b[i], rtol, atol):
            return False
    return True

cdef inline bint anyclose(double[:] a, double[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef unsigned long i
    for i in range(a.shape[0]):
        if sops.isclose(a[i], b[i], rtol, atol):
            return True
    return False

cdef inline bint canyclose(double complex[:] a, double complex[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    if a.shape[0] != b.shape[0]:
        return False
    cdef unsigned long i
    for i in range(a.shape[0]):
        if sops.cisclose(a[i], b[i], rtol, atol):
            return True
    return False

cdef inline int[:] sign(double[:] xs) nogil:
    cdef unsigned long i
    cdef int[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(int), format='i')
    for i in range(xs.shape[0]):
        res[i] = sops.sign(xs[i])
    return res

cdef inline double complex[:] csign(double complex[:] xs) nogil:
    cdef unsigned long i
    cdef double complex[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double complex), format='c')
    for i in range(xs.shape[0]):
        res[i] = sops.csign(xs[i])
    return res

cdef inline double[:] fabs(double[:] xs) nogil:
    cdef unsigned long i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double), format='d')
    for i in range(xs.shape[0]):
        res[i] = math.fabs(xs[i])
    return res

cdef inline double[:] cabs(double complex[:] xs) nogil:
    cdef unsigned long i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double), format='d')
    for i in range(xs.shape[0]):
        res[i] = sops.cabs(xs[i])
    return res

cdef inline double width(double[:] xs) nogil:
    cdef unsigned long argmin_i, argmax_i
    argmin_i, argmax_i = argminmax(xs)
    return xs[argmax_i] - xs[argmin_i]

cdef inline double cwidth(double complex[:] xs) nogil:
    cdef unsigned long argmin_i, argmax_i
    cdef double[:] xs_abs = cabs(xs)
    argmin_i, argmax_i = argminmax(xs_abs)
    return xs_abs[argmax_i] - xs_abs[argmin_i]

cdef inline double[:] sqrt(double[:] xs) nogil:
    cdef unsigned long i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double), format='d')
    for i in range(xs.shape[0]):
        res[i] = sops.sqrt(xs[i])
    return res

cdef inline double complex[:] csqrt(double complex[:] xs) nogil:
    cdef unsigned long i
    cdef double complex[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double complex), format='C')
    for i in range(xs.shape[0]):
        res[i] = sops.csqrt(xs[i])
    return res

cdef inline double norm(double[:] xs, double order=2) nogil:
    if order == math.INFINITY:
        return max(fabs(xs))
    if order == -math.INFINITY:
        return min(fabs(xs))
    cdef unsigned long i
    cdef double res = 0
    if order == 0:
        for i in range(xs.shape[0]):
            res += xs[i] != 0
        return res
    for i in range(xs.shape[0]):
        res += sops.fabs(xs[i]) ** order
    return res ** (1 / order)

cdef inline double cnorm(double complex[:] xs, double order=2) nogil:
    if order == math.INFINITY:
        return max(cabs(xs))
    if order == -math.INFINITY:
        return min(cabs(xs))
    cdef unsigned long i
    cdef double res = 0
    if order == 0:
        for i in range(xs.shape[0]):
            res += xs[i] != 0
        return res
    for i in range(xs.shape[0]):
        res += sops.cabs(xs[i]) ** order
    return res ** (1 / order)

cdef inline double[:] permute(double[:] xs, unsigned long[:] inds) nogil:
    cdef unsigned long i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double), format='d')
    for i in range(xs.shape[0]):
        res[i] = xs[inds[i]]
    return res

cdef inline numeric sum(numeric[:] xs) nogil:
    cdef unsigned long i
    cdef numeric res = 0
    for i in range(xs.shape[0]):
        res += xs[i]
    return res

cdef inline numeric prod(numeric[:] xs) nogil:
    cdef numeric res = 1
    cdef unsigned long i
    for i in range(xs.shape[0]):
        res *= xs[i]
    return res

cdef inline double complex cprod(double complex[:] xs) nogil:
    cdef double complex res = 1
    cdef unsigned long i
    for i in range(xs.shape[0]):
        res *= xs[i]
    return res

cdef inline double mean(double[:] xs) nogil:
    return sum[double](xs) / xs.shape[0]

cdef inline double min(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i
    cdef double minimum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
    return minimum

cdef inline double max(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i
    cdef double maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] > maximum:
            maximum = xs[i]
    return maximum

cdef inline unsigned long argmin(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i, argmin_i = 0
    cdef double minimum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
            argmin_i = i
    return argmin_i

cdef inline unsigned long argmax(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i, argmax_i = 0
    cdef double maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] > maximum:
            maximum = xs[i]
            argmax_i = i
    return argmax_i

cdef inline (unsigned long, unsigned long) argminmax(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i, argmin_i = 0, argmax_i = 0
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
    unsigned long id
    double val

cdef bint _ascending_cmp(_IndexedDouble &lhs, _IndexedDouble &rhs) nogil:
    return lhs.val < rhs.val

cdef bint _descending_cmp(_IndexedDouble &lhs, _IndexedDouble &rhs) nogil:
    return lhs.val > rhs.val

cdef inline unsigned long[:] argsort(double[:] xs, bint reverse=False) nogil:
    cdef unsigned long i
    cdef vector[_IndexedDouble] indexed_xs = vector[_IndexedDouble](xs.shape[0])
    for i in range(xs.shape[0]):
        indexed_xs[i].id = i
        indexed_xs[i].val = xs[i]
    if not reverse:
        cpp_sort(indexed_xs.begin(), indexed_xs.end(), &_ascending_cmp)
    else:
        cpp_sort(indexed_xs.begin(), indexed_xs.end(), &_descending_cmp)

    cdef unsigned long[:] inds
    with gil:
        inds = view.array(shape=(xs.shape[0],), itemsize=sizeof(long), format='L')
    for i in range(xs.shape[0]):
        inds[i] = indexed_xs[i].id
    return inds
