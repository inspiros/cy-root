# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from cython cimport view
from libcpp.algorithm cimport sort as cpp_sort
from libcpp.vector cimport vector
from libc cimport math

cdef extern from '<complex>':
    double _cabs 'abs'(double complex) nogil

cdef inline bint fallclose(double[:] a, double[:] b, double rtol=1e-5, double atol=1e-8) nogil:
    cdef unsigned long i
    for i in range(a.shape[0]):
        if ((math.isinf(a[i]) and not math.isinf(b[i])) or
                (math.isinf(b[i]) and not math.isinf(a[i])) or
                math.fabs(a[i] - b[i]) > atol + rtol * math.fabs(b[i])):
            return False
    return True

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
        res[i] = _cabs(xs[i])
    return res

cdef inline double fabs_width(double[:] xs) nogil:
    cdef unsigned long argmin_i, argmax_i
    argmin_i, argmax_i = fargminmax(xs)
    return xs[argmax_i] - xs[argmin_i]

cdef inline double cabs_width(double complex[:] xs) nogil:
    cdef unsigned long argmin_i, argmax_i
    cdef double[:] xs_abs = cabs(xs)
    argmin_i, argmax_i = fargminmax(xs_abs)
    return xs_abs[argmax_i] - xs_abs[argmin_i]

cdef inline double[:] fpermute(double[:] xs, unsigned long[:] inds) nogil:
    cdef unsigned long i
    cdef double[:] res
    with gil:
        res = view.array(shape=(xs.shape[0],), itemsize=sizeof(double), format='d')
    for i in range(xs.shape[0]):
        res[i] = xs[inds[i]]
    return res

cdef inline double fsum(double[:] xs) nogil:
    cdef unsigned long i
    cdef double sum = 0.
    for i in range(xs.shape[0]):
        sum += xs[i]
    return sum

cdef inline double fmean(double[:] xs) nogil:
    if xs.shape[0]:
        return fsum(xs) / xs.shape[0]
    return math.NAN

cdef inline double fmin(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i
    cdef double minimum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
    return minimum

cdef inline double fmax(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i
    cdef double maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] > maximum:
            maximum = xs[i]
    return maximum

cdef inline unsigned long fargmin(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i, argmin_i = 0
    cdef double minimum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] < minimum:
            minimum = xs[i]
            argmin_i = i
    return argmin_i

cdef inline unsigned long fargmax(double[:] xs) nogil:
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    cdef unsigned long i, argmax_i = 0
    cdef double maximum = xs[0]
    for i in range(1, xs.shape[0]):
        if xs[i] > maximum:
            maximum = xs[i]
            argmax_i = i
    return argmax_i

cdef inline (unsigned long, unsigned long) fargminmax(double[:] xs) nogil:
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

cdef inline void fsort(double[::1] xs) nogil:
    cpp_sort(&xs[0], (&xs[0]) + xs.shape[0])

cdef struct _IndexedDouble:
    unsigned long id
    double val

cdef bint _ascending_cmp(_IndexedDouble &lhs, _IndexedDouble &rhs) nogil:
    return lhs.val < rhs.val

cdef bint _descending_cmp(_IndexedDouble &lhs, _IndexedDouble &rhs) nogil:
    return lhs.val > rhs.val

cdef inline unsigned long[:] fargsort(double[:] xs, bint reverse=False) nogil:
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