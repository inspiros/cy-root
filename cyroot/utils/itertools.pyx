# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from cython cimport view
import numpy as np
cimport numpy as np

from ..ops cimport vector_ops

__all__ = [
    'product',
]

cdef inline void product_kernel(
        unsigned int[:] ns,
        unsigned int[:, :] out,
        unsigned int d=0) nogil:
    if d >= ns.shape[0]:
        return
    cdef unsigned int n = ns[d]
    cdef unsigned int step = vector_ops.prod[np.uint32_t](ns[d + 1:])
    cdef unsigned int i
    for i in range(out.shape[0]):
        out[i, d] = <unsigned int> (i / step) % n
    product_kernel(ns, out, d + 1)

cpdef unsigned int[:, :] product(unsigned int[:] ns):
    cdef np.ndarray[np.uint32_t, ndim=2] empty = np.empty((0, 0), dtype=np.uint32)
    cdef unsigned int i
    for i in range(ns.shape[0]):
        if ns[i] == 0:
            return empty

    cdef unsigned int[:, :] out = view.array(
        shape=(vector_ops.prod[np.uint32_t](ns), ns.shape[0]),
        itemsize=sizeof(int),
        format='I')
    product_kernel(ns, out, 0)
    return out
