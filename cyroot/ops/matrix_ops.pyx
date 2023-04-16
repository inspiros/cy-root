# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

import numpy as np
import scipy as sp

cdef inline bint is_square(np.ndarray[np.float64_t, ndim=2] A):
    return A.shape[0] == A.shape[1]

cdef inline bint is_singular(np.ndarray[np.float64_t, ndim=2] A):
    return np.linalg.cond(A) > 1 / np.finfo(A.dtype).eps

cdef np.ndarray inv(np.ndarray[np.float64_t, ndim=2] A,
                    np.ndarray b=None,
                    int method=1,
                    bint force=False):
    cdef unsigned long m = A.shape[0]
    cdef unsigned long n = A.shape[1]
    cdef np.ndarray A_inv
    if method == 0:  # inv
        if (not is_square(A) or is_singular(A)) and not force:
            method += 1
        else:
            A_inv = sp.linalg.inv(A)
            if b is not None:
                return A_inv.dot(b)
            return A_inv
    if method == 1:  # solve
        if (not is_square(A) or is_singular(A)) and not force:
            method += 2
        else:
            return sp.linalg.solve(A, b if b is not None else np.eye(m))
    if method == 2:  # lu_solve
        if (not is_square(A) or is_singular(A)) and not force:
            method += 1
        else:
            return sp.linalg.lu_solve(sp.linalg.lu_factor(A), b if b is not None else np.eye(m))
    if method == 3:  # pinv
        A_inv = sp.linalg.pinv(A)
        if b is not None:
            return A_inv.dot(b)
        return A_inv
    if method == 4:  # lstsq
        return sp.linalg.lstsq(A, b if b is not None else np.eye(m))[0]
    raise NotImplementedError(f'No implementation for method={method}.')
