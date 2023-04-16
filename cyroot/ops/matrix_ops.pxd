cimport numpy as np

cdef np.ndarray inv(np.ndarray[np.float64_t, ndim=2] A, np.ndarray b=*, int method=*, bint force=*)
