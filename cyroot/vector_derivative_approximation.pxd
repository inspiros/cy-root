cimport numpy as np

from .fptr cimport NdArrayFPtr

################################################################################
# Base Class
################################################################################
cdef class VectorDerivativeApproximation(NdArrayFPtr):
    cdef NdArrayFPtr F
    cpdef np.ndarray eval(self, np.ndarray x)
    cpdef np.ndarray eval_with_f_val(self, np.ndarray x, np.ndarray F_x)

################################################################################
# Finite Difference
################################################################################
cdef np.ndarray generalized_finite_difference_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=1] F_x,
        np.ndarray[np.float64_t, ndim=1] h,
        int order=*,
        int kind=*)

cdef class GeneralizedFiniteDifference(VectorDerivativeApproximation):
    cdef public int order, kind
    cdef public np.ndarray h
    cpdef np.ndarray eval(self, np.ndarray x)
    cpdef np.ndarray eval_with_f_val(self, np.ndarray x, np.ndarray F_x)
