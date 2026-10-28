"""External consumer of the retained ndarray derivative Cython entry points."""
import numpy as np
cimport numpy as np

from cyroot.fptr cimport NdArrayFPtr, PyNdArrayFPtr
from cyroot.vector_derivative_approximation cimport (
    GeneralizedFiniteDifference, VectorDerivativeApproximation,
    generalized_finite_difference_kernel)


def kernel(function, x, base, h, int order=1, int kind=0):
    cdef NdArrayFPtr callback = PyNdArrayFPtr.from_f(function)
    return generalized_finite_difference_kernel(
        callback, np.asarray(x, dtype=np.float64),
        np.asarray(base, dtype=np.float64), np.asarray(h, dtype=np.float64),
        order, kind)


def compiled_evals(function, x, base, h=.25, order=3, kind=0):
    cdef GeneralizedFiniteDifference derivative = GeneralizedFiniteDifference(
        function, h, order, kind)
    cdef VectorDerivativeApproximation parent = derivative
    cdef NdArrayFPtr as_function = derivative
    cdef np.ndarray point = np.asarray(x, dtype=np.float64)
    cdef np.ndarray value = np.asarray(base, dtype=np.float64)
    uncached = as_function.eval(point)
    cached = parent.eval2(point, value)
    exposed = parent.eval_with_f_val(point, value)
    return uncached, cached, exposed, derivative.n_f_calls
