from .fptr cimport DoubleScalarFPtr

################################################################################
# Base Class
################################################################################
cdef class DerivativeApproximation(DoubleScalarFPtr):
    cdef DoubleScalarFPtr f
    cdef double eval2(self, double x, double f_x) except * nogil
    cpdef double eval_with_f_val(self, double x, double f_x)

################################################################################
# Finite Difference
################################################################################
cdef double finite_difference_kernel(
        DoubleScalarFPtr f,
        double x,
        double f_x,
        double h=*,
        int order=*,
        int kind=*) nogil

cdef class FiniteDifference(DerivativeApproximation):
    cdef public int order, kind
    cdef public double h
