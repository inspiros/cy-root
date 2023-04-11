from .fptr cimport DoubleScalarFPtr

################################################################################
# Base Class
################################################################################
cdef class DerivativeApproximation(DoubleScalarFPtr):
    cdef DoubleScalarFPtr f
    cpdef double eval(self, double x) except *
    cpdef double eval_with_f_val(self, double x, double f_x) except *

################################################################################
# Finite Difference
################################################################################
cdef double finite_difference_kernel(
        DoubleScalarFPtr f,
        double x,
        double f_x,
        double h=*,
        int order=*,
        int kind=*)

cdef class FiniteDifference(DerivativeApproximation):
    cdef public int order, kind
    cdef public double h
    cpdef double eval(self, double x) except *
    cpdef double eval_with_f_val(self, double x, double f_x) except *
