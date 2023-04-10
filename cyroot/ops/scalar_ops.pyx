# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from libc cimport math

cdef inline bint fisclose(double a, double b, double rtol=1e-5, double atol=1e-8) nogil:
    if math.isinf(b) or math.isinf(a):
        return math.isinf(a) and math.isinf(b)
    return math.fabs(a - b) <= atol + rtol * math.fabs(b)

cdef inline bint cisclose(double complex a, double complex b, double rtol=1e-5, double atol=1e-8) nogil:
    return fisclose(a.real, b.real, rtol, atol) and fisclose(a.imag, b.imag, rtol, atol)