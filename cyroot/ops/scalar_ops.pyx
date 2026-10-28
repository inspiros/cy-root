# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from libc cimport math

cdef inline bint isclose(double a, double b, double rtol=1e-5, double atol=1e-8) noexcept nogil:
    if a == b:
        return True
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return math.fabs(a - b) <= atol + rtol * math.fabs(b)

cdef inline bint cisclose(double complex a, double complex b, double rtol=1e-5, double atol=1e-8) noexcept nogil:
    return isclose(a.real, b.real, rtol, atol) and isclose(a.imag, b.imag, rtol, atol)

cdef inline real_t min(real_t a, real_t b) noexcept nogil:
    return b if a > b else a

cdef inline real_t max(real_t a, real_t b) noexcept nogil:
    return b if a < b else a
