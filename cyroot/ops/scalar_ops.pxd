cdef bint fisclose(double a, double b, double rtol=*, double atol=*) nogil
cdef bint cisclose(double complex a, double complex b, double rtol=*, double atol=*) nogil

cdef extern from '<math.h>' nogil:
    double fabs(double)
    double sqrt(double)

cdef extern from '<complex>' nogil:
    double cabs 'abs'(double complex)
    double complex csqrt 'sqrt'(double complex)

cdef extern from * nogil:
    """
    #pragma once
    #include <complex>

    template <typename T>
    int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }

    template <typename T>
    std::complex<T> csign(std::complex<T> val) {
        return val / std::norm(val);
    }
    """
    int sign(double)
    double complex csign(double complex)
