from .scalar_types cimport bool_t, real_t

cdef bint isclose(double a, double b, double rtol=*, double atol=*) noexcept nogil
cdef bint cisclose(double complex a, double complex b, double rtol=*, double atol=*) noexcept nogil
cdef real_t min(real_t a, real_t b) noexcept nogil
cdef real_t max(real_t a, real_t b) noexcept nogil

cdef extern from '<math.h>' nogil:
    double fabs(double)
    double sqrt(double)

cdef extern from '<complex>' nogil:
    double cabs 'abs'(double complex)
    double complex csqrt 'sqrt'(double complex)

cdef extern from * nogil:
    """
#include <complex>

template<typename T>
CYTHON_INLINE int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename T>
CYTHON_INLINE std::complex<T> csign(std::complex<T> val) {
    return val / std::norm(val);
}

template<typename T = std::size_t, typename R = T>
CYTHON_INLINE R factorial(T n) {
    R f = 1;
    std::size_t i;
#pragma unroll
    for (i = 1; i < n + 1; i++)
        f *= static_cast<R>(i);
    return f;
}

template<typename T = std::size_t>
CYTHON_INLINE T binomial_coef(T n, T k) {
    T bin_coef = 1;
    std::size_t i;
    if (k <= n / 2) {
#pragma unroll
        for (i = 0; i < k; i++)
            bin_coef *= n - static_cast<T>(i);
        return bin_coef / factorial(k);
    }
#pragma unroll
    for (i = 0; i < n - k; i++)
        bin_coef *= n - static_cast<T>(i);
    return bin_coef / factorial(n - k);
}
    """
    int sign(double)
    double complex csign(double complex)
    inline T binomial_coef[T](T, T)
