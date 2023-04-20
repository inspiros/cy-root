ctypedef fused real:
    int
    unsigned int
    long
    unsigned long
    float
    double

ctypedef fused numeric:
    int
    unsigned int
    long
    unsigned long
    float
    double
    double complex

cdef bint isclose(double a, double b, double rtol=*, double atol=*) nogil
cdef bint cisclose(double complex a, double complex b, double rtol=*, double atol=*) nogil
cdef real min(real a, real b) nogil
cdef real max(real a, real b) nogil

cdef extern from '<math.h>' nogil:
    double fabs(double)
    double sqrt(double)

cdef extern from '<complex>' nogil:
    double cabs 'abs'(double complex)
    double complex csqrt 'sqrt'(double complex)

cdef extern from * nogil:
    """
    #include <complex>

    template <typename T>
    int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }

    template <typename T>
    std::complex<T> csign(std::complex<T> val) {
        return val / std::norm(val);
    }

    unsigned long factorial(unsigned int n) {
        unsigned long f = 1;
        for (unsigned int i = 1; i < n + 1; i++)
            f *= i;
        return f;
    }

    unsigned long binomial_coef(unsigned long n, unsigned long k) {
        unsigned long bin_coef = 1;
        unsigned int i;
        if (k <= n / 2) {
            for (i = 0; i < k; i++)
                bin_coef *= n - i;
            return bin_coef / factorial(k);
        }
        for (i = 0; i < n - k; i++)
            bin_coef *= n - i;
        return bin_coef / factorial(n - k);
    }
    """
    int sign(double)
    double complex csign(double complex)
    unsigned long factorial(unsigned int)
    unsigned long binomial_coef(unsigned long, unsigned long)
