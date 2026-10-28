from .scalar_types cimport bool_t, real_t, numeric_t

cdef bint equal(real_t[:] a, real_t[:] b) nogil
cdef bint cequal(double complex[:] a, double complex[:] b) nogil
cdef bint allclose(double[:] a, double[:] b, double rtol=*, double atol=*) nogil
cdef bint callclose(double complex[:] a, double complex[:] b, double rtol=*, double atol=*) nogil
cdef bint anyclose(double[:] a, double[:] b, double rtol=*, double atol=*) nogil
cdef bint canyclose(double complex[:] a, double complex[:] b, double rtol=*, double atol=*) nogil
cdef int[:] sign(double[:] xs) nogil
cdef double complex[:] csign(double complex[:] xs) nogil
cdef double[:] fabs(double[:] xs) nogil
cdef double[:] cabs(double complex[:] xs) nogil
cdef double width(double[:] xs) nogil
cdef double cwidth(double complex[:] xs) nogil
cdef double[:] sqrt(double[:] xs) nogil
cdef double complex[:] csqrt(double complex[:] xs) nogil
cdef double norm(double[:] xs, double order=*) nogil
cdef double cnorm(double complex[:] xs, double order=*) nogil
cdef double[:] permute(double[:] xs, size_t[:] inds) nogil
cdef numeric_t sum(numeric_t[:] xs) nogil
cdef numeric_t prod(numeric_t[:] xs) nogil
cdef double complex cprod(double complex[:] xs) nogil
cdef double mean(double[:] xs) nogil
cdef double min(double[:] xs) nogil
cdef double max(double[:] xs) nogil
cdef size_t argmin(double[:] xs) nogil
cdef size_t argmax(double[:] xs) nogil
cdef (size_t, size_t) argminmax(double[:] xs) nogil
cdef void sort(double[::1] xs) nogil
cdef size_t[:] argsort(double[:] xs, bool_t reverse=*) nogil
