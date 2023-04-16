from .scalar_ops cimport real, numeric

cdef bint equal(real[:] a, real[:] b) nogil
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
cdef double[:] permute(double[:] xs, unsigned long[:] inds) nogil
cdef numeric sum(numeric[:] xs) nogil
cdef numeric prod(numeric[:] xs) nogil
cdef double complex cprod(double complex[:] xs) nogil
cdef double mean(double[:] xs) nogil
cdef double min(double[:] xs) nogil
cdef double max(double[:] xs) nogil
cdef unsigned long argmin(double[:] xs) nogil
cdef unsigned long argmax(double[:] xs) nogil
cdef (unsigned long, unsigned long) argminmax(double[:] xs) nogil
cdef void sort(double[::1] xs) nogil
cdef unsigned long[:] argsort(double[:] xs, bint reverse=*) nogil
