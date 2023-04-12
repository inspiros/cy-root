from .scalar_ops cimport real

cdef bint fallclose(double[:] a, double[:] b, double rtol=*, double atol=*) nogil
cdef int[:] sign(double[:] xs) nogil
cdef double complex[:] csign(double complex[:] xs) nogil
cdef double[:] fabs(double[:] xs) nogil
cdef double[:] cabs(double complex[:] xs) nogil
cdef double fabs_width(double[:] xs) nogil
cdef double cabs_width(double complex[:] xs) nogil
cdef double[:] sqrt(double[:] xs) nogil
cdef double complex[:] csqrt(double complex[:] xs) nogil
cdef double norm(double[:] xs, double order=*) nogil
cdef double cnorm(double complex[:] xs, double order=*) nogil
cdef double[:] fpermute(double[:] xs, unsigned long[:] inds) nogil
cdef real sum(real[:] xs) nogil
cdef real prod(real[:] xs) nogil
cdef double complex cprod(double complex[:] xs) nogil
cdef double fmean(double[:] xs) nogil
cdef double fmin(double[:] xs) nogil
cdef double fmax(double[:] xs) nogil
cdef unsigned long fargmin(double[:] xs) nogil
cdef unsigned long fargmax(double[:] xs) nogil
cdef (unsigned long, unsigned long) fargminmax(double[:] xs) nogil
cdef void fsort(double[::1] xs) nogil
cdef unsigned long[:] fargsort(double[:] xs, bint reverse=*) nogil
