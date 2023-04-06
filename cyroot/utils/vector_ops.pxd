cdef bint fallclose(double[:] a, double[:] b, double rtol=*, double atol=*) nogil
cdef double[:] fabs(double[:] xs) nogil
cdef double[:] cabs(double complex[:] xs) nogil
cdef double fabs_width(double[:] xs) nogil
cdef double cabs_width(double complex[:] xs) nogil
cdef double[:] fpermute(double[:] xs, unsigned long[:] inds) nogil
cdef double fsum(double[:] xs) nogil
cdef double fmean(double[:] xs) nogil
cdef double fmin(double[:] xs) nogil
cdef double fmax(double[:] xs) nogil
cdef unsigned long fargmin(double[:] xs) nogil
cdef unsigned long fargmax(double[:] xs) nogil
cdef (unsigned long, unsigned long) fargminmax(double[:] xs) nogil
cdef void fsort(double[::1] xs) nogil
cdef unsigned unsigned long[:] fargsort(double[:] xs, bint reverse=*) nogil
