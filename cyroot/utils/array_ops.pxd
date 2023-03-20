cdef double[:] fabs(double[:] xs) nogil
cdef double[:] cabs(double complex[:] xs) nogil
cdef double fabs_width(double[:] xs) nogil
cdef double cabs_width(double complex[:] xs) nogil
cdef double[:] permute(double[:] xs, long[:] inds) nogil
cdef long argmin(double[:] xs) nogil
cdef long argmax(double[:] xs) nogil
cdef (long, long) argminmax(double[:] xs) nogil
cdef void sort(double[::1] xs) nogil
cdef long[:] argsort(double[:] xs, bint reverse=*) nogil
