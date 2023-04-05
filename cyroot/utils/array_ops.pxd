cdef double[:] fabs(double[:] xs) nogil
cdef double[:] cabs(double complex[:] xs) nogil
cdef double fabs_width(double[:] xs) nogil
cdef double cabs_width(double complex[:] xs) nogil
cdef bint allclose(double[:] a, double[:] b, double rtol=*, double atol=*) nogil
cdef double[:] permute(double[:] xs, unsigned long[:] inds) nogil
cdef unsigned long argmin(double[:] xs) nogil
cdef unsigned long argmax(double[:] xs) nogil
cdef (unsigned long, unsigned long) argminmax(double[:] xs) nogil
cdef void sort(double[::1] xs) nogil
cdef unsigned unsigned long[:] argsort(double[:] xs, bint reverse=*) nogil
