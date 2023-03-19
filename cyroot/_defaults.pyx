# distutils: language=c++
from libc cimport math

cdef double PHI = (1 + math.sqrt(5)) / 2
cdef double ETOL = 1e-10
cdef double PTOL = 1e-12
