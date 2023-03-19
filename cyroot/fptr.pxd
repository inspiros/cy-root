# distutils: language=c++

cdef class TrackedFPtr:
    cdef public unsigned long n_f_calls

# --------------------------------
# Double
# --------------------------------
ctypedef double (*f_ptr)(double)

cdef class DoubleScalarFPtr(TrackedFPtr):
    cdef double eval(self, double x) except *

cdef class CyDoubleScalarFPtr(DoubleScalarFPtr):
    cdef f_ptr f
    @staticmethod
    cdef CyDoubleScalarFPtr from_f(f_ptr f)
    cdef double eval(self, double x) except *

cdef class PyDoubleScalarFPtr(DoubleScalarFPtr):
    cdef object f
    @staticmethod
    cdef PyDoubleScalarFPtr from_f(object f)
    cdef double eval(self, double x) except *

ctypedef fused func_type:
    f_ptr
    DoubleScalarFPtr

# --------------------------------
# Complex
# --------------------------------
ctypedef double complex (*cf_ptr)(double complex)

cdef class ComplexScalarFPtr(TrackedFPtr):
    cdef double complex eval(self, double complex x) except *

cdef class CyComplexScalarFPtr(ComplexScalarFPtr):
    cdef cf_ptr f
    @staticmethod
    cdef CyComplexScalarFPtr from_f(cf_ptr f)
    cdef double complex eval(self, double complex x) except *

cdef class PyComplexScalarFPtr(ComplexScalarFPtr):
    cdef object f
    @staticmethod
    cdef PyComplexScalarFPtr from_f(object f)
    cdef double complex eval(self, double complex x) except *

ctypedef fused complex_func_type:
    cf_ptr
    ComplexScalarFPtr

# --------------------------------
# Double MemoryView
# --------------------------------
ctypedef double[:] (*mvf_ptr)(double[:])

cdef class DoubleMemoryViewFPtr(TrackedFPtr):
    cdef double[:] eval(self, double[:] x) except *

cdef class CyDoubleMemoryViewFPtr(DoubleMemoryViewFPtr):
    cdef mvf_ptr f
    @staticmethod
    cdef CyDoubleMemoryViewFPtr from_f(mvf_ptr f)
    cdef double[:] eval(self, double[:] x) except *

cdef class PyDoubleMemoryViewFPtr(DoubleMemoryViewFPtr):
    cdef object f
    @staticmethod
    cdef PyDoubleMemoryViewFPtr from_f(object f)
    cdef double[:] eval(self, double[:] x) except *

ctypedef fused mv_func_type:
    mvf_ptr
    DoubleMemoryViewFPtr

# --------------------------------
# Double Complex MemoryView
# --------------------------------
ctypedef double complex[:] (*cmvf_ptr)(double complex[:])

cdef class ComplexMemoryViewFPtr(TrackedFPtr):
    cdef double complex[:] eval(self, double complex[:] x) except *

cdef class CyComplexMemoryViewFPtr(ComplexMemoryViewFPtr):
    cdef cmvf_ptr f
    @staticmethod
    cdef CyComplexMemoryViewFPtr from_f(cmvf_ptr f)
    cdef double complex[:] eval(self, double complex[:] x) except *

cdef class PyComplexMemoryViewFPtr(ComplexMemoryViewFPtr):
    cdef object f
    @staticmethod
    cdef PyComplexMemoryViewFPtr from_f(object f)
    cdef double complex[:] eval(self, double complex[:] x) except *

ctypedef fused complex_mv_func_type:
    cmvf_ptr
    ComplexMemoryViewFPtr
