# distutils: language=c++

cdef class TrackedFPtr:
    cdef public unsigned long n_f_calls

# --------------------------------
# Double
# --------------------------------
ctypedef double (*dsf_ptr)(double)

cdef class DoubleScalarFPtr(TrackedFPtr):
    cdef double eval(self, double x) except *

cdef class CyDoubleScalarFPtr(DoubleScalarFPtr):
    cdef dsf_ptr f
    @staticmethod
    cdef CyDoubleScalarFPtr from_f(dsf_ptr f)
    cdef double eval(self, double x) except *

cdef class PyDoubleScalarFPtr(DoubleScalarFPtr):
    cdef object f
    @staticmethod
    cdef PyDoubleScalarFPtr from_f(object f)
    cdef double eval(self, double x) except *

ctypedef fused double_scalar_func_type:
    dsf_ptr
    DoubleScalarFPtr

# --------------------------------
# Complex
# --------------------------------
ctypedef double complex (*csf_ptr)(double complex)

cdef class ComplexScalarFPtr(TrackedFPtr):
    cdef double complex eval(self, double complex x) except *

cdef class CyComplexScalarFPtr(ComplexScalarFPtr):
    cdef csf_ptr f
    @staticmethod
    cdef CyComplexScalarFPtr from_f(csf_ptr f)
    cdef double complex eval(self, double complex x) except *

cdef class PyComplexScalarFPtr(ComplexScalarFPtr):
    cdef object f
    @staticmethod
    cdef PyComplexScalarFPtr from_f(object f)
    cdef double complex eval(self, double complex x) except *

ctypedef fused complex_scalar_func_type:
    csf_ptr
    ComplexScalarFPtr

# --------------------------------
# Double MemoryView
# --------------------------------
ctypedef double[:] (*dvf_ptr)(double[:])

cdef class DoubleVectorFPtr(TrackedFPtr):
    cdef double[:] eval(self, double[:] x) except *

cdef class CyDoubleVectorFPtr(DoubleVectorFPtr):
    cdef dvf_ptr f
    @staticmethod
    cdef CyDoubleVectorFPtr from_f(dvf_ptr f)
    cdef double[:] eval(self, double[:] x) except *

cdef class PyDoubleVectorFPtr(DoubleVectorFPtr):
    cdef object f
    @staticmethod
    cdef PyDoubleVectorFPtr from_f(object f)
    cdef double[:] eval(self, double[:] x) except *

ctypedef fused double_vector_func_type:
    dvf_ptr
    DoubleVectorFPtr

# --------------------------------
# Double Complex MemoryView
# --------------------------------
ctypedef double complex[:] (*cvf_ptr)(double complex[:])

cdef class ComplexVectorFPtr(TrackedFPtr):
    cdef double complex[:] eval(self, double complex[:] x) except *

cdef class CyComplexVectorFPtr(ComplexVectorFPtr):
    cdef cvf_ptr f
    @staticmethod
    cdef CyComplexVectorFPtr from_f(cvf_ptr f)
    cdef double complex[:] eval(self, double complex[:] x) except *

cdef class PyComplexVectorFPtr(ComplexVectorFPtr):
    cdef object f
    @staticmethod
    cdef PyComplexVectorFPtr from_f(object f)
    cdef double complex[:] eval(self, double complex[:] x) except *

ctypedef fused complex_vector_func_type:
    cvf_ptr
    ComplexVectorFPtr
