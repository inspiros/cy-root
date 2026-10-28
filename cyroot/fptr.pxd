# distutils: language=c++

cimport numpy as np

cdef class TrackedFPtr:
    cdef public unsigned long n_f_calls

# --------------------------------
# Double Scalar
# --------------------------------
ctypedef double (*dsf_ptr)(double) nogil

cdef class DoubleScalarFPtr(TrackedFPtr):
    cdef double eval(self, double x) except * nogil
    cdef double eval2(self, double x1, double x2) except * nogil

cdef class CyDoubleScalarFPtr(DoubleScalarFPtr):
    cdef dsf_ptr f
    @staticmethod
    cdef CyDoubleScalarFPtr from_f(dsf_ptr f)
    cdef double eval(self, double x) except * nogil

cdef class PyDoubleScalarFPtr(DoubleScalarFPtr):
    cdef object f
    @staticmethod
    cdef DoubleScalarFPtr from_f(object f)
    cdef double eval(self, double x) except * nogil

ctypedef fused double_scalar_func_type:
    dsf_ptr
    DoubleScalarFPtr

# --------------------------------
# Double Bi-Scalar
# --------------------------------
ctypedef (double, double) (*dbsf_ptr)(double, double) nogil

cdef class DoubleBiScalarFPtr(TrackedFPtr):
    cdef (double, double) eval(self, double a, double b) except * nogil

cdef class CyDoubleBiScalarFPtr(DoubleBiScalarFPtr):
    cdef dbsf_ptr f
    @staticmethod
    cdef CyDoubleBiScalarFPtr from_f(dbsf_ptr f)
    cdef (double, double) eval(self, double a, double b) except * nogil

cdef class PyDoubleBiScalarFPtr(DoubleBiScalarFPtr):
    cdef object f
    @staticmethod
    cdef DoubleBiScalarFPtr from_f(object f)
    cdef (double, double) eval(self, double a, double b) except * nogil

ctypedef fused double_bi_scalar_func_type:
    dbsf_ptr
    DoubleBiScalarFPtr

# --------------------------------
# Complex
# --------------------------------
ctypedef double complex (*csf_ptr)(double complex) nogil

cdef class ComplexScalarFPtr(TrackedFPtr):
    cdef double complex eval(self, double complex x) except * nogil

cdef class CyComplexScalarFPtr(ComplexScalarFPtr):
    cdef csf_ptr f
    @staticmethod
    cdef CyComplexScalarFPtr from_f(csf_ptr f)
    cdef double complex eval(self, double complex x) except * nogil

cdef class PyComplexScalarFPtr(ComplexScalarFPtr):
    cdef object f
    @staticmethod
    cdef ComplexScalarFPtr from_f(object f)
    cdef double complex eval(self, double complex x) except * nogil

ctypedef fused complex_scalar_func_type:
    csf_ptr
    ComplexScalarFPtr

# --------------------------------
# Complex Bi-Scalar
# --------------------------------
ctypedef (double complex, double complex) (*cbsf_ptr)(double complex, double complex) nogil

cdef class ComplexBiScalarFPtr(TrackedFPtr):
    cdef (double complex, double complex) eval(self, double complex a, double complex b) except * nogil

cdef class CyComplexBiScalarFPtr(ComplexBiScalarFPtr):
    cdef cbsf_ptr f
    @staticmethod
    cdef CyComplexBiScalarFPtr from_f(cbsf_ptr f)
    cdef (double complex, double complex) eval(self, double complex a, double complex b) except * nogil

cdef class PyComplexBiScalarFPtr(ComplexBiScalarFPtr):
    cdef object f
    @staticmethod
    cdef ComplexBiScalarFPtr from_f(object f)
    cdef (double complex, double complex) eval(self, double complex a, double complex b) except * nogil

ctypedef fused complex_bi_scalar_func_type:
    cbsf_ptr
    ComplexBiScalarFPtr

# --------------------------------
# Double MemoryView
# --------------------------------
ctypedef double[:] (*dvf_ptr)(double[:])

cdef class DoubleVectorFPtr(TrackedFPtr):
    cdef double[:] eval(self, double[:] x) except *
    cdef double[:] eval2(self, double[:] x1, double[:] x2) except *

cdef class CyDoubleVectorFPtr(DoubleVectorFPtr):
    cdef dvf_ptr f
    @staticmethod
    cdef CyDoubleVectorFPtr from_f(dvf_ptr f)
    cdef double[:] eval(self, double[:] x) except *

cdef class PyDoubleVectorFPtr(DoubleVectorFPtr):
    cdef object f
    @staticmethod
    cdef DoubleVectorFPtr from_f(object f)
    cdef double[:] eval(self, double[:] x) except *

ctypedef fused double_vector_func_type:
    dvf_ptr
    DoubleVectorFPtr

# out
ctypedef void (*dvf_out_ptr)(double[:], double[:]) nogil

cdef class DoubleVectorOutFPtr(TrackedFPtr):
    cdef void eval(self, double[:] x, double[:] out) except * nogil

cdef class CyDoubleVectorOutFPtr(DoubleVectorOutFPtr):
    cdef dvf_out_ptr f
    @staticmethod
    cdef CyDoubleVectorOutFPtr from_f(dvf_out_ptr f)
    cdef void eval(self, double[:] x, double[:] out) except * nogil

cdef class PyDoubleVectorOutFPtr(DoubleVectorOutFPtr):
    cdef object f
    @staticmethod
    cdef DoubleVectorOutFPtr from_f(object f)
    cdef void eval(self, double[:] x, double[:] out) except * nogil

ctypedef fused double_vector_out_func_type:
    dvf_out_ptr
    DoubleVectorOutFPtr

# reduction
ctypedef double (*dvrf_ptr)(double[:]) except * nogil

cdef class DoubleVectorReductionFPtr(TrackedFPtr):
    cdef double eval(self, double[:] x) except * nogil

cdef class CyDoubleVectorReductionFPtr(DoubleVectorReductionFPtr):
    cdef dvrf_ptr f
    @staticmethod
    cdef CyDoubleVectorReductionFPtr from_f(dvrf_ptr f)
    cdef double eval(self, double[:] x) except * nogil

cdef class PyDoubleVectorReductionFPtr(DoubleVectorReductionFPtr):
    cdef object f
    @staticmethod
    cdef DoubleVectorReductionFPtr from_f(object f)
    cdef double eval(self, double[:] x) except * nogil

ctypedef fused double_vector_reduction_func_type:
    dvrf_ptr
    DoubleVectorReductionFPtr

# --------------------------------
# Complex MemoryView
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
    cdef ComplexVectorFPtr from_f(object f)
    cdef double complex[:] eval(self, double complex[:] x) except *

ctypedef fused complex_vector_func_type:
    cvf_ptr
    ComplexVectorFPtr

# --------------------------------
# Numpy Array
# --------------------------------
ctypedef np.ndarray (*ndarray_f_ptr)(np.ndarray)

cdef class NdArrayFPtr(TrackedFPtr):
    cdef np.ndarray eval(self, np.ndarray x)
    cdef np.ndarray eval2(self, np.ndarray x1, np.ndarray x2)

cdef class CyNdArrayFPtr(NdArrayFPtr):
    cdef ndarray_f_ptr f
    @staticmethod
    cdef CyNdArrayFPtr from_f(ndarray_f_ptr f)
    cpdef np.ndarray eval(self, np.ndarray x)

cdef class PyNdArrayFPtr(NdArrayFPtr):
    cdef object f
    @staticmethod
    cdef NdArrayFPtr from_f(object f)
    cpdef np.ndarray eval(self, np.ndarray x)

ctypedef fused ndarray_func_type:
    ndarray_f_ptr
    NdArrayFPtr
