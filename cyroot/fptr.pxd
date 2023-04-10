# distutils: language=c++

import numpy as np
cimport numpy as np

cdef class TrackedFPtr:
    cdef public unsigned long n_f_calls

# --------------------------------
# Double Scalar
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
# Double Bi-Scalar
# --------------------------------
ctypedef (double, double) (*dbsf_ptr)(double, double)

cdef class DoubleBiScalarFPtr(TrackedFPtr):
    cdef (double, double) eval(self, double a, double b) except *

cdef class CyDoubleBiScalarFPtr(DoubleBiScalarFPtr):
    cdef dbsf_ptr f
    @staticmethod
    cdef CyDoubleBiScalarFPtr from_f(dbsf_ptr f)
    cdef (double, double) eval(self, double a, double b) except *

cdef class PyDoubleBiScalarFPtr(DoubleBiScalarFPtr):
    cdef object f
    @staticmethod
    cdef PyDoubleBiScalarFPtr from_f(object f)
    cdef (double, double) eval(self, double a, double b) except *

ctypedef fused double_bi_scalar_func_type:
    dbsf_ptr
    DoubleBiScalarFPtr

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

# --------------------------------
# Double Numpy Array
# --------------------------------
ctypedef np.ndarray (*ndarray_f_ptr)(np.ndarray)

cdef class NdArrayFPtr(TrackedFPtr):
    cdef np.ndarray eval(self, np.ndarray x)

cdef class CyNdArrayFPtr(NdArrayFPtr):
    cdef ndarray_f_ptr f
    @staticmethod
    cdef CyNdArrayFPtr from_f(ndarray_f_ptr f)
    cdef np.ndarray eval(self, np.ndarray x)

cdef class PyNdArrayFPtr(NdArrayFPtr):
    cdef object f
    @staticmethod
    cdef PyNdArrayFPtr from_f(object f)
    cdef np.ndarray eval(self, np.ndarray x)

ctypedef fused ndarray_func_type:
    ndarray_f_ptr
    NdArrayFPtr
