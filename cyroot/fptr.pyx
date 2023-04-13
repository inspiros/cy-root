# distutils: language=c++

import numpy as np

__all__ = [
    'DoubleScalarFPtr', 'CyDoubleScalarFPtr', 'PyDoubleScalarFPtr',
    'DoubleBiScalarFPtr', 'CyDoubleBiScalarFPtr', 'PyDoubleBiScalarFPtr',
    'ComplexScalarFPtr', 'CyComplexScalarFPtr', 'PyComplexScalarFPtr',
    'ComplexBiScalarFPtr', 'CyComplexBiScalarFPtr', 'PyComplexBiScalarFPtr',
    'DoubleVectorFPtr', 'CyDoubleVectorFPtr', 'PyDoubleVectorFPtr',
    'ComplexVectorFPtr', 'CyComplexVectorFPtr', 'PyComplexVectorFPtr',
    'NdArrayFPtr', 'CyNdArrayFPtr', 'PyNdArrayFPtr',
]

cdef class TrackedFPtr:
    """abstract class"""
    def __cinit__(self):
        self.n_f_calls = 0

# --------------------------------
# Double Scalar
# --------------------------------
cdef class DoubleScalarFPtr(TrackedFPtr):
    def __call__(self, double x):
        return self.eval(x)

    cdef double eval(self, double x) except *:
        raise NotImplementedError

cdef class CyDoubleScalarFPtr(DoubleScalarFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyDoubleScalarFPtr from_f(dsf_ptr f):
        cdef CyDoubleScalarFPtr wrapper = CyDoubleScalarFPtr.__new__(CyDoubleScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef double eval(self, double x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyDoubleScalarFPtr(DoubleScalarFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef DoubleScalarFPtr from_f(object f):
        if isinstance(f, DoubleScalarFPtr):
            return f
        cdef PyDoubleScalarFPtr wrapper = PyDoubleScalarFPtr.__new__(PyDoubleScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef double eval(self, double x) except *:
        self.n_f_calls += 1
        return self.f(x)

# --------------------------------
# Double Bi-Scalar
# --------------------------------
cdef class DoubleBiScalarFPtr(TrackedFPtr):
    def __call__(self, double a, double b):
        return self.eval(a, b)

    cdef (double, double) eval(self, double a, double b) except *:
        raise NotImplementedError

cdef class CyDoubleBiScalarFPtr(DoubleBiScalarFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyDoubleBiScalarFPtr from_f(dbsf_ptr f):
        cdef CyDoubleBiScalarFPtr wrapper = CyDoubleBiScalarFPtr.__new__(CyDoubleBiScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef (double, double) eval(self, double a, double b) except *:
        self.n_f_calls += 1
        return self.f(a, b)

cdef class PyDoubleBiScalarFPtr(DoubleBiScalarFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef DoubleBiScalarFPtr from_f(object f):
        if isinstance(f, DoubleBiScalarFPtr):
            return f
        cdef PyDoubleBiScalarFPtr wrapper = PyDoubleBiScalarFPtr.__new__(PyDoubleBiScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef (double, double) eval(self, double a, double b) except *:
        self.n_f_calls += 1
        return self.f(a, b)

# --------------------------------
# Complex
# --------------------------------
cdef class ComplexScalarFPtr(TrackedFPtr):
    def __call__(self, double complex x):
        return self.eval(x)

    cdef double complex eval(self, double complex x) except *:
        raise NotImplementedError

cdef class CyComplexScalarFPtr(ComplexScalarFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyComplexScalarFPtr from_f(csf_ptr f):
        cdef CyComplexScalarFPtr wrapper = CyComplexScalarFPtr.__new__(CyComplexScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef double complex eval(self, double complex x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyComplexScalarFPtr(ComplexScalarFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef ComplexScalarFPtr from_f(object f):
        if isinstance(f, ComplexScalarFPtr):
            return f
        cdef PyComplexScalarFPtr wrapper = PyComplexScalarFPtr.__new__(PyComplexScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef double complex eval(self, double complex x) except *:
        self.n_f_calls += 1
        return self.f(x)

# --------------------------------
# Complex Bi-Scalar
# --------------------------------
cdef class ComplexBiScalarFPtr(TrackedFPtr):
    def __call__(self, double complex a, double complex b):
        return self.eval(a, b)

    cdef (double complex, double complex) eval(self, double complex a, double complex b) except *:
        raise NotImplementedError

cdef class CyComplexBiScalarFPtr(ComplexBiScalarFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyComplexBiScalarFPtr from_f(cbsf_ptr f):
        cdef CyComplexBiScalarFPtr wrapper = CyComplexBiScalarFPtr.__new__(CyComplexBiScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef (double complex, double complex) eval(self, double complex a, double complex b) except *:
        self.n_f_calls += 1
        return self.f(a, b)

cdef class PyComplexBiScalarFPtr(ComplexBiScalarFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef ComplexBiScalarFPtr from_f(object f):
        if isinstance(f, ComplexBiScalarFPtr):
            return f
        cdef PyComplexBiScalarFPtr wrapper = PyComplexBiScalarFPtr.__new__(PyComplexBiScalarFPtr)
        wrapper.f = f
        return wrapper

    cpdef (double complex, double complex) eval(self, double complex a, double complex b) except *:
        self.n_f_calls += 1
        return self.f(a, b)

# --------------------------------
# Double MemoryView
# --------------------------------
cdef class DoubleVectorFPtr(TrackedFPtr):
    def __call__(self, double[:] x):
        return self.eval(x)

    cdef double[:] eval(self, double[:] x) except *:
        raise NotImplementedError

cdef class CyDoubleVectorFPtr(DoubleVectorFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyDoubleVectorFPtr from_f(dvf_ptr f):
        cdef CyDoubleVectorFPtr wrapper = CyDoubleVectorFPtr.__new__(CyDoubleVectorFPtr)
        wrapper.f = f
        return wrapper

    cpdef double[:] eval(self, double[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyDoubleVectorFPtr(DoubleVectorFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef DoubleVectorFPtr from_f(object f):
        if isinstance(f, DoubleVectorFPtr):
            return f
        cdef PyDoubleVectorFPtr wrapper = PyDoubleVectorFPtr.__new__(PyDoubleVectorFPtr)
        wrapper.f = f
        return wrapper

    cpdef double[:] eval(self, double[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

# --------------------------------
# Complex MemoryView
# --------------------------------
cdef class ComplexVectorFPtr(TrackedFPtr):
    def __call__(self, double complex[:] x):
        return self.eval(x)

    cdef double complex[:] eval(self, double complex[:] x) except *:
        raise NotImplementedError

cdef class CyComplexVectorFPtr(ComplexVectorFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyComplexVectorFPtr from_f(cvf_ptr f):
        cdef CyComplexVectorFPtr wrapper = CyComplexVectorFPtr.__new__(CyComplexVectorFPtr)
        wrapper.f = f
        return wrapper

    cpdef double complex[:] eval(self, double complex[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyComplexVectorFPtr(ComplexVectorFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef ComplexVectorFPtr from_f(object f):
        if isinstance(f, ComplexVectorFPtr):
            return f
        cdef PyComplexVectorFPtr wrapper = PyComplexVectorFPtr.__new__(PyComplexVectorFPtr)
        wrapper.f = f
        return wrapper

    cpdef double complex[:] eval(self, double complex[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

# --------------------------------
# Numpy Array
# --------------------------------
cdef class NdArrayFPtr(TrackedFPtr):
    def __call__(self, np.ndarray x):
        return self.eval(x)

    cdef np.ndarray eval(self, np.ndarray x):
        raise NotImplementedError

cdef class CyNdArrayFPtr(NdArrayFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyNdArrayFPtr from_f(ndarray_f_ptr f):
        cdef CyNdArrayFPtr wrapper = CyNdArrayFPtr.__new__(CyNdArrayFPtr)
        wrapper.f = f
        return wrapper

    cpdef np.ndarray eval(self, np.ndarray x):
        self.n_f_calls += 1
        return self.f(x)

cdef class PyNdArrayFPtr(NdArrayFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef NdArrayFPtr from_f(object f):
        if isinstance(f, NdArrayFPtr):
            return f
        cdef PyNdArrayFPtr wrapper = PyNdArrayFPtr.__new__(PyNdArrayFPtr)
        wrapper.f = f
        return wrapper

    cpdef np.ndarray eval(self, np.ndarray x):
        self.n_f_calls += 1
        return np.asarray(self.f(x), dtype=np.float64)
