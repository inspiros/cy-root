# distutils: language=c++

__all__ = [
    'DoubleScalarFPtr', 'CyDoubleScalarFPtr', 'PyDoubleScalarFPtr',
    'ComplexScalarFPtr', 'CyComplexScalarFPtr', 'PyComplexScalarFPtr',
    'DoubleVectorFPtr', 'CyDoubleVectorFPtr', 'PyDoubleVectorFPtr',
    'ComplexVectorFPtr', 'CyComplexVectorFPtr', 'PyComplexVectorFPtr',
]

cdef class TrackedFPtr:
    """abstract class"""
    def __cinit__(self):
        self.n_f_calls = 0

# --------------------------------
# Double
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

    cdef inline double eval(self, double x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyDoubleScalarFPtr(DoubleScalarFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef PyDoubleScalarFPtr from_f(object f):
        cdef PyDoubleScalarFPtr wrapper = PyDoubleScalarFPtr.__new__(PyDoubleScalarFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double eval(self, double x) except *:
        self.n_f_calls += 1
        return self.f(x)

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

    cdef inline double complex eval(self, double complex x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyComplexScalarFPtr(ComplexScalarFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef PyComplexScalarFPtr from_f(object f):
        cdef PyComplexScalarFPtr wrapper = PyComplexScalarFPtr.__new__(PyComplexScalarFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double complex eval(self, double complex x) except *:
        self.n_f_calls += 1
        return self.f(x)

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

    cdef double[:] eval(self, double[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyDoubleVectorFPtr(DoubleVectorFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef PyDoubleVectorFPtr from_f(object f):
        cdef PyDoubleVectorFPtr wrapper = PyDoubleVectorFPtr.__new__(PyDoubleVectorFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double[:] eval(self, double[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

# --------------------------------
# Double Complex MemoryView
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

    cdef double complex[:] eval(self, double complex[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyComplexVectorFPtr(ComplexVectorFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef PyComplexVectorFPtr from_f(object f):
        cdef PyComplexVectorFPtr wrapper = PyComplexVectorFPtr.__new__(PyComplexVectorFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double complex[:] eval(self, double complex[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)
