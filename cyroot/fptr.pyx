# distutils: language=c++

__all__ = [
    'DoubleScalarFPtr', 'CyDoubleScalarFPtr', 'PyDoubleScalarFPtr',
    'ComplexScalarFPtr', 'CyComplexScalarFPtr', 'PyComplexScalarFPtr',
    'DoubleMemoryViewFPtr', 'CyDoubleMemoryViewFPtr', 'PyDoubleMemoryViewFPtr',
    'ComplexMemoryViewFPtr', 'CyComplexMemoryViewFPtr', 'PyComplexMemoryViewFPtr',
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
    cdef CyDoubleScalarFPtr from_f(f_ptr f):
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
    cdef CyComplexScalarFPtr from_f(cf_ptr f):
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
cdef class DoubleMemoryViewFPtr(TrackedFPtr):
    def __call__(self, double[:] x):
        return self.eval(x)

    cdef double[:] eval(self, double[:] x) except *:
        raise NotImplementedError

cdef class CyDoubleMemoryViewFPtr(DoubleMemoryViewFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyDoubleMemoryViewFPtr from_f(mvf_ptr f):
        cdef CyDoubleMemoryViewFPtr wrapper = CyDoubleMemoryViewFPtr.__new__(CyDoubleMemoryViewFPtr)
        wrapper.f = f
        return wrapper

    cdef double[:] eval(self, double[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyDoubleMemoryViewFPtr(DoubleMemoryViewFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef PyDoubleMemoryViewFPtr from_f(object f):
        cdef PyDoubleMemoryViewFPtr wrapper = PyDoubleMemoryViewFPtr.__new__(PyDoubleMemoryViewFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double[:] eval(self, double[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

# --------------------------------
# Double Complex MemoryView
# --------------------------------
cdef class ComplexMemoryViewFPtr(TrackedFPtr):
    def __call__(self, double complex[:] x):
        return self.eval(x)

    cdef double complex[:] eval(self, double complex[:] x) except *:
        raise NotImplementedError

cdef class CyComplexMemoryViewFPtr(ComplexMemoryViewFPtr):
    def __init__(self):
        raise TypeError('This class cannot be instantiated directly.')

    @staticmethod
    cdef CyComplexMemoryViewFPtr from_f(cmvf_ptr f):
        cdef CyComplexMemoryViewFPtr wrapper = CyComplexMemoryViewFPtr.__new__(CyComplexMemoryViewFPtr)
        wrapper.f = f
        return wrapper

    cdef double complex[:] eval(self, double complex[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)

cdef class PyComplexMemoryViewFPtr(ComplexMemoryViewFPtr):
    def __init__(self, f):
        self.f = f

    @staticmethod
    cdef PyComplexMemoryViewFPtr from_f(object f):
        cdef PyComplexMemoryViewFPtr wrapper = PyComplexMemoryViewFPtr.__new__(PyComplexMemoryViewFPtr)
        wrapper.f = f
        return wrapper

    cdef inline double complex[:] eval(self, double complex[:] x) except *:
        self.n_f_calls += 1
        return self.f(x)
