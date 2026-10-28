"""Native scalar/interval producers for the installed Hybisect contract."""
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.hybisect cimport make_interval_callback
from libc.math cimport fmin, fmax
cdef extern from 'Python.h' nogil:
    int PyGILState_Check()

cdef struct Context:
    int calls
    int interval_calls
    int gil_held
    int failure

cdef class Owner:
    cdef Context state
    cdef object __weakref__

    @property
    def calls(self):return self.state.calls
    @property
    def interval_calls(self):return self.state.interval_calls
    @property
    def gil_held(self):return self.state.gil_held
    @property
    def failure(self):return self.state.failure
    @failure.setter
    def failure(self,value):self.state.failure=value

    def residual(self):
        return make_scalar_callback(residual,<void*>&self.state,self)

    def enclosure(self):
        return make_interval_callback(enclosure,<void*>&self.state,self)

cdef int residual(double x,double* out,void* context) noexcept nogil:
    cdef Context* owner = <Context*>context
    owner.calls += 1
    owner.gil_held += PyGILState_Check()
    if owner.failure == 1:return -17
    out[0] = x*x-1.
    return 0

cdef int enclosure(double a,double b,double* lo,double* hi,void* context) noexcept nogil:
    cdef Context* owner = <Context*>context
    owner.interval_calls += 1
    owner.gil_held += PyGILState_Check()
    if owner.failure == 2:return 23
    lo[0] = -1. if a<=0<=b else fmin(a*a,b*b)-1.
    hi[0] = fmax(a*a,b*b)-1.
    return 0
