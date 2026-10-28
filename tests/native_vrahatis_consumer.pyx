"""External square/rectangular callbacks with native counters and lifetimes."""
from libc.stddef cimport size_t
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.rectangular cimport make_rectangular_callback
cdef extern from 'Python.h' nogil:
    int PyGILState_Check()

cdef struct Context:
    size_t calls
    size_t gil_held
    int fail_after
    int code
    int partial_after
    double target

cdef int square(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Context* state = <Context*>context
    state.calls += 1
    state.gil_held += PyGILState_Check()
    if n != 2:return 91
    if state.fail_after>=0 and state.calls><size_t>state.fail_after:return state.code
    if state.partial_after>=0 and state.calls><size_t>state.partial_after:return 0
    out[0],out[1] = x[0]-state.target,x[1]-state.target
    return 0

cdef int rectangular(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Context* state = <Context*>context
    cdef size_t j
    state.calls += 1
    state.gil_held += PyGILState_Check()
    if n < m:return 92
    if state.fail_after>=0 and state.calls><size_t>state.fail_after:return state.code
    if state.partial_after>=0 and state.calls><size_t>state.partial_after:return 0
    for j in range(m):out[j] = x[j]-state.target
    return 0

cdef class Owner:
    cdef Context state
    cdef object __weakref__
    def __init__(self,target=.7,fail_after=-1,code=-17,partial_after=-1):
        self.state.target=target
        self.state.fail_after=fail_after
        self.state.code=code
        self.state.partial_after=partial_after
    def callback(self):return make_vector_callback(square,2,<void*>&self.state,self)
    def rectangular(self,n=3,m=2):
        return make_rectangular_callback(rectangular,n,m,<void*>&self.state,self)
    @property
    def calls(self):return self.state.calls
    @property
    def gil_held(self):return self.state.gil_held
