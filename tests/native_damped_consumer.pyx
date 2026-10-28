"""Compiled consumer verifying the damped states and callback ABI under nogil."""
from libc.stddef cimport size_t
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.damped_newton cimport DampedNewton, VectorDampedNewton

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Context:
    size_t f_calls, j_calls, gil_held
    Py_ssize_t fail_after

cdef int scalar_f(double x,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    p.f_calls+=1
    if PyGILState_Check():p.gil_held+=1
    if p.fail_after>=0 and p.f_calls><size_t>p.fail_after:return 37
    out[0]=x*x-2.
    return 0

cdef int scalar_j(double x,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    p.j_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=2*x
    return 0

cdef int vector_f(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef int status=scalar_f(x[0],out,context)
    if status!=0:return status
    out[1]=2*out[0]
    return 0

cdef int vector_j(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    scalar_j(x[0],out,context)
    out[1]=2*out[0]
    return 0

cdef class Owner:
    cdef Context context
    cdef object __weakref__
    def __init__(self,fail_after=-1):self.context.fail_after=fail_after
    def callbacks(self,vector=False):
        if vector:
            return (make_rectangular_callback(vector_f,1,2,<void*>&self.context,self),
                    make_rectangular_callback(vector_j,1,2,<void*>&self.context,self,jacobian=True))
        return (make_scalar_callback(scalar_f,<void*>&self.context,self),
                make_scalar_callback(scalar_j,<void*>&self.context,self))
    @property
    def f_calls(self):return self.context.f_calls
    @property
    def j_calls(self):return self.context.j_calls
    @property
    def gil_held(self):return self.context.gil_held
