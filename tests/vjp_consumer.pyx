"""Compiled transpose producer for native ABI, lifetime, and no-GIL checks."""
from libc.stddef cimport size_t
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.vjp cimport make_jacobian_transpose_callback,cyroot_vjp_descriptor

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Context:
    size_t gil_held
    int code
    bint partial

cdef int analytic(const double* x,const double* w,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    if PyGILState_Check():p.gil_held+=1
    out[0]=2*x[0]*w[0]+x[1]*w[1]+2*w[2]
    if not p.partial:out[1]=w[0]+x[0]*w[1]-w[2]
    return p.code

cdef class Owner:
    cdef Context context
    cdef cyroot_vjp_descriptor descriptor
    cdef object __weakref__
    def __init__(self,code=0,partial=False):
        self.context.code=code;self.context.partial=partial
    def callback(self):return make_jacobian_transpose_callback(analytic,2,3,<void*>&self.context,self)
    def capsule(self,abi=1,small=False,null_function=False):
        self.descriptor.struct_size=0 if small else sizeof(cyroot_vjp_descriptor)
        self.descriptor.abi_version=abi
        self.descriptor.input_dimension=2;self.descriptor.output_dimension=3
        self.descriptor.function=analytic
        if null_function:self.descriptor.function=NULL
        self.descriptor.context=<void*>&self.context
        return PyCapsule_New(&self.descriptor,b'cyroot.jacobian_transpose_callback.v1',NULL)
    @property
    def gil_held(self):return self.context.gil_held
