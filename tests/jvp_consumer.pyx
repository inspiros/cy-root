"""Compiled producer and direct caller of allocation-free JVP plans."""
from libc.stddef cimport size_t
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.jvp cimport make_jacobian_vector_callback,JacobianVectorPlan,cyroot_jvp_descriptor
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.vector cimport make_vector_callback

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Context:
    size_t f_calls,gil_held
    Py_ssize_t fail_after
    int product_code

cdef int residual(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    p.f_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=x[0]*x[0]+x[1]
    if p.fail_after>=0 and p.f_calls><size_t>p.fail_after:return 37
    out[1]=x[0]*x[1]
    if m==3:out[2]=2*x[0]-x[1]
    return 0

cdef int square_residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    return residual(x,n,n,out,context)

cdef int analytic(const double* x,const double* v,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    if PyGILState_Check():p.gil_held+=1
    out[0]=2*x[0]*v[0]+v[1]
    if p.product_code!=0:return p.product_code
    out[1]=x[1]*v[0]+x[0]*v[1]
    out[2]=2*v[0]-v[1]
    return 0

cdef class Owner:
    cdef Context context
    cdef cyroot_jvp_descriptor descriptor
    cdef object __weakref__
    def __init__(self,fail_after=-1,product_code=0):
        self.context.fail_after=fail_after;self.context.product_code=product_code
    def residual(self,square=False):
        if square:return make_vector_callback(square_residual,2,<void*>&self.context,self)
        return make_rectangular_callback(residual,2,3,<void*>&self.context,self)
    def callback(self):return make_jacobian_vector_callback(analytic,2,3,<void*>&self.context,self)
    def capsule(self,abi=1):
        self.descriptor.struct_size=sizeof(cyroot_jvp_descriptor);self.descriptor.abi_version=abi
        self.descriptor.input_dimension=2;self.descriptor.output_dimension=3
        self.descriptor.function=analytic;self.descriptor.context=<void*>&self.context
        return PyCapsule_New(&self.descriptor,b'cyroot.jacobian_vector_callback.v1',NULL)
    def raw_plan(self,JacobianVectorPlan plan):
        cdef double x[2]
        cdef double v[2]
        cdef double out[3]
        cdef double work[13]
        cdef size_t calls=0
        cdef int status,code=0
        if not plan._initialized or plan._n!=2 or plan._m!=3:raise ValueError('Expected initialized 2-to-3 plan.')
        x[0]=1.2;x[1]=.7;v[0]=.5;v[1]=-2.
        with nogil:status=plan.compute(x,v,NULL,out,work,residual,<void*>&self.context,&calls,&code)
        return status,(out[0],out[1],out[2]),calls,code
    @property
    def f_calls(self):return self.context.f_calls
    @property
    def gil_held(self):return self.context.gil_held
