"""External native producer used to verify ABI, lifetimes and actual GIL state."""
from libc.stddef cimport size_t
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.complex_callbacks cimport make_complex_scalar_callback
from cyroot.native.differentiation cimport (
    make_complex_vector_callback, cyroot_complex_vector_descriptor,
    complex_step_jacobian, adaptive_jacobian, EstimateReport, AdaptiveHessian,
)
from cpython.pycapsule cimport PyCapsule_New
from libc.math cimport NAN

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Counters:
    size_t calls
    size_t gil
    Py_ssize_t fail_after
    Py_ssize_t partial_after

cdef int count(Counters* state) noexcept nogil:
    state.calls+=1
    if PyGILState_Check():state.gil+=1
    if state.fail_after>=0 and state.calls><size_t>state.fail_after:return -37
    return 0

cdef int scalar(double x,double* out,void* context) noexcept nogil:
    cdef Counters* state=<Counters*>context
    if count(state):return -37
    if state.partial_after<0 or state.calls<=<size_t>state.partial_after:out[0]=x*x-2.
    return 0

cdef int complex_scalar(double xr,double xi,double* fr,double* fi,void* context) noexcept nogil:
    cdef Counters* state=<Counters*>context
    if count(state):return -37
    fr[0]=xr*xr-xi*xi-2.
    if state.partial_after<0 or state.calls<=<size_t>state.partial_after:fi[0]=2.*xr*xi
    return 0

cdef int vector(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Counters* state=<Counters*>context
    cdef size_t i,j
    cdef double coefficient,value
    if count(state):return -37
    for i in range(m):
        coefficient=1.;value=0.
        for j in range(n):
            value+=coefficient*(x[j]*x[j]-(j+1.)*(j+1.))
            coefficient*=i+1.
        out[i]=value
        if state.partial_after>=0 and state.calls><size_t>state.partial_after:break
    return 0

cdef int mixed(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Counters* state=<Counters*>context
    cdef size_t i,j,k
    cdef double value
    if count(state):return -37
    for i in range(m):
        value=0.
        for j in range(n):
            value+=(i+1.)**j*x[j]*x[j]*x[j]
            for k in range(j+1,n):value+=(i+1.)*(j+1.)*(k+1.)*x[j]*x[k]
        out[i]=value
        if state.partial_after>=0 and state.calls><size_t>state.partial_after:break
    return 0

cdef int square(const double* x,size_t n,double* out,void* context) noexcept nogil:
    return vector(x,n,n,out,context)

cdef int complex_vector(const double* xr,const double* xi,size_t n,size_t m,
                       double* fr,double* fi,void* context) noexcept nogil:
    cdef Counters* state=<Counters*>context
    cdef size_t i,j
    cdef double coefficient,real,imag
    if count(state):return -37
    for i in range(m):
        coefficient=1.;real=imag=0.
        for j in range(n):
            real+=coefficient*(xr[j]*xr[j]-xi[j]*xi[j]-(j+1.)*(j+1.))
            imag+=coefficient*2.*xr[j]*xi[j]
            coefficient*=i+1.
        fr[i],fi[i]=real,imag
        if state.partial_after>=0 and state.calls><size_t>state.partial_after:break
    return 0

cdef class Owner:
    cdef Counters state
    cdef cyroot_complex_vector_descriptor descriptor
    cdef object __weakref__
    def __init__(self,fail_after=-1,partial_after=-1):
        self.state.fail_after=fail_after
        self.state.partial_after=partial_after
    def scalar(self):return make_scalar_callback(scalar,<void*>&self.state,self)
    def complex_scalar(self):return make_complex_scalar_callback(complex_scalar,<void*>&self.state,self)
    def vector(self,n,m):return make_rectangular_callback(vector,n,m,<void*>&self.state,self)
    def mixed(self,n,m):return make_rectangular_callback(mixed,n,m,<void*>&self.state,self)
    def square(self,n):return make_vector_callback(square,n,<void*>&self.state,self)
    def complex_vector(self,n,m):return make_complex_vector_callback(complex_vector,n,m,<void*>&self.state,self)
    def capsule(self,n=2,m=3,version=1,truncated=False):
        self.descriptor.struct_size=1 if truncated else sizeof(cyroot_complex_vector_descriptor)
        self.descriptor.abi_version=version
        self.descriptor.input_dimension=n
        self.descriptor.output_dimension=m
        self.descriptor.function=complex_vector
        self.descriptor.context=<void*>&self.state
        return PyCapsule_New(&self.descriptor,b'cyroot.complex_vector_callback.v1',NULL)
    @property
    def counters(self):return self.state.calls,self.state.gil


def guarded_kernel(bint adaptive=False,fail_after=-1,partial_after=-1):
    """Raw header calls with external guarded output, scratch and input copies."""
    cdef double x[2]
    cdef double output[8]
    cdef double error[8]
    cdef double work[50]
    cdef double steps[2]
    cdef EstimateReport report
    cdef Owner owner=Owner(fail_after,partial_after)
    cdef int status,i
    x[0]=.5;x[1]=1.
    for i in range(8):output[i]=error[i]=1234567.
    for i in range(50):work[i]=7654321.
    steps[0]=steps[1]=.125 if adaptive else 1e-20
    with nogil:
        if adaptive:
            # work length 2+(6+5)*3=35 doubles.
            status=adaptive_jacobian(x,2,3,steps,1e-10,1e-8,6,&output[1],&error[1],
                                     &work[1],vector,<void*>&owner.state,report)
        else:
            # work length 2*2+2*3=10 doubles.
            status=complex_step_jacobian(x,2,3,steps,&output[1],&work[1],
                                         complex_vector,<void*>&owner.state,report)
    return dict(status=status,calls=report.calls,code=report.callback_code,
        output=tuple([output[i] for i in range(1,7)]),point=(x[0],x[1]),
        guards=(output[0],output[7],error[0],error[7],work[0],work[36 if adaptive else 11]),
        counters=owner.counters)


def guarded_hessian(fail_after=-1,partial_after=-1):
    """Exercise the provider callback with guarded caller output and borrowed x."""
    cdef double x[2]
    cdef double output[14]
    cdef int code,i
    cdef Owner owner=Owner(fail_after,partial_after)
    cdef AdaptiveHessian provider=AdaptiveHessian(owner.mixed(2,3),2,3)
    x[0]=.5;x[1]=1.
    for i in range(14):output[i]=1234567.
    provider._enter()
    try:
        with nogil:code=provider.evaluate_raw(x,&output[1])
    finally:provider._leave()
    return dict(code=code,output=tuple([output[i] for i in range(1,13)]),
        guards=(output[0],output[13]),point=(x[0],x[1]),
        estimate=provider.last_estimate,counters=owner.counters)
