"""Native Newton-Krylov producers and an independent direct GMRES caller."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.jvp cimport make_jacobian_vector_callback
from cyroot.native.newton_krylov cimport NewtonKrylov

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil
cdef extern from "gmres.hpp" namespace "cyroot::native" nogil:
    ctypedef int (*KrylovOperator)(const double*,double*,void*) noexcept
    int restarted_gmres(size_t,size_t,size_t,double,const double*,double*,double*,double*,KrylovOperator,void*,size_t*,double*) noexcept

cdef struct Context:
    size_t f_calls,jvp_calls,gil_held
    Py_ssize_t fail_f,fail_p

cdef int residual(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    p.f_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=x[0]*x[0]-2.
    if p.fail_f>=0 and p.f_calls><size_t>p.fail_f:return 37
    out[1]=x[1]*x[1]-3.
    return 0
cdef int square_f(const double* x,size_t n,double* out,void* context) noexcept nogil:
    return residual(x,n,n,out,context)
cdef int product(const double* x,const double* v,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef Context* p=<Context*>context
    p.jvp_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=2*x[0]*v[0]
    if p.fail_p>=0 and p.jvp_calls><size_t>p.fail_p:return 41
    out[1]=2*x[1]*v[1]
    return 0
cdef class Owner:
    cdef Context context
    cdef object __weakref__
    def __init__(self,fail_f=-1,fail_p=-1):self.context.fail_f=fail_f;self.context.fail_p=fail_p
    def residual(self,square=False):
        if square:return make_vector_callback(square_f,2,<void*>&self.context,self)
        return make_rectangular_callback(residual,2,2,<void*>&self.context,self)
    def product(self):return make_jacobian_vector_callback(product,2,2,<void*>&self.context,self)
    @property
    def f_calls(self):return self.context.f_calls
    @property
    def jvp_calls(self):return self.context.jvp_calls
    @property
    def gil_held(self):return self.context.gil_held

cdef struct DenseContext:
    size_t n
    const double* matrix
    bint partial
cdef int dense_product(const double* v,double* out,void* context) noexcept nogil:
    cdef DenseContext* p=<DenseContext*>context
    cdef size_t i,j
    for i in range(p.n):
        if p.partial and i+1==p.n:continue
        out[i]=0.
        for j in range(p.n):out[i]+=p.matrix[i*p.n+j]*v[j]
    return 0

def linear_solve(a,b,restart,max_iter,rtol,partial=False):
    cdef size_t n=len(b),k=min(n,restart),iterations=0,i,j
    cdef int status
    cdef double relative,tolerance=rtol
    cdef size_t budget=max_iter
    cdef vector[double] matrix,rhs,solution,applied,work
    cdef DenseContext context
    if n==0 or k==0 or budget==0 or not 0.<tolerance<1.:raise ValueError('Invalid direct test solve configuration.')
    matrix.resize(n*n);rhs.resize(n);solution.resize(n);applied.resize(n)
    work.resize((k+1)*n+(k+1)*k+4*k+1+2*n)
    for i in range(n):
        rhs[i]=b[i]
        for j in range(n):matrix[i*n+j]=a[i][j]
    context.n=n;context.matrix=&matrix[0];context.partial=partial
    with nogil:status=restarted_gmres(n,k,budget,tolerance,&rhs[0],&solution[0],&applied[0],&work[0],dense_product,<void*>&context,&iterations,&relative)
    return status,tuple([solution[i] for i in range(n)]),relative,iterations
