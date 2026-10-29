"""Native Newton-Krylov producers and an independent direct GMRES caller."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.jvp cimport make_jacobian_vector_callback
from cyroot.native.vjp cimport make_jacobian_transpose_callback
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


cdef struct RectangularContext:
    size_t f_calls,jvp_calls,vjp_calls,gil_held
    Py_ssize_t fail_f,fail_p,fail_vjp
    bint partial_vjp

cdef int rectangular_residual(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef RectangularContext* p=<RectangularContext*>context
    cdef double u=x[0]*x[0]-2.,v=x[1]*x[1]-3.
    p.f_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=u+(x[2] if n==3 else 0.)
    if p.fail_f>=0 and p.f_calls><size_t>p.fail_f:return 37
    out[1]=v+(2*x[2] if n==3 else 0.)
    if m==3:out[2]=u+2*v
    return 0

cdef int rectangular_product(const double* x,const double* v,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef RectangularContext* p=<RectangularContext*>context
    cdef double a=2*x[0]*v[0],b=2*x[1]*v[1]
    p.jvp_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=a+(v[2] if n==3 else 0.)
    if p.fail_p>=0 and p.jvp_calls><size_t>p.fail_p:return 41
    out[1]=b+(2*v[2] if n==3 else 0.)
    if m==3:out[2]=a+2*b
    return 0

cdef int rectangular_transpose(const double* x,const double* w,size_t n,size_t m,double* out,void* context) noexcept nogil:
    cdef RectangularContext* p=<RectangularContext*>context
    p.vjp_calls+=1
    if PyGILState_Check():p.gil_held+=1
    out[0]=2*x[0]*(w[0]+(w[2] if m==3 else 0.))
    if p.fail_vjp>=0 and p.vjp_calls><size_t>p.fail_vjp:return 43
    if not p.partial_vjp or n==3:
        out[1]=2*x[1]*(w[1]+(2*w[2] if m==3 else 0.))
    if n==3 and not p.partial_vjp:out[2]=w[0]+2*w[1]
    return 0

cdef class RectangularOwner:
    """Tall F=(u,v,u+2v), or wide F=(u+x2,v+2*x2), u=x0²-2,v=x1²-3."""
    cdef RectangularContext context
    cdef size_t _n,_m
    cdef object __weakref__
    def __init__(self,wide=False,fail_f=-1,fail_p=-1,fail_vjp=-1,partial_vjp=False):
        self._n=3 if wide else 2;self._m=2 if wide else 3
        self.context.fail_f=fail_f;self.context.fail_p=fail_p;self.context.fail_vjp=fail_vjp
        self.context.partial_vjp=partial_vjp
    def residual(self):return make_rectangular_callback(rectangular_residual,self._n,self._m,<void*>&self.context,self)
    def product(self):return make_jacobian_vector_callback(rectangular_product,self._n,self._m,<void*>&self.context,self)
    def transpose(self):return make_jacobian_transpose_callback(rectangular_transpose,self._n,self._m,<void*>&self.context,self)
    @property
    def f_calls(self):return self.context.f_calls
    @property
    def jvp_calls(self):return self.context.jvp_calls
    @property
    def vjp_calls(self):return self.context.vjp_calls
    @property
    def gil_held(self):return self.context.gil_held


cdef extern from "lsqr.hpp" namespace "cyroot::native" nogil:
    int native_lsqr(size_t,size_t,size_t,double,const double*,double*,double*,double*,KrylovOperator,KrylovOperator,void*,size_t*,double*,double*) noexcept

cdef struct RectangularDenseContext:
    size_t n,m
    const double* matrix
    bint partial

cdef int rectangular_dense_product(const double* v,double* out,void* context) noexcept nogil:
    cdef RectangularDenseContext* p=<RectangularDenseContext*>context
    cdef size_t i,j
    for i in range(p.m):
        if p.partial and i+1==p.m:continue
        out[i]=0.
        for j in range(p.n):out[i]+=p.matrix[i*p.n+j]*v[j]
    return 0

cdef int rectangular_dense_transpose(const double* w,double* out,void* context) noexcept nogil:
    cdef RectangularDenseContext* p=<RectangularDenseContext*>context
    cdef size_t i,j
    for j in range(p.n):
        if p.partial and j+1==p.n:continue
        out[j]=0.
        for i in range(p.m):out[j]+=p.matrix[i*p.n+j]*w[i]
    return 0


def direct_lsqr(a,b,max_iter,rtol,partial=False):
    """Return status,step,relative residual,relative stationarity,iterations."""
    cdef size_t m=len(b),n,iterations=0,i,j
    cdef size_t budget=max_iter
    cdef int status
    cdef double relative,stationarity,tolerance=rtol
    cdef vector[double] matrix,rhs,solution,applied,work
    cdef RectangularDenseContext context
    if m==0 or len(a)!=m:raise ValueError('Matrix rows must match the nonempty RHS.')
    n=len(a[0])
    if n==0 or budget==0 or not 0.<tolerance<1.:raise ValueError('Invalid direct rectangular test solve configuration.')
    matrix.resize(m*n);rhs.resize(m);solution.resize(n);applied.resize(m)
    work.resize(3*m+4*n)
    for i in range(m):
        rhs[i]=b[i]
        if len(a[i])!=n:raise ValueError('Matrix rows must have equal lengths.')
        for j in range(n):matrix[i*n+j]=a[i][j]
    context.n=n;context.m=m;context.matrix=&matrix[0];context.partial=partial
    with nogil:
        status=native_lsqr(n,m,budget,tolerance,&rhs[0],&solution[0],&applied[0],&work[0],
            rectangular_dense_product,rectangular_dense_transpose,<void*>&context,
            &iterations,&relative,&stationarity)
    return status,tuple([solution[i] for i in range(n)]),relative,stationarity,iterations
