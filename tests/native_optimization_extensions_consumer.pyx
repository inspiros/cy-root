"""Small external callback fixture for BFGS's compiled/no-GIL boundary."""
from cyroot.native.bfgs_math cimport bfgs_update
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.bfgs cimport BFGS,LBFGS
from cyroot.native.levenberg_marquardt cimport LevenbergMarquardt
from cyroot.native.lbfgs_math cimport lbfgs_store,lbfgs_direction
from cyroot.native.dogleg_math cimport dogleg_step

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct Parameters:
    double* matrix
    double* target
    size_t n
    size_t m
    size_t calls[2]
    size_t gil_held
    int fail_stage

cdef int residual(const double* x, size_t n, size_t m, double* out,
                  void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t i,j
    p.calls[0] += 1
    if PyGILState_Check():p.gil_held += 1
    if p.fail_stage == 1:return -41
    if n != p.n or m != p.m:return -99
    for i in range(m):
        out[i] = -p.target[i]
        for j in range(n):out[i] += p.matrix[i+m*j]*x[j]
    return 0

cdef int jacobian(const double* x, size_t n, size_t m, double* out,
                  void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t i
    p.calls[1] += 1
    if PyGILState_Check():p.gil_held += 1
    if p.fail_stage == 2:return -42
    if n != p.n or m != p.m:return -99
    for i in range(n*m):out[i] = p.matrix[i]
    return 0

cdef class Owner:
    cdef Parameters params
    cdef vector[double] matrix,target
    cdef object __weakref__
    def __init__(self,matrix,target):
        cdef size_t i,j
        self.params.m = len(matrix)
        self.params.n = len(matrix[0])
        self.matrix.resize(self.params.n*self.params.m)
        self.target.resize(self.params.m)
        for i in range(self.params.m):
            self.target[i] = target[i]
            for j in range(self.params.n):
                self.matrix[i+self.params.m*j] = matrix[i][j]
        self.params.matrix,self.params.target = self.matrix.data(),self.target.data()
    def callbacks(self):
        return (make_rectangular_callback(residual,self.params.n,self.params.m,
                                         <void*>&self.params,self),
                make_rectangular_callback(jacobian,self.params.n,self.params.m,
                                          <void*>&self.params,self,True))
    @property
    def counters(self):return (self.params.calls[0],self.params.calls[1],self.params.gil_held)
    @property
    def fail_stage(self):return self.params.fail_stage
    @fail_stage.setter
    def fail_stage(self,value):self.params.fail_stage = value



def run_raw(state):
    cdef BFGS quasi
    cdef LevenbergMarquardt trust
    if isinstance(state,BFGS):
        quasi=state;quasi._enter()
        try:
            with nogil:quasi._run()
            quasi._raise_callback_error()
            return quasi._snapshot(quasi._state,quasi._x,quasi._f)
        finally:quasi._release()
    else:
        trust=state;trust._enter()
        try:
            with nogil:trust._run()
            trust._raise_callback_error()
            return trust._snapshot(trust._state,trust._x,trust._f)
        finally:trust._release()


def pair_state(LBFGS state):
    state._enter()
    try:return state._pair_count,state._corrections,state._h==NULL,state._candidate==NULL
    finally:state._release()


def ring_direction(pairs,gradient,capacity):
    cdef size_t n=len(gradient),cap=capacity,count=0,next_slot=0,i,k
    cdef vector[double] us,vs,ratios,curvatures,alphas,q,p,g,zero,s,y,u,v
    cdef double gamma=1.,slope=0.
    cdef int status
    if n==0 or cap==0:raise ValueError('Invalid fixture capacity.')
    us.resize(cap*n);vs.resize(cap*n);ratios.resize(cap);curvatures.resize(cap);alphas.resize(cap)
    q.resize(n);p.resize(n);g.resize(n);zero.resize(n,0.);s.resize(n);y.resize(n);u.resize(n);v.resize(n)
    for i in range(n):g[i]=gradient[i]
    codes=[]
    for step,delta in pairs:
        if len(step)!=n or len(delta)!=n:raise ValueError('Invalid fixture shape.')
        for i in range(n):s[i]=step[i];y[i]=delta[i]
        with nogil:
            status=lbfgs_store(zero.data(),y.data(),s.data(),n,cap,&count,&next_slot,
                us.data(),vs.data(),ratios.data(),curvatures.data(),&gamma,u.data(),v.data())
        codes.append(status)
    with nogil:
        status=lbfgs_direction(g.data(),n,cap,count,next_slot,us.data(),vs.data(),
            ratios.data(),curvatures.data(),gamma,alphas.data(),q.data(),p.data(),&slope)
    return status,tuple(p[i] for i in range(n)),count,next_slot,gamma,tuple(codes)


def dogleg_vector(matrix,gradient,gauss_newton,radius):
    cdef size_t n=len(gradient),m=len(matrix),i,k
    cdef vector[double] a,g,gn,c,ag,out
    cdef double delta=radius
    cdef int status,kind=-1
    if n==0 or m==0 or len(gauss_newton)!=n:raise ValueError('Invalid fixture shape.')
    a.resize(m*n);g.resize(n);gn.resize(n);c.resize(n);ag.resize(m);out.resize(n)
    for i in range(m):
        if len(matrix[i])!=n:raise ValueError('Invalid fixture shape.')
        for k in range(n):a[i+m*k]=matrix[i][k]
    for k in range(n):g[k]=gradient[k];gn[k]=gauss_newton[k]
    with nogil:status=dogleg_step(a.data(),g.data(),gn.data(),n,m,delta,c.data(),ag.data(),out.data(),&kind)
    return status,kind,tuple(out[i] for i in range(n))
