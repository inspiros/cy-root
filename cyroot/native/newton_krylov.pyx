# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Matrix-free square Newton with restarted GMRES and Armijo backtracking."""
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport fabs,fmax,isfinite,NAN,INFINITY
from libc.string cimport memcpy,memset
from cpython.buffer cimport PyObject_CheckBuffer
from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer
from .rectangular cimport RectangularFunction
from .scalar import _nonnegative_integer,NativeCallbackError
from .derivatives cimport DIFF_OK,DIFF_NONFINITE,DIFF_BREAKDOWN
from cyroot._check_args import _check_stop_cond_args
from cyroot._stopping cimport RUNNING,RESIDUAL_TOLERANCE,MAX_ITERATIONS,NONFINITE,CALLBACK_FAILURE,STAGNATION,LINE_SEARCH_FAILURE,init_stop,advance_x,check_stop
from cyroot.status import TerminationReason

__all__=['NewtonKrylov','NewtonKrylovResult','newton_krylov']

cdef enum:
    LINEAR_FAILURE=13
    MAX_EVALUATIONS=14
    DERIVATIVE_BREAKDOWN=15

_REASONS=(None,'residual_tolerance','bracket_tolerance','stagnation','max_iterations',
          'zero_derivative','breakdown','nonfinite','max_split_iterations','completed',
          'unspecified','callback_failure','line_search_failure','linear_failure',
          'max_evaluations','derivative_breakdown')

class NewtonKrylovResult(NamedTuple):
    """Copied point/residual and diagnostics; linear convergence is not root success."""
    root: tuple
    f_root: tuple
    iters: int
    f_calls: tuple
    precision: float
    error: float
    termination_reason: object
    linear_iterations: int
    linear_residual: float
    linear_status: int
    step_scale: float
    backtracks: int
    callback_code: int
    callback_stage: object
    @property
    def success(self):return self.termination_reason=='residual_tolerance'
    @property
    def converged(self):return self.success
    @property
    def optimal(self):return self.success

cdef extern from "gmres.hpp" namespace "cyroot::native" nogil:
    ctypedef int (*KrylovOperator)(const double*,double*,void*) noexcept
    int restarted_gmres(size_t,size_t,size_t,double,const double*,double*,double*,double*,
                        KrylovOperator,void*,size_t*,double*) noexcept
    int krylov_merit_slope(const double*,const double*,size_t,double,double*) noexcept
cdef extern from "bfgs_math.hpp" namespace "cyroot::native" nogil:
    int bfgs_change(const double*,const double*,size_t,double,double*) noexcept

cdef void _fill(double* out,size_t n,double value) noexcept nogil:
    cdef size_t i
    for i in range(n):out[i]=value

cdef double _norm(const double* values,size_t n) noexcept nogil:
    cdef size_t i
    cdef double value=0.
    for i in range(n):
        if not isfinite(values[i]):return INFINITY
        value=fmax(value,fabs(values[i]))
    return value

cdef void _pack(object source,double* out,size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source,(str,bytes,bytearray)):raise TypeError('Expected a numeric vector.')
    if PyObject_CheckBuffer(source):
        guard=DoubleBuffer(source);view=guard.vector_view()
        if view.size()!=n:raise ValueError('NewtonKrylov requires equal nonempty point/residual lengths.')
        for i in range(n):out[i]=view.get(i)
    else:
        if len(source)!=<Py_ssize_t>n:raise ValueError('NewtonKrylov requires equal nonempty point/residual lengths.')
        for i in range(n):out[i]=source[i]

cdef int _operator(const double* v,double* out,void* context) noexcept nogil:
    return (<NewtonKrylov><object>context)._apply(v,out)

cdef int _sample(const double* x,size_t n,size_t m,double* out,void* context) noexcept nogil:
    return (<NewtonKrylov><object>context)._evaluate(x,out)

cdef class NewtonKrylov:
    """Resumable square Newton using J(x)v, never a materialized Jacobian.

    jvp accepts a Python (x,v) product, JacobianVectorCallback, matching immutable
    JacobianVectorPlan, or None for forward differences with cached F(x).
    GMRES uses a fixed forcing tolerance and bounded restarts, with no initial
    preconditioner. Each accepted step must pass Armijo decrease. Singular or
    poorly conditioned systems may fail the linear solve; no dense fallback is
    used. Rectangular problems remain supported by the other vector solvers.

    f_calls is (residual samples, product applications); numerical products count
    their residual samples. max_evaluations includes initialization, differences,
    and rejected trials. linear_iterations is cumulative Arnoldi work;
    linear_residual is the most recent verified relative 2-norm residual.
    History stores accepted states and terminal diagnostics, not inner GMRES or
    line-search trials. All native workspaces are allocated before stepping.
    """
    def __init__(self,function,x0,*,jvp=None,f_x0=None,restart=20,max_linear_iter=100,
                 linear_rtol=1e-3,c1=1e-4,contraction=.5,max_line_search=32,
                 etol=1e-10,ertol=8.881784197001252e-16,
                 ptol=1e-12,prtol=8.881784197001252e-16,max_iter=200,
                 max_evaluations=10000,history=0,history_every=1,
                 history_bytes=8*1024*1024,workspace_bytes=64*1024*1024):
        cdef DoubleBuffer guard
        cdef size_t n,i
        cdef double* data
        cdef double fa,fr,xa,xr
        if self._constructing or self._initialized or self._storage!=NULL or self._function is not None or self._square_function is not None:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing=True
        try:
            if isinstance(function,VectorCallback):self._square_function=function
            elif isinstance(function,RectangularCallback):self._function=function
            if isinstance(jvp,JacobianVectorCallback):self._product_callback=jvp
            elif isinstance(jvp,JacobianVectorPlan):self._plan=jvp
            self._enter()
            try:
                etol,ertol,ptol,prtol,max_iter=_check_stop_cond_args(etol,ertol,ptol,prtol,max_iter)
                limit=_nonnegative_integer(workspace_bytes,'workspace_bytes')
                history_limit=_nonnegative_integer(history,'history')
                every=_nonnegative_integer(history_every,'history_every')
                history_limit_bytes=_nonnegative_integer(history_bytes,'history_bytes')
                restart=_nonnegative_integer(restart,'restart')
                max_linear_iter=_nonnegative_integer(max_linear_iter,'max_linear_iter')
                max_evaluations=_nonnegative_integer(max_evaluations,'max_evaluations')
                max_line_search=_nonnegative_integer(max_line_search,'max_line_search')
                if not restart or not max_linear_iter or not max_evaluations or not max_line_search or not every:
                    raise ValueError('Restart, linear/evaluation/search budgets and history_every must be positive.')
                self._linear_rtol=float(linear_rtol);self._c1=float(c1);self._contraction=float(contraction)
                if not isfinite(self._linear_rtol) or not 0.<self._linear_rtol<1.:
                    raise ValueError('linear_rtol must be finite and strictly between zero and one.')
                if not isfinite(self._c1) or not 0.<self._c1<1. or not isfinite(self._contraction) or not 0.<self._contraction<1.:
                    raise ValueError('c1 and contraction must be finite and strictly between zero and one.')
                if isinstance(x0,(str,bytes,bytearray)):raise TypeError('Expected a numeric point vector.')
                if PyObject_CheckBuffer(x0):
                    guard=DoubleBuffer(x0);dimension=int(guard.vector_view().size())
                else:dimension=len(x0)
                if dimension<=0 or dimension>INT_MAX:raise ValueError('Point dimension must be positive and fit C int.')
                n=dimension
                if self._square_function is not None:
                    if self._square_function._dimension!=n:raise ValueError('NewtonKrylov requires a square residual of the point dimension.')
                elif self._function is not None:
                    if self._function._n!=n or self._function._m!=n:raise ValueError('NewtonKrylov currently supports square systems only.')
                else:
                    self._function=RectangularCallback(function,n,n);self._function._active=True
                if self._product_callback is not None:
                    if self._product_callback._n!=n or self._product_callback._m!=n:raise ValueError('Product dimensions must match the square system.')
                elif self._plan is not None:
                    if not self._plan._initialized or self._plan._constructing:raise RuntimeError('Product plan is not initialized.')
                    if self._plan._n!=n or self._plan._m!=n:raise ValueError('Product plan dimensions must match the square system.')
                elif jvp is None:self._plan=JacobianVectorPlan(n,n,scheme='forward',memory_bytes=limit)
                else:
                    self._product_callback=JacobianVectorCallback(jvp,n,n);self._product_callback._active=True
                k=min(dimension,restart,max_linear_iter)
                gmres=(k+1)*dimension+(k+1)*k+4*k+1+2*dimension
                diff=0 if self._plan is None else int(self._plan._work_size)
                count=7*dimension+diff+gmres
                history_count=2*dimension*history_limit
                if count>OwnedDoubleBuffer.max_size() or count*sizeof(double)>limit:
                    raise ValueError('Native Newton-Krylov workspace exceeds workspace_bytes.')
                if history_count>OwnedDoubleBuffer.max_size() or history_limit*(2*dimension*sizeof(double)+sizeof(KrylovRecord))>history_limit_bytes:
                    raise ValueError('Requested history exceeds history_bytes.')
                self._n=n;self._restart=k;self._max_linear_iter=max_linear_iter
                self._max_iter=max_iter;self._max_evaluations=max_evaluations;self._max_line_search=max_line_search
                self._history_every=every;self._memory_bytes=count*sizeof(double)
                self._storage=new OwnedDoubleBuffer(count);self._history_data=new OwnedDoubleBuffer(history_count)
                self._records.resize(history_limit)
                data=self._storage.data();self._x=data;self._f=data+n;self._delta=data+2*n
                self._trial=data+3*n;self._trial_f=data+4*n;self._applied=data+5*n;self._rhs=data+6*n
                self._difference_work=data+7*n;self._gmres_work=self._difference_work+<size_t>diff
                _pack(x0,self._x,n);_fill(self._f,n,NAN)
                if f_x0 is not None:_pack(f_x0,self._f,n)
                memset(&self._state,0,sizeof(KrylovRecord))
                self._state.precision=INFINITY;self._state.linear_residual=INFINITY;self._state.linear_status=-1
                fa,fr,xa,xr=etol,ertol,ptol,prtol
                init_stop(&self._stop,0.,_norm(self._x,n),fa,fr,xa,xr)
                with nogil:
                    if not isfinite(_norm(self._x,n)):self._state.reason=NONFINITE
                    elif f_x0 is None:self._evaluate(self._x,self._f)
                    self._state.error=_norm(self._f,n);self._stop.f_scale=self._state.error
                    if self._state.reason==RUNNING:self._check()
                    self._record()
                self._initialized=True
                self._raise_callback_error()
            finally:self._leave()
        finally:self._constructing=False

    def __dealloc__(self):
        if self._storage!=NULL:del self._storage
        if self._history_data!=NULL:del self._history_data

    cdef void _enter(self) except *:
        if self._busy:raise RuntimeError('This solver is already running.')
        if self._square_function is not None:
            if self._square_function._function==NULL or self._square_function._constructing:raise RuntimeError('Residual callback is not initialized.')
            if self._square_function._jacobian:raise ValueError('Expected a residual-kind callback.')
            if self._square_function._active:raise RuntimeError('This residual callback is already active.')
        if self._function is not None:
            if self._function._function==NULL or self._function._constructing:raise RuntimeError('Residual callback is not initialized.')
            if self._function._jacobian:raise ValueError('Expected a residual-kind callback.')
            if self._function._active:raise RuntimeError('This residual callback is already active.')
        if self._product_callback is not None:
            if self._product_callback._function==NULL or self._product_callback._constructing:raise RuntimeError('Product callback is not initialized.')
            if self._product_callback._active:raise RuntimeError('This product callback is already active.')
        self._busy=True
        if self._square_function is not None:self._square_function._active=True;self._square_function._error=None
        if self._function is not None:self._function._active=True;self._function._error=None
        if self._product_callback is not None:self._product_callback._active=True;self._product_callback._error=None

    cdef void _leave(self) noexcept:
        if self._square_function is not None:self._square_function._active=False
        if self._function is not None:self._function._active=False
        if self._product_callback is not None:self._product_callback._active=False
        self._busy=False

    cdef void _ready(self) except *:
        if self._busy:raise RuntimeError('This solver is already running.')
        if not self._initialized:raise RuntimeError('Solver is not initialized.')

    cdef int _evaluate(self,const double* point,double* out) noexcept nogil:
        cdef int code
        _fill(out,self._n,NAN)
        if self._state.f_calls>=self._max_evaluations:
            self._state.reason=MAX_EVALUATIONS;return -1
        self._state.f_calls+=1
        if self._square_function is not None:
            code=self._square_function._function(point,self._n,out,self._square_function._context)
        else:code=self._function.evaluate_raw(point,out)
        if code!=0:
            self._state.reason=CALLBACK_FAILURE;self._state.callback_code=code;self._state.callback_stage=1
            _fill(out,self._n,NAN);return code
        return 0

    cdef int _product(self,const double* direction,double* out) noexcept nogil:
        cdef size_t samples=0
        cdef int code=0,status
        self._state.jvp_calls+=1
        _fill(out,self._n,NAN)
        if self._plan is not None:
            status=self._plan.compute(self._x,direction,self._f,out,self._difference_work,
                                      _sample,<void*>self,&samples,&code)
            if self._state.reason!=RUNNING:return -1
            if status!=DIFF_OK:
                self._state.reason=NONFINITE if status==DIFF_NONFINITE else DERIVATIVE_BREAKDOWN
                return -1
        else:
            code=self._product_callback.evaluate_raw(self._x,direction,out)
            if code!=0:
                self._state.reason=CALLBACK_FAILURE;self._state.callback_code=code;self._state.callback_stage=2
                return code
        if not isfinite(_norm(out,self._n)):
            self._state.reason=NONFINITE;return -1
        return 0

    cdef int _apply(self,const double* direction,double* out) noexcept nogil:
        cdef size_t i
        if self._product(direction,out)!=0:return -1
        for i in range(self._n):
            out[i]/=self._scale
            if not isfinite(out[i]):self._state.reason=NONFINITE;return -1
        return 0

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop,self._state.error,self._state.precision,self._state.iters>0):
            self._state.reason=<int>self._stop.reason
        elif self._max_iter and self._state.iters>=self._max_iter:
            self._state.reason=MAX_ITERATIONS

    cdef void _record(self) noexcept nogil:
        cdef size_t slot
        cdef double* data
        if self._records.size()==0:return
        if self._record_count and self._records[self._record_count-1].iters==self._state.iters:slot=self._record_count-1
        else:
            if self._state.iters%self._history_every and self._state.reason==RUNNING:return
            if self._record_count>=self._records.size():self._history_truncated=True;return
            slot=self._record_count;self._record_count+=1
        self._records[slot]=self._state;data=self._history_data.data()+slot*2*self._n
        memcpy(data,self._x,self._n*sizeof(double));memcpy(data+self._n,self._f,self._n*sizeof(double))

    cdef void _step(self) noexcept nogil:
        cdef size_t i,attempt,iterations=0
        cdef double slope,change,precision,difference,alpha=1.
        cdef bint finite_trial
        if self._state.reason!=RUNNING:return
        self._state.iters+=1;self._state.step_scale=0.;self._state.backtracks=0
        self._scale=self._state.error
        for i in range(self._n):self._rhs[i]=-self._f[i]/self._scale
        self._state.linear_status=restarted_gmres(self._n,self._restart,self._max_linear_iter,
            self._linear_rtol,self._rhs,self._delta,self._applied,self._gmres_work,
            _operator,<void*>self,&iterations,&self._state.linear_residual)
        self._state.linear_iterations+=iterations
        if self._state.reason!=RUNNING:self._record();return
        if self._state.linear_status!=0:
            self._state.reason=NONFINITE if self._state.linear_status==3 else LINEAR_FAILURE
            self._record();return
        if krylov_merit_slope(self._f,self._applied,self._n,self._scale,&slope)!=0:
            self._state.reason=NONFINITE
        elif slope>=0.:self._state.reason=LINE_SEARCH_FAILURE
        else:
            for attempt in range(self._max_line_search):
                precision=0.;finite_trial=True
                for i in range(self._n):
                    self._trial[i]=self._x[i]+alpha*self._delta[i]
                    difference=fabs(self._trial[i]-self._x[i]);precision=fmax(precision,difference)
                    if not isfinite(self._trial[i]) or not isfinite(difference):finite_trial=False
                if finite_trial:
                    if precision==0.:self._state.precision=0.;self._state.reason=STAGNATION;break
                    if self._evaluate(self._trial,self._trial_f)!=0:break
                    if (isfinite(_norm(self._trial_f,self._n)) and bfgs_change(self._f,self._trial_f,self._n,self._scale,&change)==0
                            and change<0. and change<=self._c1*alpha*slope):
                        memcpy(self._x,self._trial,self._n*sizeof(double));memcpy(self._f,self._trial_f,self._n*sizeof(double))
                        self._state.error=_norm(self._f,self._n);self._state.precision=precision;self._state.step_scale=alpha
                        advance_x(&self._stop,_norm(self._x,self._n));self._check();break
                self._state.backtracks+=1;alpha*=self._contraction
            else:self._state.reason=LINE_SEARCH_FAILURE
        self._record()

    cdef void _run(self) noexcept nogil:
        while self._state.reason==RUNNING:self._step()

    cdef void _raise_callback_error(self) except *:
        error=None
        if self._square_function is not None and self._square_function._error is not None:
            error=self._square_function._error;self._square_function._error=None
        elif self._function is not None and self._function._error is not None:
            error=self._function._error;self._function._error=None
        elif self._product_callback is not None and self._product_callback._error is not None:
            error=self._product_callback._error;self._product_callback._error=None
        if error is not None:raise error
        if self._state.reason==CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code,self._snapshot(self._state,self._x,self._f),
                                      'function' if self._state.callback_stage==1 else 'jacobian_vector_product')

    cdef object _snapshot(self,KrylovRecord state,const double* x,const double* f):
        cdef size_t i
        reason=None if state.reason==RUNNING else TerminationReason(_REASONS[state.reason])
        return NewtonKrylovResult(tuple([x[i] for i in range(self._n)]),tuple([f[i] for i in range(self._n)]),
            state.iters,(state.f_calls,state.jvp_calls),state.precision,state.error,reason,state.linear_iterations,
            state.linear_residual,state.linear_status,state.step_scale,state.backtracks,state.callback_code,
            'function' if state.callback_stage==1 else 'jacobian_vector_product' if state.callback_stage==2 else None)

    def step(self):
        self._ready();self._enter()
        try:
            if self._state.reason==RUNNING:
                with nogil:self._step()
                self._raise_callback_error()
            return self._snapshot(self._state,self._x,self._f)
        finally:self._leave()

    def run(self):
        self._ready();self._enter()
        try:
            if self._state.reason==RUNNING:
                with nogil:self._run()
                self._raise_callback_error()
            return self._snapshot(self._state,self._x,self._f)
        finally:self._leave()

    @property
    def result(self):
        self._ready();self._busy=True
        try:return self._snapshot(self._state,self._x,self._f)
        finally:self._busy=False
    @property
    def history(self):
        cdef size_t i
        self._ready();self._busy=True
        try:return tuple([self._snapshot(self._records[i],self._history_data.data()+i*2*self._n,
                                         self._history_data.data()+i*2*self._n+self._n) for i in range(self._record_count)])
        finally:self._busy=False
    @property
    def done(self):self._ready();return self._state.reason!=RUNNING
    @property
    def history_truncated(self):self._ready();return bool(self._history_truncated)
    @property
    def workspace_bytes(self):self._ready();return self._memory_bytes
    @property
    def dimension(self):self._ready();return self._n
    @property
    def restart(self):self._ready();return self._restart
    def __iter__(self):return self
    def __next__(self):
        if self.done:raise StopIteration
        return self.step()


def newton_krylov(function,x0,*,jvp=None,**options):
    """Solve a square system through the resumable NewtonKrylov engine."""
    return NewtonKrylov(function,x0,jvp=jvp,**options).run()
