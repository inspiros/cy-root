# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Inverse-BFGS for residual objectives, with bounded strong-Wolfe search."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.math cimport NAN, INFINITY, fabs, fmax, isfinite
from libc.string cimport memcpy
from numbers import Complex, Real
from cyroot._stopping cimport within_tolerance
from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .higher_derivatives import _checked_integer
from .lbfgs_math cimport lbfgs_direction,lbfgs_store
from .bfgs_math cimport (bfgs_value,bfgs_gradient,bfgs_stationarity,bfgs_change,bfgs_dot,bfgs_identity,
                         bfgs_direction,bfgs_trial,bfgs_update)

cdef enum:
    B_RUNNING=0
    B_ROOT=1
    B_STATIONARY=2
    B_STAGNATION=3
    B_ITERATIONS=4
    B_EVALUATIONS=5
    B_LINE_SEARCH=6
    B_NONFINITE=7
    B_CALLBACK=8
    B_DERIVATIVE=9

_REASONS=(None,'residual_tolerance','stationary_nonroot','stagnation',
          'max_iterations','max_evaluations','line_search_failure','nonfinite',
          'callback_failure','derivative_breakdown')

class BFGSResult(NamedTuple):
    root: tuple
    f_root: tuple
    error: float
    precision: float
    value: float
    gradient: object
    objective_scale: float
    step_size: float
    iters: int
    f_calls: int
    jac_calls: int
    rejected_trials: int
    skipped_updates: int
    restarts: int
    success: bool
    termination_reason: object
    callback_code: int
    callback_stage: object

cdef double _norm(const double* values, size_t count) noexcept nogil:
    cdef size_t i
    cdef double norm=0.
    for i in range(count):
        if not isfinite(values[i]):return INFINITY
        norm=fmax(norm,fabs(values[i]))
    return norm

cdef double _option(object value, str name) except *:
    cdef double result
    if isinstance(value,(str,bytes,bytearray,bool)) or (isinstance(value,Complex) and not isinstance(value,Real)):
        raise TypeError(name+' must be a real scalar.')
    result=value
    if not isfinite(result) or result<0.:
        raise ValueError(name+' must be finite and nonnegative.')
    return result

cdef void _pack(object point, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(point,(str,bytes,bytearray)):
        raise TypeError('Expected a numeric point vector.')
    if is_native_double_buffer(point):
        guard=DoubleBuffer(point)
        view=guard.vector_view()
        if view.size()!=n:raise ValueError('Point length must match input_dimension.')
        for i in range(n):out[i]=view.get(i)
    else:
        if len(point)!=n:raise ValueError('Point length must match input_dimension.')
        for i in range(n):
            value=point[i]
            if isinstance(value,Complex) and not isinstance(value,Real):
                raise TypeError('Point coordinates must be real scalars.')
            out[i]=value

cdef int _difference_residual(const double* x,size_t n,size_t m,
                              double* out,void* context) noexcept nogil:
    return (<BFGS><object>context)._difference_sample(x,out)

cdef class BFGS:
    """Minimize 1/2||F/scale||^2 with a fixed initial residual scale.

    Only residual tolerance means root success. Native callbacks run without
    GIL; absent Jacobians use native central differences. value/gradient refer
    to the scaled objective; objective_scale is fixed throughout the solve.
    gtol bounds the residual-relative stationarity ||J^T F||/(||F|| max|J_ij|),
    so it does not depend on the residual scale or on how close F is to zero.
    Strong-Wolfe search uses bounded doubling/bisection. Root candidates may
    terminate before sampling their Jacobian; their gradient is None.
    """
    def __init__(self,objective,x0,*,etol=ETOL,ertol=ERTOL,ptol=PTOL,prtol=PRTOL,
                 gtol=1e-12,max_iter=MAX_ITER,max_evaluations=10000,max_line_search=32,
                 c1=1e-4,c2=.9,history=0,history_every=1,
                 workspace_bytes=64*1024*1024):
        self._setup(objective,x0,etol,ertol,ptol,prtol,gtol,max_iter,max_evaluations,
                    max_line_search,c1,c2,history,history_every,workspace_bytes,None)

    cdef void _setup(self,object objective,object x0,object etol,object ertol,
        object ptol,object prtol,object gtol,object max_iter,object max_evaluations,
        object max_line_search,object c1,object c2,object history,object history_every,
        object workspace_bytes,object corrections) except *:
        cdef size_t i
        cdef int status=0
        cdef bint reserved=False
        if self._constructing or self._initialized or self._objective is not None:
            raise RuntimeError('Solver states cannot be reinitialized.')
        self._constructing=True
        self._busy=True
        try:
            if not isinstance(objective,LeastSquaresObjective):
                raise TypeError('objective must be a LeastSquaresObjective.')
            self._objective=objective
            self._objective._enter()
            reserved=True
            self._objective._callback_code,self._objective._callback_stage=0,0
            self._n,self._m=self._objective._n,self._objective._m
            limit=_checked_integer(workspace_bytes,'workspace_bytes')
            # As for every root solver, zero or None removes the iteration limit;
            # the positive evaluation budget still bounds the run.
            max_iter=0 if max_iter is None else _checked_integer(max_iter,'max_iter')
            max_evaluations=_checked_integer(max_evaluations,'max_evaluations')
            max_line_search=_checked_integer(max_line_search,'max_line_search')
            history=_checked_integer(history,'history')
            history_every=_checked_integer(history_every,'history_every')
            if max_evaluations==0 or max_line_search==0 or history_every==0:
                raise ValueError('Evaluation/search budgets and history_every must be positive.')
            self._max_iter,self._max_evaluations,self._max_line_search=max_iter,max_evaluations,max_line_search
            self._history_every=history_every
            self._etol,self._ertol=_option(etol,'etol'),_option(ertol,'ertol')
            self._ptol,self._prtol=_option(ptol,'ptol'),_option(prtol,'prtol')
            self._gtol=_option(gtol,'gtol')
            self._c1,self._c2=_option(c1,'c1'),_option(c2,'c2')
            if self._ertol>=1.:raise ValueError('ertol must be smaller than one.')
            if not 0.<self._c1<self._c2<1.:
                raise ValueError('Line search requires 0 < c1 < c2 < 1.')
            self._limited=corrections is not None
            correction_count=0 if corrections is None else _checked_integer(corrections,'corrections')
            if self._limited and correction_count==0:
                raise ValueError('corrections must be positive.')
            self._corrections=correction_count
            n,m=int(self._n),int(self._m)
            diff_count=3*n+2*m if self._objective._j==NULL else 0
            model_count=2*correction_count*n+3*correction_count if self._limited else 2*n*n
            count=9*n+2*m+n*m+model_count+diff_count
            history_count=history*(n+m)
            total=(count+history_count)*sizeof(double)+history*sizeof(BFGSRecord)
            if (count>OwnedDoubleBuffer.max_size() or history_count>OwnedDoubleBuffer.max_size() or
                    history>self._records.max_size() or total>limit):
                raise ValueError('Solver workspace and history exceed workspace_bytes or native capacities.')
            if self._objective._j==NULL:
                self._difference=RectangularJacobianPlan(n,m,memory_bytes=limit)
            self._square=0 if self._limited else n*n
            self._memory_bytes=total
            self._storage=new OwnedDoubleBuffer(count)
            self._history_data=new OwnedDoubleBuffer(history_count)
            self._records.resize(history)
            self._x=self._storage.data()
            self._f=self._x+self._n
            self._g=self._f+self._m
            self._trial=self._g+self._n
            self._trial_f=self._trial+self._n
            self._trial_g=self._trial_f+self._m
            self._j=self._trial_g+self._n
            self._direction=self._j+self._n*self._m
            self._dx=self._direction+self._n
            self._u=self._dx+self._n
            if self._limited:
                self._pairs_u=self._u
                self._pairs_v=self._pairs_u+self._corrections*self._n
                self._ratios=self._pairs_v+self._corrections*self._n
                self._curvatures=self._ratios+self._corrections
                self._alphas=self._curvatures+self._corrections
                self._u=self._alphas+self._corrections
            else:
                self._h=self._u
                self._candidate=self._h+self._square
                self._u=self._candidate+self._square
            self._v=self._u+self._n
            self._w=self._v+self._n
            self._diff_work=self._w+self._n
            self._state.error,self._state.precision=INFINITY,INFINITY
            self._state.value,self._state.gradient=NAN,NAN
            self._scale=1.
            _pack(x0,self._x,self._n)
            if not isfinite(_norm(self._x,self._n)):
                raise ValueError('Initial point must be finite.')
            for i in range(self._m):self._f[i]=NAN
            self._initialized=True
            with nogil:
                self._reset_model()
                status=self._evaluate(self._x,self._f)
                if status!=0:self._fail(status)
                else:
                    self._state.error=_norm(self._f,self._m)
                    self._scale=self._state.error if self._state.error>0. else 1.
                    status=bfgs_value(self._f,self._m,self._scale,&self._state.value)
                    if status!=0:self._fail(status)
                    elif within_tolerance(self._state.error,self._etol,self._ertol,self._scale):
                        self._state.reason=B_ROOT
                    else:
                        status=self._gradient(self._x,self._f,self._g)
                        if status!=0:self._fail(status)
                        else:
                            self._state.gradient=_norm(self._g,self._n)
                            if self._stationary():self._state.reason=B_STATIONARY
                self._record()
            self._raise_callback_error()
        finally:
            if reserved:self._objective._release()
            self._busy=False
            self._constructing=False

    def __dealloc__(self):
        if self._storage!=NULL:del self._storage
        if self._history_data!=NULL:del self._history_data

    cdef void _reset_model(self) noexcept nogil:
        if self._limited:
            self._pair_count,self._pair_next=0,0
            self._pair_scale=1.
        else:bfgs_identity(self._h,self._n)

    cdef int _direction_model(self,double* slope) noexcept nogil:
        if self._limited:
            return lbfgs_direction(self._g,self._n,self._corrections,self._pair_count,
                self._pair_next,self._pairs_u,self._pairs_v,self._ratios,self._curvatures,
                self._pair_scale,self._alphas,self._w,self._direction,slope)
        return bfgs_direction(self._h,self._g,self._n,self._direction,slope)

    cdef int _update_model(self) noexcept nogil:
        if self._limited:
            return lbfgs_store(self._g,self._trial_g,self._dx,self._n,self._corrections,
                &self._pair_count,&self._pair_next,self._pairs_u,self._pairs_v,
                self._ratios,self._curvatures,&self._pair_scale,self._u,self._v)
        return bfgs_update(self._h,self._g,self._trial_g,self._dx,self._n,
            self._candidate,self._u,self._v,self._w)

    cdef void _ready(self) except *:
        if self._constructing or not self._initialized:
            raise RuntimeError('Solver state is not initialized.')
        if self._busy:raise RuntimeError('Solver state is already active.')

    cdef void _enter(self) except *:
        self._ready()
        self._busy=True
        try:self._objective._enter()
        except:
            self._busy=False
            raise
        self._objective._callback_code=self._state.callback_code
        self._objective._callback_stage=self._state.callback_stage

    cdef void _release(self) noexcept:
        self._objective._release()
        self._busy=False

    cdef int _evaluate(self,const double* point,double* out) noexcept nogil:
        self._state.f_calls+=1
        return self._objective._sample(1,point,out)

    cdef int _difference_sample(self,const double* point,double* out) noexcept nogil:
        cdef int status=self._objective._sample(1,point,out)
        return self._objective._callback_code if status==3 else 0

    cdef int _gradient(self,const double* point,const double* f,double* out) noexcept nogil:
        cdef int status=0,code=0
        if self._difference is not None and self._max_evaluations-self._state.f_calls<2*self._n:
            return 4
        self._state.jac_calls+=1
        if self._difference is None:
            status=self._objective._sample(2,point,self._j)
        else:
            status=self._difference.compute(point,f,self._j,self._diff_work,
                _difference_residual,<void*>self,&self._state.f_calls,&code)
        if status!=0:return status
        return bfgs_gradient(f,self._j,self._n,self._m,self._scale,out)

    cdef bint _stationary(self) noexcept nogil:
        # _j holds the Jacobian sampled with the latest accepted gradient.
        cdef double measure=INFINITY
        if bfgs_stationarity(self._f,self._j,self._n,self._m,&measure)!=0:return False
        return measure<=self._gtol

    cdef void _fail(self,int status) noexcept nogil:
        if status==3:
            self._state.reason=B_CALLBACK
            self._state.callback_code=self._objective._callback_code
            self._state.callback_stage=self._objective._callback_stage
        elif status==4:self._state.reason=B_EVALUATIONS
        else:self._state.reason=B_NONFINITE if status==1 else B_DERIVATIVE

    cdef void _record(self) noexcept nogil:
        cdef size_t offset
        if self._records.empty():return
        if self._state.reason==B_RUNNING and self._state.iters%self._history_every!=0:return
        if self._record_count==self._records.size():
            self._history_truncated=True
            return
        self._records[self._record_count]=self._state
        offset=self._record_count*(self._n+self._m)
        memcpy(self._history_data.data()+offset,self._x,self._n*sizeof(double))
        memcpy(self._history_data.data()+offset+self._n,self._f,self._m*sizeof(double))
        self._record_count+=1

    cdef void _step(self) noexcept nogil:
        cdef size_t search
        cdef int status=0
        cdef double slope=0.,next_slope=0.,alpha=1.,lo=0.,hi=INFINITY
        cdef double lo_value=self._state.value,value=0.,change=0.,precision=0.,xscale=0.,next_alpha=0.
        cdef bint accepted=False,root=False
        if self._state.reason!=B_RUNNING:return
        if self._max_iter and self._state.iters>=self._max_iter:self._state.reason=B_ITERATIONS
        elif self._state.f_calls>=self._max_evaluations:self._state.reason=B_EVALUATIONS
        if self._state.reason!=B_RUNNING:
            self._record()
            return
        status=self._direction_model(&slope)
        if status!=0 or not slope<0.:
            self._reset_model()
            self._state.restarts+=1
            status=self._direction_model(&slope)
        if status!=0 or not slope<0.:
            self._state.reason=B_NONFINITE if status!=0 else B_STAGNATION
            self._record()
            return
        self._state.iters+=1
        for search in range(self._max_line_search):
            if self._state.f_calls>=self._max_evaluations:
                self._state.reason=B_EVALUATIONS
                break
            status=bfgs_trial(self._x,self._direction,self._n,alpha,self._trial,self._dx)
            if status==0:
                precision=_norm(self._dx,self._n)
                if precision==0.:
                    # Keep precision tied to the last accepted step.
                    self._state.reason=B_STAGNATION
                    break
                status=self._evaluate(self._trial,self._trial_f)
                if status==3:
                    self._fail(status)
                    break
                if status==0:
                    root=within_tolerance(_norm(self._trial_f,self._m),self._etol,self._ertol,self._scale)
                    status=bfgs_value(self._trial_f,self._m,self._scale,&value)
                    if status==0:status=bfgs_change(self._f,self._trial_f,self._m,self._scale,&change)
                    if status==0:
                        if root:
                            accepted=True
                            break
                        if change<=self._c1*alpha*slope and (lo==0. or value<lo_value):
                            status=self._gradient(self._trial,self._trial_f,self._trial_g)
                            if status!=0:
                                self._fail(status)
                                break
                            status=bfgs_dot(self._trial_g,self._direction,self._n,&next_slope)
                            if status!=0:
                                self._state.reason=B_NONFINITE
                                break
                            if fabs(next_slope)<=-self._c2*slope:
                                accepted=True
                                break
                            if next_slope<0.:
                                lo=alpha
                                lo_value=value
                            else:hi=alpha
                        else:hi=alpha
                    else:hi=alpha
                else:hi=alpha
            else:hi=alpha
            self._state.rejected_trials+=1
            if isfinite(hi):next_alpha=lo+.5*(hi-lo)
            else:next_alpha=alpha*2.
            if not isfinite(next_alpha) or next_alpha==alpha or next_alpha==lo or next_alpha>1e16:
                break
            alpha=next_alpha
        if accepted:
            xscale=fmax(_norm(self._x,self._n),_norm(self._trial,self._n))
            if not root:
                status=self._update_model()
                if status!=0:self._state.skipped_updates+=1
                memcpy(self._g,self._trial_g,self._n*sizeof(double))
                self._state.gradient=_norm(self._g,self._n)
            else:self._state.gradient=NAN
            memcpy(self._x,self._trial,self._n*sizeof(double))
            memcpy(self._f,self._trial_f,self._m*sizeof(double))
            self._state.error=_norm(self._f,self._m)
            self._state.value=value
            self._state.precision=precision
            self._state.step_size=alpha
            if root:self._state.reason=B_ROOT
            elif self._stationary():self._state.reason=B_STATIONARY
            elif within_tolerance(precision,self._ptol,self._prtol,xscale):self._state.reason=B_STAGNATION
            elif self._max_iter and self._state.iters>=self._max_iter:self._state.reason=B_ITERATIONS
            elif self._state.f_calls>=self._max_evaluations:self._state.reason=B_EVALUATIONS
        elif self._state.reason==B_RUNNING:self._state.reason=B_LINE_SEARCH
        self._record()

    cdef void _run(self) noexcept nogil:
        while self._state.reason==B_RUNNING:self._step()

    cdef object _snapshot(self,BFGSRecord record,const double* x,const double* f):
        return BFGSResult(tuple([x[i] for i in range(self._n)]),tuple([f[i] for i in range(self._m)]),
            record.error,record.precision,record.value,
            record.gradient if isfinite(record.gradient) else None,self._scale,record.step_size,
            record.iters,record.f_calls,record.jac_calls,record.rejected_trials,
            record.skipped_updates,record.restarts,record.reason==B_ROOT,_REASONS[record.reason],
            record.callback_code,'function' if record.callback_stage==1 else
            'jacobian' if record.callback_stage==2 else None)

    cdef void _raise_callback_error(self) except *:
        if self._state.reason==B_CALLBACK:
            self._objective._raise_callback_error(self._snapshot(self._state,self._x,self._f))

    def step(self):
        self._enter()
        try:
            active=self._state.reason==B_RUNNING
            with nogil:self._step()
            if active:self._raise_callback_error()
            return self._snapshot(self._state,self._x,self._f)
        finally:self._release()

    def run(self):
        self._enter()
        try:
            active=self._state.reason==B_RUNNING
            with nogil:self._run()
            if active:self._raise_callback_error()
            return self._snapshot(self._state,self._x,self._f)
        finally:self._release()

    @property
    def result(self):
        self._enter()
        try:return self._snapshot(self._state,self._x,self._f)
        finally:self._release()

    @property
    def history(self):
        cdef size_t i
        cdef const double* data
        self._enter()
        try:
            records=[]
            for i in range(self._record_count):
                data=self._history_data.data()+i*(self._n+self._m)
                records.append(self._snapshot(self._records[i],data,data+self._n))
            return tuple(records)
        finally:self._release()

    @property
    def done(self):self._ready();return self._state.reason!=B_RUNNING
    @property
    def history_truncated(self):self._ready();return bool(self._history_truncated)
    @property
    def memory_bytes(self):self._ready();return self._memory_bytes


def bfgs(objective,x0,**options):
    """Find a root through BFGS minimization of an existing residual objective."""
    return BFGS(objective,x0,**options).run()


cdef class LBFGS(BFGS):
    """BFGS line-search/state contract with a bounded secant-pair ring.

    corrections limits retained pairs; inverse-Hessian storage is O(n*corrections).
    The residual Jacobian is still dense. No box bounds are imposed.
    """
    def __init__(self,objective,x0,*,corrections=10,etol=ETOL,ertol=ERTOL,ptol=PTOL,
                 prtol=PRTOL,gtol=1e-12,max_iter=MAX_ITER,max_evaluations=10000,
                 max_line_search=32,c1=1e-4,c2=.9,history=0,history_every=1,
                 workspace_bytes=64*1024*1024):
        self._setup(objective,x0,etol,ertol,ptol,prtol,gtol,max_iter,max_evaluations,
                    max_line_search,c1,c2,history,history_every,workspace_bytes,corrections)


def lbfgs(objective,x0,**options):
    """Find a root through limited-memory BFGS; returns the shared BFGSResult."""
    return LBFGS(objective,x0,**options).run()
