# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Scaled Levenberg-Marquardt with augmented rank-aware least-squares steps."""
from typing import NamedTuple
from libc.math cimport NAN, INFINITY, fabs, fmax, fmin, isfinite
from libc.string cimport memcpy
from numbers import Complex, Real
from cyroot._stopping cimport within_tolerance
from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .higher_derivatives import _checked_integer
from .dogleg_math cimport dogleg_prepare,dogleg_step
from .lm_math cimport lm_prepare, lm_trial, lm_reductions

cdef enum:
    LM_RUNNING=0
    LM_ROOT=1
    LM_STATIONARY=2
    LM_STAGNATION=3
    LM_ITERATIONS=4
    LM_EVALUATIONS=5
    LM_NO_STEP=6
    LM_LINEAR=7
    LM_NONFINITE=8
    LM_CALLBACK=9
    LM_DERIVATIVE=10

_REASONS = (None, 'residual_tolerance', 'stationary_nonroot', 'stagnation',
            'max_iterations', 'max_evaluations', 'no_acceptable_step',
            'linear_failure', 'nonfinite', 'callback_failure', 'derivative_breakdown')

class LevenbergMarquardtResult(NamedTuple):
    root: tuple
    f_root: tuple
    error: float
    precision: float
    iters: int
    f_calls: int
    jac_calls: int
    rejected_trials: int
    damping: float
    gradient: float
    success: bool
    termination_reason: object
    linear_status: int
    linear_rank: object
    callback_code: int
    callback_stage: object

class PowellDoglegResult(NamedTuple):
    root: tuple
    f_root: tuple
    error: float
    precision: float
    iters: int
    f_calls: int
    jac_calls: int
    rejected_trials: int
    trust_radius: float
    gradient: float
    success: bool
    termination_reason: object
    linear_status: int
    linear_rank: object
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

cdef int _difference_residual(const double* x, size_t n, size_t m,
                              double* out, void* context) noexcept nogil:
    # The owner holds its own and the objective's reservation and lifetime.
    return (<LevenbergMarquardt><object>context)._difference_sample(x,out)

cdef class LevenbergMarquardt:
    """Minimize 1/2||F||^2 using damped, column-scaled Gauss-Newton steps.

    Only residual tolerance means root success. A stationary least-squares
    point may have a nonzero residual. The existing objective supplies native
    or Python callbacks; absent Jacobians use native central differences.
    """
    def __init__(self, objective, x0, *, etol=1e-10, ertol=0., ptol=1e-12,
                 prtol=0., gtol=1e-12, max_iter=100, max_evaluations=10000,
                 max_trials=16, damping=1e-3, rcond=1e-12, history=0,
                 history_every=1, workspace_bytes=64*1024*1024):
        self._setup(objective,x0,etol,ertol,ptol,prtol,gtol,max_iter,max_evaluations,
                    max_trials,damping,rcond,history,history_every,workspace_bytes,
                    False,None,None,None)

    cdef void _setup(self,object objective,object x0,object etol,object ertol,
        object ptol,object prtol,object gtol,object max_iter,object max_evaluations,
        object max_trials,object damping,object rcond,object history,object history_every,
        object workspace_bytes,bint dogleg,object initial_radius,object max_radius,
        object min_radius) except *:
        cdef size_t i
        cdef int status
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
            # All user conversions occur after both reservations are held.
            self._n,self._m=self._objective._n,self._objective._m
            self._dogleg=dogleg
            self._rows=self._m if dogleg else self._n+self._m
            if self._rows>32767:
                raise ValueError('Augmented rows (input_dimension+output_dimension) must not exceed 32767.')
            limit=_checked_integer(workspace_bytes,'workspace_bytes')
            # As for every root solver, zero or None removes the iteration limit;
            # the positive evaluation budget still bounds the run.
            max_iter=0 if max_iter is None else _checked_integer(max_iter,'max_iter')
            max_evaluations=_checked_integer(max_evaluations,'max_evaluations')
            max_trials=_checked_integer(max_trials,'max_trials')
            history=_checked_integer(history,'history')
            history_every=_checked_integer(history_every,'history_every')
            if max_evaluations==0 or max_trials==0 or history_every==0:
                raise ValueError('max_evaluations, max_trials and history_every must be positive.')
            self._max_iter,self._max_evaluations,self._max_trials=max_iter,max_evaluations,max_trials
            self._history_every=history_every
            self._etol,self._ertol=_option(etol,'etol'),_option(ertol,'ertol')
            self._ptol,self._prtol=_option(ptol,'ptol'),_option(prtol,'prtol')
            self._gtol=_option(gtol,'gtol')
            if dogleg:
                self._state.radius=_option(initial_radius,'initial_radius')
                self._min_radius=_option(min_radius,'min_radius')
                self._max_radius=_option(max_radius,'max_radius')
                if not 0.<self._min_radius<=self._state.radius<=self._max_radius:
                    raise ValueError('Require 0 < min_radius <= initial_radius <= max_radius.')
            else:self._state.damping=_option(damping,'damping')
            if self._ertol>=1.:raise ValueError('ertol must be smaller than one.')
            if not dogleg and not 1e-15<=self._state.damping<=1e15:
                raise ValueError('damping must be between 1e-15 and 1e15.')
            # Validate dimensions/budget before allocating solver arrays. Plans
            # query LAPACK but allocate no evaluation workspace themselves.
            self._linear=LeastSquaresPlan(self._rows,self._n,rcond=rcond,workspace_bytes=limit)
            if self._objective._j==NULL:
                self._difference=RectangularJacobianPlan(self._n,self._m,memory_bytes=limit)
            n,m=int(self._n),int(self._m)
            diff_count=3*n+2*m if self._difference is not None else 0
            count=5*n+2*m+n*m+int(self._rows)*n+int(self._rows)+int(self._linear._work_size)+diff_count+(3*n+m if dogleg else 0)
            history_count=history*(n+m)
            total=(count+history_count)*sizeof(double)+n*sizeof(int)+history*sizeof(LMRecord)
            if (count>OwnedDoubleBuffer.max_size() or history_count>OwnedDoubleBuffer.max_size() or
                    history>self._records.max_size() or total>limit):
                raise ValueError('Solver workspace and history exceed workspace_bytes or native capacities.')
            self._memory_bytes=total
            self._storage=new OwnedDoubleBuffer(count)
            self._history_data=new OwnedDoubleBuffer(history_count)
            self._pivots.resize(n)
            self._records.resize(history)
            self._x=self._storage.data()
            self._f=self._x+self._n
            self._j=self._f+self._m
            self._trial=self._j+self._n*self._m
            self._trial_f=self._trial+self._n
            self._scales=self._trial_f+self._m
            self._z=self._scales+self._n
            self._dx=self._z+self._n
            self._augmented=self._dx+self._n
            self._rhs=self._augmented+self._rows*self._n
            self._work=self._rhs+self._rows
            self._diff_work=self._work+self._linear._work_size
            if dogleg:
                self._dogleg_g=self._diff_work+<size_t>diff_count
                self._dogleg_c=self._dogleg_g+self._n
                self._dogleg_z=self._dogleg_c+self._n
                self._dogleg_ag=self._dogleg_z+self._n
            self._state.error,self._state.precision,self._state.gradient=INFINITY,INFINITY,NAN
            self._state.linear_rank=-1
            _pack(x0,self._x,self._n)
            if not isfinite(_norm(self._x,self._n)):
                raise ValueError('Initial point must be finite.')
            for i in range(self._m):self._f[i]=NAN
            self._initialized=True
            with nogil:
                status=self._evaluate(self._x,self._f)
                if status!=0:self._fail(status)
                else:
                    self._state.error=_norm(self._f,self._m)
                    self._initial_error=self._state.error
                    if within_tolerance(self._state.error,self._etol,self._ertol,self._initial_error):
                        self._state.reason=LM_ROOT
                self._record()
            self._raise_callback_error()
        finally:
            if reserved:self._objective._leave()
            self._busy=False
            self._constructing=False

    def __dealloc__(self):
        if self._storage!=NULL:del self._storage
        if self._history_data!=NULL:del self._history_data

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

    cdef void _leave(self) noexcept:
        self._objective._leave()
        self._busy=False

    cdef int _evaluate(self, const double* point, double* out) noexcept nogil:
        self._state.f_calls+=1
        return self._objective._sample(1,point,out)

    cdef int _difference_sample(self, const double* point, double* out) noexcept nogil:
        cdef int status=self._objective._sample(1,point,out)
        return self._objective._callback_code if status==3 else 0

    cdef void _fail(self, int status) noexcept nogil:
        if status==3:
            self._state.reason=LM_CALLBACK
            self._state.callback_code=self._objective._callback_code
            self._state.callback_stage=self._objective._callback_stage
        else:self._state.reason=LM_NONFINITE if status==1 else LM_DERIVATIVE

    cdef void _record(self) noexcept nogil:
        cdef size_t offset
        if self._records.empty():return
        if self._state.reason==LM_RUNNING and self._state.iters%self._history_every!=0:return
        if self._record_count==self._records.size():
            self._history_truncated=True
            return
        self._records[self._record_count]=self._state
        offset=self._record_count*(self._n+self._m)
        memcpy(self._history_data.data()+offset,self._x,self._n*sizeof(double))
        memcpy(self._history_data.data()+offset+self._n,self._f,self._m*sizeof(double))
        self._record_count+=1

    cdef void _step(self) noexcept nogil:
        cdef int status=0, code=0
        cdef size_t trial
        cdef double scale=1.,gradient=0.,actual=0.,predicted=0.,ratio=0.,precision=0.,xscale=0.
        cdef bint accepted=False
        if self._state.reason!=LM_RUNNING:return
        if self._dogleg:
            self._step_dogleg()
            return
        if self._max_iter and self._state.iters>=self._max_iter:
            self._state.reason=LM_ITERATIONS
        elif self._state.f_calls>=self._max_evaluations:
            self._state.reason=LM_EVALUATIONS
        elif self._difference is not None and self._max_evaluations-self._state.f_calls<2*self._n:
            self._state.reason=LM_EVALUATIONS
        if self._state.reason!=LM_RUNNING:
            self._record()
            return
        self._state.jac_calls+=1
        if self._difference is None:
            status=self._objective._sample(2,self._x,self._j)
        else:
            status=self._difference.compute(self._x,self._f,self._j,self._diff_work,
                _difference_residual,<void*>self,&self._state.f_calls,&code)
        if status!=0:
            self._fail(status)
            self._record()
            return
        self._state.iters+=1
        for trial in range(self._max_trials):
            status=lm_prepare(self._f,self._j,self._n,self._m,self._state.damping,
                self._scales,self._augmented,self._rhs,&scale,&gradient)
            if status!=0:
                self._state.reason=LM_NONFINITE
                break
            self._state.gradient=gradient
            if gradient<=self._gtol:
                self._state.reason=LM_STATIONARY
                break
            if self._state.f_calls>=self._max_evaluations:
                self._state.reason=LM_EVALUATIONS
                break
            self._state.linear_status=self._linear.compute(self._augmented,self._rhs,
                self._z,self._work,self._pivots.data(),&self._state.linear_rank,&status)
            if self._state.linear_status!=0:
                self._state.reason=LM_LINEAR
                break
            status=lm_trial(self._x,self._z,self._scales,self._n,scale,self._trial,self._dx)
            if status==0:
                precision=_norm(self._dx,self._n)
                if precision==0.:
                    self._state.precision=0.
                    self._state.reason=LM_STAGNATION
                    break
                status=self._evaluate(self._trial,self._trial_f)
                if status==3:
                    self._fail(status)
                    break
                if status==0:
                    status=lm_reductions(self._f,self._j,self._trial_f,self._dx,
                        self._n,self._m,scale,&actual,&predicted)
                    if status==0 and actual>0. and predicted>0.:
                        ratio=actual/predicted
                        if ratio>=1e-4:
                            xscale=fmax(_norm(self._x,self._n),_norm(self._trial,self._n))
                            memcpy(self._x,self._trial,self._n*sizeof(double))
                            memcpy(self._f,self._trial_f,self._m*sizeof(double))
                            self._state.error=_norm(self._f,self._m)
                            self._state.precision=precision
                            accepted=True
                            if ratio>.75:self._state.damping=fmax(1e-15,self._state.damping*.3)
                            elif ratio<.25:self._state.damping=fmin(1e15,self._state.damping*2.)
                            if within_tolerance(self._state.error,self._etol,self._ertol,self._initial_error):
                                self._state.reason=LM_ROOT
                            elif within_tolerance(precision,self._ptol,self._prtol,xscale):
                                self._state.reason=LM_STAGNATION
                            break
            self._state.rejected_trials+=1
            self._state.damping=fmin(1e15,self._state.damping*10.)
        if self._state.reason==LM_RUNNING:
            if not accepted:self._state.reason=LM_NO_STEP
            elif self._max_iter and self._state.iters>=self._max_iter:self._state.reason=LM_ITERATIONS
            elif self._state.f_calls>=self._max_evaluations:self._state.reason=LM_EVALUATIONS
        self._record()

    cdef void _step_dogleg(self) noexcept nogil:
        cdef int status=0,code=0,kind=0,info=0
        cdef size_t trial
        cdef double scale=1.,gradient=0.,actual=0.,predicted=0.,ratio=-1.
        cdef double precision=0.,xscale=0.,radius=0.
        cdef bint accepted=False
        if self._max_iter and self._state.iters>=self._max_iter:self._state.reason=LM_ITERATIONS
        elif self._state.f_calls>=self._max_evaluations:self._state.reason=LM_EVALUATIONS
        elif self._difference is not None and self._max_evaluations-self._state.f_calls<2*self._n:
            self._state.reason=LM_EVALUATIONS
        if self._state.reason!=LM_RUNNING:
            self._record()
            return
        self._state.jac_calls+=1
        if self._difference is None:status=self._objective._sample(2,self._x,self._j)
        else:
            status=self._difference.compute(self._x,self._f,self._j,self._diff_work,
                _difference_residual,<void*>self,&self._state.f_calls,&code)
        if status!=0:
            self._fail(status)
            self._record()
            return
        self._state.iters+=1
        status=dogleg_prepare(self._f,self._j,self._n,self._m,self._scales,
            self._augmented,self._rhs,self._dogleg_g,&scale,&gradient)
        self._state.gradient=gradient
        if status!=0:self._state.reason=LM_NONFINITE
        elif gradient<=self._gtol:self._state.reason=LM_STATIONARY
        else:
            self._state.linear_status=self._linear.compute(self._augmented,self._rhs,
                self._z,self._work,self._pivots.data(),&self._state.linear_rank,&info)
            if self._state.linear_status!=0:self._state.reason=LM_LINEAR
        if self._state.reason!=LM_RUNNING:
            self._record()
            return
        for trial in range(self._max_trials):
            if self._state.f_calls>=self._max_evaluations:
                self._state.reason=LM_EVALUATIONS
                break
            radius=self._state.radius
            status=dogleg_step(self._augmented,self._dogleg_g,self._z,self._n,self._m,
                radius,self._dogleg_c,self._dogleg_ag,self._dogleg_z,&kind)
            if status!=0:
                self._state.reason=LM_NONFINITE if status==1 else LM_NO_STEP
                break
            ratio=-1.
            status=lm_trial(self._x,self._dogleg_z,self._scales,self._n,scale,self._trial,self._dx)
            if status==0:
                precision=_norm(self._dx,self._n)
                if precision==0.:
                    self._state.reason=LM_STAGNATION
                    break
                status=self._evaluate(self._trial,self._trial_f)
                if status==3:
                    self._fail(status)
                    break
                if status==0:
                    status=lm_reductions(self._f,self._j,self._trial_f,self._dx,
                        self._n,self._m,scale,&actual,&predicted)
                    if status==0 and predicted>0.:ratio=actual/predicted
                    if status==0 and actual>0. and ratio>=1e-4:
                        xscale=fmax(_norm(self._x,self._n),_norm(self._trial,self._n))
                        memcpy(self._x,self._trial,self._n*sizeof(double))
                        memcpy(self._f,self._trial_f,self._m*sizeof(double))
                        self._state.error=_norm(self._f,self._m)
                        self._state.precision=precision
                        accepted=True
                        if within_tolerance(self._state.error,self._etol,self._ertol,self._initial_error):
                            self._state.reason=LM_ROOT
                        elif within_tolerance(precision,self._ptol,self._prtol,xscale):
                            self._state.reason=LM_STAGNATION
            if ratio<.25:self._state.radius=fmax(self._min_radius,.25*radius)
            elif ratio>.75 and kind!=0:self._state.radius=fmin(self._max_radius,2.*radius)
            if accepted:break
            self._state.rejected_trials+=1
            if radius<=self._min_radius:
                self._state.reason=LM_NO_STEP
                break
        if self._state.reason==LM_RUNNING:
            if not accepted:self._state.reason=LM_NO_STEP
            elif self._max_iter and self._state.iters>=self._max_iter:self._state.reason=LM_ITERATIONS
            elif self._state.f_calls>=self._max_evaluations:self._state.reason=LM_EVALUATIONS
        self._record()

    cdef void _run(self) noexcept nogil:
        while self._state.reason==LM_RUNNING:self._step()

    cdef object _snapshot(self, LMRecord record, const double* x, const double* f):
        result_type=PowellDoglegResult if self._dogleg else LevenbergMarquardtResult
        return result_type(
            tuple([x[i] for i in range(self._n)]),tuple([f[i] for i in range(self._m)]),
            record.error,record.precision,record.iters,record.f_calls,record.jac_calls,
            record.rejected_trials,record.radius if self._dogleg else record.damping,record.gradient,record.reason==LM_ROOT,
            _REASONS[record.reason],record.linear_status,
            None if record.linear_rank<0 else record.linear_rank,record.callback_code,
            'function' if record.callback_stage==1 else 'jacobian' if record.callback_stage==2 else None)

    cdef void _raise_callback_error(self) except *:
        if self._state.reason==LM_CALLBACK:
            self._objective._raise_callback_error(self._snapshot(self._state,self._x,self._f))

    def step(self):
        self._enter()
        try:
            active=self._state.reason==LM_RUNNING
            with nogil:self._step()
            if active:self._raise_callback_error()
            return self._snapshot(self._state,self._x,self._f)
        finally:self._leave()

    def run(self):
        self._enter()
        try:
            active=self._state.reason==LM_RUNNING
            with nogil:self._run()
            if active:self._raise_callback_error()
            return self._snapshot(self._state,self._x,self._f)
        finally:self._leave()

    @property
    def result(self):
        self._enter()
        try:return self._snapshot(self._state,self._x,self._f)
        finally:self._leave()

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
        finally:self._leave()

    @property
    def done(self):self._ready();return self._state.reason!=LM_RUNNING
    @property
    def history_truncated(self):self._ready();return bool(self._history_truncated)
    @property
    def memory_bytes(self):self._ready();return self._memory_bytes


def levenberg_marquardt(objective,x0,**options):
    """Solve an existing LeastSquaresObjective; return a copied root result."""
    return LevenbergMarquardt(objective,x0,**options).run()


cdef class PowellDogleg(LevenbergMarquardt):
    """Column/residual-scaled Powell dogleg for square or rectangular systems.

    trust_radius measures the Euclidean normalized step z, with dx_j=s*z_j/d_j,
    s=||F||inf and d_j=max_i|J_ij| (1 for zero columns). The model and scaling
    are refreshed at each accepted point. Rank describes the undamped scaled J.
    """
    def __init__(self,objective,x0,*,initial_radius=1.,max_radius=1e6,min_radius=1e-15,
                 etol=1e-10,ertol=0.,ptol=1e-12,prtol=0.,gtol=1e-12,max_iter=100,
                 max_evaluations=10000,max_trials=16,rcond=1e-12,history=0,
                 history_every=1,workspace_bytes=64*1024*1024):
        self._setup(objective,x0,etol,ertol,ptol,prtol,gtol,max_iter,max_evaluations,
                    max_trials,1e-3,rcond,history,history_every,workspace_bytes,
                    True,initial_radius,max_radius,min_radius)


def powell_dogleg(objective,x0,**options):
    """Find a root using scaled Powell dogleg; only residual tolerance succeeds."""
    return PowellDogleg(objective,x0,**options).run()
