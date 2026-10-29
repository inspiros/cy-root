# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Interval-enclosure splitting bisection with fixed native workspace."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.math cimport fabs, fmax, isfinite, NAN
from libc.string cimport memset
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from cyroot._check_args import _check_stop_cond_args
from cyroot._stopping cimport (
    init_stop, check_stop, residual_met, successful, RUNNING, COMPLETED,
    RESIDUAL_TOLERANCE, MAX_ITERATIONS, MAX_SPLIT_ITERATIONS, STAGNATION,
    NONFINITE, BREAKDOWN, CALLBACK_FAILURE,
)
from cyroot.return_types import _normalize_reason
from cyroot.return_types cimport SplittingBracketingMethodsReturnType
from .scalar import NativeCallbackError, _nonnegative_integer


class HybisectStep(NamedTuple):
    """One copied search event, with global call/split/iteration counters."""
    bracket: tuple
    root: float
    f_root: float
    roots_found: int
    split_iters: int
    iters: int
    f_calls: tuple
    pending: int
    event: str
    termination_reason: object


cdef int _python_interval(double a, double b, double* lower,
                          double* upper, void* context) noexcept nogil:
    with gil:
        callback = <IntervalCallback><object>context
        try:
            first, second = callback._python_function(a,b)
            lower[0], upper[0] = first, second
        except BaseException as error:
            callback._error = error
            return 1
    return 0


cdef class IntervalCallback:
    """An enclosure function called as function(a,b), returning (lower,upper).

    A valid enclosure contains every f(x) for a <= x <= b. The producer is
    responsible for its numerical validity, including outward rounding.
    """
    def __init__(self,function):
        if self._function != NULL:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        if not callable(function):
            raise TypeError('interval function must be callable.')
        self._python_function = function
        self._function = _python_interval
        self._context = <void*>self

    @property
    def native(self):
        return bool(self._native)

    @classmethod
    def from_capsule(cls,capsule,keepalive=None):
        cdef cyroot_interval_descriptor* descriptor
        if not PyCapsule_IsValid(capsule,b'cyroot.interval_callback.v1'):
            raise ValueError('Expected a cyroot.interval_callback.v1 capsule.')
        descriptor = <cyroot_interval_descriptor*>PyCapsule_GetPointer(capsule,b'cyroot.interval_callback.v1')
        if descriptor.struct_size < sizeof(cyroot_interval_descriptor) or descriptor.abi_version != 1:
            raise ValueError('Unsupported interval callback descriptor size or ABI version.')
        return make_interval_callback(descriptor.function,descriptor.context,(capsule,keepalive))


cdef IntervalCallback make_interval_callback(IntervalFunction function,
                                             void* context,object keepalive=None):
    if function == NULL:
        raise ValueError('Native interval callback function must not be NULL.')
    cdef IntervalCallback callback = IntervalCallback.__new__(IntervalCallback)
    callback._function,callback._context = function,context
    callback._keepalive,callback._native = keepalive,True
    return callback


cdef class Hybisect:
    """Search enclosures depth first, then bisect each sign-changing interval.

    step() advances one interval/split/bisection event. run() releases the GIL
    for the full remaining search. Native scalar and interval callbacks make
    that search entirely native. A sign-changing interval yields one root;
    this algorithm is not an isolation/completeness certificate for all roots.
    """
    def __init__(self,function,interval_function,a,b,*,
                 etol=ETOL,ertol=ERTOL,
                 ptol=PTOL,prtol=PRTOL,
                 max_iter=MAX_ITER,max_split_iter=200,max_pending=4096,max_roots=4096,
                 workspace_bytes=8*1024*1024,history=0,history_every=1,
                 history_bytes=8*1024*1024):
        cdef double first,second,fa,fr,xa,xr
        cdef size_t pending_limit,root_limit,limit,every,bytes_limit,work
        if self._constructing or self._initialized or self._function is not None:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            self._function = function if isinstance(function,ScalarCallback) else ScalarCallback(function)
            self._interval_function = interval_function if isinstance(interval_function,IntervalCallback) else IntervalCallback(interval_function)
            self._enter()
            try:
                etol,ertol,ptol,prtol,max_iter = _check_stop_cond_args(etol,ertol,ptol,prtol,max_iter)
                self._max_iter = max_iter
                self._max_splits = _nonnegative_integer(max_split_iter,'max_split_iter')
                pending_limit = _nonnegative_integer(max_pending,'max_pending')
                root_limit = _nonnegative_integer(max_roots,'max_roots')
                work = _nonnegative_integer(workspace_bytes,'workspace_bytes')
                limit = _nonnegative_integer(history,'history')
                every = _nonnegative_integer(history_every,'history_every')
                bytes_limit = _nonnegative_integer(history_bytes,'history_bytes')
                if pending_limit == 0 or root_limit == 0 or every == 0:
                    raise ValueError('max_pending, max_roots and history_every must be positive.')
                if pending_limit > work // sizeof(Interval):
                    raise ValueError('Requested pending intervals exceed workspace_bytes.')
                work -= pending_limit*sizeof(Interval)
                if root_limit > work // sizeof(ScalarRecord):
                    raise ValueError('Requested root storage exceeds workspace_bytes.')
                if limit > bytes_limit // sizeof(SplitRecord):
                    raise ValueError('Requested history exceeds history_bytes.')
                first,second = a,b
                if second < first:first,second = second,first
                fa,fr,xa,xr = etol,ertol,ptol,prtol
                self._pending.resize(pending_limit)
                self._roots.resize(root_limit)
                self._records.resize(limit)
                self._history_every = every
                memset(&self._state,0,sizeof(ScalarRecord))
                self._state.a,self._state.b = first,second
                self._state.root,self._state.f_root = NAN,NAN
                init_stop(&self._stop,0.,fmax(fabs(first),fabs(second)),fa,fr,xa,xr)
                self._pending[0].a,self._pending[0].b = first,second
                self._pending_count = 1
                self._search_reason = RUNNING
                if not isfinite(first) or not isfinite(second):
                    self._search_reason = NONFINITE
                self._initialized = True
                self._record()
            finally:
                self._release()
        finally:
            self._constructing = False

    cdef void _ready(self) except *:
        if self._busy:raise RuntimeError('This solver is already running.')
        if not self._initialized:raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._busy:raise RuntimeError('This solver is already running.')
        if self._function is None or self._interval_function is None:
            raise RuntimeError('Both callback adapters are required.')
        if self._function._function == NULL or self._interval_function._function == NULL:
            raise RuntimeError('Callback adapters must be initialized.')
        if self._function._active or self._interval_function._active:
            raise RuntimeError('This callback adapter is already active.')
        self._busy = self._function._active = self._interval_function._active = True
        self._function._error = self._interval_function._error = None

    cdef void _release(self) noexcept:
        self._function._active = self._interval_function._active = self._busy = False

    cdef int _evaluate(self,double point,double* out) noexcept nogil:
        self._state.f_calls += 1
        out[0] = NAN
        cdef int code = self._function._function(point,out,self._function._context)
        if code != 0:
            self._state.callback_code = code
            self._callback_stage = 1
            self._search_reason = CALLBACK_FAILURE
        elif not isfinite(out[0]):
            self._search_reason = NONFINITE
        return code

    cdef void _check(self) noexcept nogil:
        self._state.precision = fabs(self._state.b-self._state.a)
        self._stop.x_scale = fmax(fabs(self._state.a),fabs(self._state.b))
        if check_stop(&self._stop,self._state.error,self._state.precision,
                      self._state.iters>0,True):return
        if self._max_iter and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS
            return
        cdef double mid = self._state.a/2+self._state.b/2
        if mid == self._state.a or mid == self._state.b:
            self._stop.reason = STAGNATION

    cdef void _finish_branch(self) noexcept nogil:
        cdef size_t i
        cdef bint duplicate = False
        self._state.reason = <int>self._stop.reason
        for i in range(self._root_count):
            if self._roots[i].root == self._state.root:
                duplicate = True
                break
        if not duplicate:
            if self._root_count == self._roots.size():
                self._workspace_exhausted = True
                self._search_reason = BREAKDOWN
                return
            self._roots[self._root_count] = self._state
            self._root_count += 1
        self._bracketing = False
        self._event = 4
        if not successful(self._stop.reason):
            self._search_reason = self._stop.reason
        elif self._pending_count == 0:
            self._search_reason = COMPLETED

    cdef void _process(self) noexcept nogil:
        cdef double a,b,lower=NAN,upper=NAN,mid,value=NAN
        cdef int code
        if self._search_reason != RUNNING:return
        if self._bracketing:
            self._event = 3
            mid = self._state.a/2+self._state.b/2
            self._state.iters += 1
            self._total_iters += 1
            if self._evaluate(mid,&value) != 0 or self._search_reason != RUNNING:return
            self._state.root,self._state.f_root,self._state.error = mid,value,fabs(value)
            if residual_met(&self._stop,self._state.error):
                self._stop.reason = RESIDUAL_TOLERANCE
            elif (value>0) == (self._state.f_a>0):
                self._state.a,self._state.f_a = mid,value
            else:
                self._state.b,self._state.f_b = mid,value
            self._check()
            if self._stop.reason != RUNNING:self._finish_branch()
            return
        if self._pending_count == 0:
            self._search_reason = COMPLETED
            return
        self._event = 1
        self._pending_count -= 1
        a,b = self._pending[self._pending_count].a,self._pending[self._pending_count].b
        self._state.a,self._state.b = a,b
        self._state.root,self._state.f_root = NAN,NAN
        self._state.iters = 0
        self._state.f_a,self._state.f_b = NAN,NAN
        self._state.precision,self._state.error = fabs(b-a),NAN
        self._interval_calls += 1
        code = self._interval_function._function(a,b,&lower,&upper,self._interval_function._context)
        if code != 0:
            self._state.callback_code = code
            self._callback_stage = 2
            self._search_reason = CALLBACK_FAILURE
            return
        if not isfinite(lower) or not isfinite(upper):
            self._search_reason = NONFINITE
            return
        if lower > upper:
            self._search_reason = BREAKDOWN
            return
        if lower > 0 or upper < 0:
            if self._pending_count == 0:self._search_reason = COMPLETED
            return
        if self._evaluate(a,&self._state.f_a) != 0 or self._search_reason != RUNNING:return
        if a == b:
            self._state.f_b = self._state.f_a
        elif self._evaluate(b,&self._state.f_b) != 0 or self._search_reason != RUNNING:
            return
        self._stop.reason = RUNNING
        self._stop.f_scale = fmax(fabs(self._state.f_a),fabs(self._state.f_b))
        if fabs(self._state.f_a) <= fabs(self._state.f_b):
            self._state.root,self._state.f_root = a,self._state.f_a
        else:
            self._state.root,self._state.f_root = b,self._state.f_b
        self._state.error = fabs(self._state.f_root)
        if residual_met(&self._stop,self._state.error):
            self._stop.reason = RESIDUAL_TOLERANCE
            self._finish_branch()
            return
        if (self._state.f_a>0) != (self._state.f_b>0):
            self._bracketing = True
            self._check()
            if self._stop.reason != RUNNING:self._finish_branch()
            return
        if self._max_splits and self._splits >= self._max_splits:
            self._search_reason = MAX_SPLIT_ITERATIONS
            return
        mid = a/2+b/2
        if mid == a or mid == b:
            self._search_reason = STAGNATION
            return
        if self._pending.size()-self._pending_count < 2:
            self._workspace_exhausted = True
            self._search_reason = BREAKDOWN
            return
        self._pending[self._pending_count].a,self._pending[self._pending_count].b = mid,b
        self._pending[self._pending_count+1].a,self._pending[self._pending_count+1].b = a,mid
        self._pending_count += 2
        self._splits += 1
        self._event = 2

    cdef SplitRecord _current(self) noexcept nogil:
        cdef SplitRecord record
        record.state = self._state
        record.splits,record.total_iters = self._splits,self._total_iters
        record.interval_calls,record.roots = self._interval_calls,self._root_count
        record.pending,record.event = self._pending_count,self._event
        record.search_reason = <int>self._search_reason
        return record

    cdef void _record(self) noexcept nogil:
        if self._records.size() == 0:return
        if self._events % self._history_every != 0 and self._search_reason == RUNNING:return
        if self._record_count >= self._records.size():
            self._history_truncated = True
            return
        self._records[self._record_count] = self._current()
        self._record_count += 1

    cdef void _step(self) noexcept nogil:
        if self._search_reason != RUNNING:return
        self._events += 1
        self._process()
        self._record()

    cdef void _raise_callback_error(self) except *:
        for callback in (self._function,self._interval_function):
            if isinstance(callback,ScalarCallback):
                error = (<ScalarCallback>callback)._error
                (<ScalarCallback>callback)._error = None
            else:
                error = (<IntervalCallback>callback)._error
                (<IntervalCallback>callback)._error = None
            if error is not None:raise error
        if self._search_reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code,self._copy_result(),
                'function' if self._callback_stage==1 else 'interval')

    cdef object _snapshot(self,SplitRecord record):
        return HybisectStep((record.state.a,record.state.b),record.state.root,
            record.state.f_root,record.roots,record.splits,record.total_iters,
            (record.state.f_calls,record.interval_calls),record.pending,
            ('initial','interval','split','bisection','root')[record.event],
            None if record.search_reason==RUNNING else _normalize_reason(record.search_reason))

    cdef object _copy_result(self):
        cdef ScalarRecord root
        roots=[];values=[];iters=[];brackets=[];f_brackets=[];precisions=[];errors=[];reasons=[]
        for i in range(self._root_count):
            root = self._roots[i]
            roots.append(root.root);values.append(root.f_root);iters.append(root.iters)
            brackets.append((root.a,root.b));f_brackets.append((root.f_a,root.f_b))
            precisions.append(root.precision);errors.append(root.error);reasons.append(root.reason)
        return SplittingBracketingMethodsReturnType(roots,values,self._splits,iters,
            (self._state.f_calls,self._interval_calls),brackets,f_brackets,
            precisions,errors,termination_reasons=reasons,
            search_termination_reason=None if self._search_reason==RUNNING else <int>self._search_reason)

    def step(self):
        self._ready();self._enter()
        try:
            with nogil:self._step()
            self._raise_callback_error()
            return self._snapshot(self._current())
        finally:self._release()

    def run(self):
        self._ready();self._enter()
        try:
            with nogil:
                while self._search_reason == RUNNING:self._step()
            self._raise_callback_error()
            return self._copy_result()
        finally:self._release()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:return self._copy_result()
        finally:self._release()

    @property
    def done(self):
        self._ready()
        return self._search_reason != RUNNING

    @property
    def history(self):
        self._ready()
        self._enter()
        try:return tuple(self._snapshot(self._records[i]) for i in range(self._record_count))
        finally:self._release()

    @property
    def history_truncated(self):
        self._ready()
        return bool(self._history_truncated)

    @property
    def workspace_exhausted(self):
        self._ready()
        return bool(self._workspace_exhausted)

    def __iter__(self):return self
    def __next__(self):
        if self.done:raise StopIteration
        return self.step()


def hybisect(function,interval_function,a,b,**options):
    return Hybisect(function,interval_function,a,b,**options).run()
