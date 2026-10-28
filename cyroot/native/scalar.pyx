# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native scalar state machines and callback bridges (opt-in preview)."""
from typing import NamedTuple
from operator import index
from libc.math cimport fabs, fmax, isfinite, NAN, INFINITY
from libc.string cimport memset
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from cyroot._check_args import _check_stop_cond_args
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, residual_met, RUNNING, RESIDUAL_TOLERANCE,
    MAX_ITERATIONS, STAGNATION, BREAKDOWN, NONFINITE, CALLBACK_FAILURE, ZERO_DERIVATIVE,
)
from cyroot.return_types cimport BracketingMethodsReturnType, NewtonMethodsReturnType
from cyroot.return_types import _normalize_reason
from .scalar_derivatives cimport (
    ScalarDerivativePlan, ScalarResidualFunction, SCALAR_DIFF_NONFINITE,
    SCALAR_DIFF_BREAKDOWN, SCALAR_DIFF_CALLBACK_FAILURE,
)


class ScalarStep(NamedTuple):
    """Immutable copied state. A running solve has termination_reason=None."""
    root: float
    f_root: float
    bracket: tuple
    f_bracket: tuple
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int


class NativeCallbackError(RuntimeError):
    """A native callback returned a nonzero status; result copies its state."""
    def __init__(self, native_status, result, callback_stage=None):
        self.native_status = native_status
        self.result = result
        self.callback_stage = callback_stage
        super().__init__(f'Native callback failed with status {native_status}.')


class NewtonStep(NamedTuple):
    """Immutable Newton state; unevaluated derivatives use NaN."""
    root: float
    f_root: float
    df_root: float
    iters: int
    f_calls: tuple
    precision: float
    error: float
    termination_reason: object
    callback_code: int
    callback_stage: object


def _nonnegative_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a nonnegative integer, not bool.')
    try:
        result = index(value)
    except TypeError:
        raise ValueError(f'{name} must be a nonnegative integer.') from None
    if result < 0 or result > <size_t>-1:
        raise ValueError(f'{name} must fit a nonnegative size_t.')
    return result


cdef int _python_scalar(double x, double* out, void* context) noexcept nogil:
    with gil:
        callback = <ScalarCallback><object>context
        try:
            out[0] = callback._python_function(x)
        except BaseException as error:
            callback._error = error
            return 1
    return 0


cdef class ScalarCallback:
    """Python callable or versioned native function/context plus a keepalive.

    Native producers use make_scalar_callback from scalar.pxd, or supply a
    capsule named cyroot.scalar_callback.v1 containing callbacks.h's descriptor.
    A native function writes out[0] and returns zero on success. Nonzero codes
    stop the solve without consuming out[0]. The callback cannot throw through
    the ABI. Concurrent/reentrant use of one adapter is rejected.
    """
    def __init__(self, function):
        if self._function != NULL:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        if not callable(function):
            raise TypeError('function must be callable.')
        self._python_function = function
        self._function = _python_scalar
        self._context = <void*>self
        self._native = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_scalar_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.scalar_callback.v1'):
            raise ValueError('Expected a cyroot.scalar_callback.v1 capsule.')
        descriptor = <cyroot_scalar_descriptor*>PyCapsule_GetPointer(capsule, b'cyroot.scalar_callback.v1')
        if descriptor.struct_size < sizeof(cyroot_scalar_descriptor) or descriptor.abi_version != 1:
            raise ValueError('Unsupported scalar callback descriptor size or ABI version.')
        return make_scalar_callback(descriptor.function, descriptor.context, (capsule, keepalive))

    @property
    def native(self):
        return bool(self._native)


cdef ScalarCallback make_scalar_callback(ScalarFunction function, void* context, object keepalive=None):
    """Create an adapter; the producer must keep context valid via keepalive."""
    if function == NULL:
        raise ValueError('Native scalar callback function must not be NULL.')
    cdef ScalarCallback callback = ScalarCallback.__new__(ScalarCallback)
    callback._function = function
    callback._context = context
    callback._keepalive = keepalive
    callback._native = True
    return callback


cdef class Bisection:
    """Resumable bisection with native execution and bounded copied history.

    step() performs at most one iteration; run() uses the remaining total
    max_iter budget. Both release the GIL. Python callback adapters reacquire
    it for their call. history=0 disables recording; otherwise initial,
    every history_every-th iteration and terminal states are sampled until
    the capacity is full. A full history never stops the numerical solve.
    """
    def __init__(self, function, a, b, f_a=None, f_b=None,
                 etol=1e-10, ertol=8.881784197001252e-16,
                 ptol=1e-12, prtol=8.881784197001252e-16,
                 max_iter=200, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024, algo=1):
        cdef double first, second, value0, value1, fa, fr, xa, xr
        cdef bint cached_a, cached_b
        if self._constructing or self._callback is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            self._callback = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            self._enter()
            try:
                if isinstance(algo,str):
                    if algo not in ('default','modified','vrahatis'):
                        raise ValueError('Unknown bisection algorithm.')
                elif _nonnegative_integer(algo,'algo') not in (1,2):
                    raise ValueError('Unknown bisection algorithm.')
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                history_limit = _nonnegative_integer(history, 'history')
                history_every = _nonnegative_integer(history_every, 'history_every')
                history_bytes = _nonnegative_integer(history_bytes, 'history_bytes')
                if history_every == 0:
                    raise ValueError('history_every must be positive.')
                if history_limit > history_bytes // sizeof(ScalarRecord):
                    raise ValueError('Requested history exceeds history_bytes.')
                first,second = a,b
                cached_a,cached_b = f_a is not None,f_b is not None
                value0 = NAN if f_a is None else f_a
                value1 = NAN if f_b is None else f_b
                if second < first:
                    first,second,value0,value1 = second,first,value1,value0
                    cached_a,cached_b = cached_b,cached_a
                fa,fr,xa,xr = etol,ertol,ptol,prtol
                self._history_every = history_every
                self._max_iter = max_iter
                self._records.resize(history_limit)
                memset(&self._state, 0, sizeof(ScalarRecord))
                self._state.a, self._state.b = first, second
                self._state.f_a,self._state.f_b = value0,value1
                self._state.root, self._state.f_root = first, value0
                self._state.precision = fabs(second-first)
                self._state.error = fabs(value0)
                init_stop(&self._stop, 0., fmax(fabs(first),fabs(second)),fa,fr,xa,xr)
                with nogil:
                    self._initialize(cached_a, cached_b)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._leave()
        finally:
            self._constructing = False

    cdef void _enter(self) except *:
        if self._callback is None or self._callback._function == NULL:
            raise RuntimeError('Solver requires an initialized callback adapter.')
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if self._callback._active:
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        self._callback._active = True
        self._callback._error = None

    cdef void _leave(self) noexcept:
        self._callback._active = False
        self._busy = False

    cdef int _evaluate(self, double x, double* out) noexcept nogil:
        self._state.f_calls += 1
        out[0] = NAN
        cdef int code = self._callback._function(x, out, self._callback._context)
        if code != 0:
            self._state.callback_code = code
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self, bint cached_a, bint cached_b) noexcept nogil:
        if not isfinite(self._state.a) or not isfinite(self._state.b):
            self._stop.reason = NONFINITE
            return
        if not cached_a and self._evaluate(self._state.a, &self._state.f_a) != 0:
            return
        if not cached_b and self._evaluate(self._state.b, &self._state.f_b) != 0:
            return
        if not isfinite(self._state.f_a) or not isfinite(self._state.f_b):
            self._stop.reason = NONFINITE
            return
        self._stop.f_scale = fmax(fabs(self._state.f_a), fabs(self._state.f_b))
        if fabs(self._state.f_a) <= fabs(self._state.f_b):
            self._state.root, self._state.f_root = self._state.a, self._state.f_a
        else:
            self._state.root, self._state.f_root = self._state.b, self._state.f_b
        self._state.error = fabs(self._state.f_root)
        if residual_met(&self._stop, self._state.error):
            self._stop.reason = RESIDUAL_TOLERANCE
        elif (self._state.f_a > 0) == (self._state.f_b > 0):
            self._stop.reason = BREAKDOWN
        self._check()

    cdef void _check(self) noexcept nogil:
        self._state.precision = fabs(self._state.b - self._state.a)
        self._stop.x_scale = fmax(fabs(self._state.a), fabs(self._state.b))
        if check_stop(&self._stop, self._state.error, self._state.precision,
                      self._state.iters > 0, True):
            return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS
            return
        cdef double midpoint = self._state.a / 2 + self._state.b / 2
        if midpoint == self._state.a or midpoint == self._state.b:
            self._stop.reason = STAGNATION

    cdef void _record(self) noexcept nogil:
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:
            return
        if (self._record_count > 0 and
                self._records[self._record_count - 1].iters == self._state.iters):
            self._records[self._record_count - 1] = self._state
            return
        if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:
            return
        if self._record_count >= self._records.size():
            self._history_truncated = True
            return
        self._records[self._record_count] = self._state
        self._record_count += 1

    cdef void _step(self) noexcept nogil:
        if self._stop.reason != RUNNING:
            return
        cdef double midpoint = self._state.a / 2 + self._state.b / 2
        cdef double value = NAN
        self._state.iters += 1
        if self._evaluate(midpoint, &value) == 0:
            if not isfinite(value):
                self._stop.reason = NONFINITE
            else:
                self._state.root, self._state.f_root = midpoint, value
                self._state.error = fabs(value)
                if residual_met(&self._stop, self._state.error):
                    self._stop.reason = RESIDUAL_TOLERANCE
                else:
                    if (value > 0) == (self._state.f_a > 0):
                        self._state.a, self._state.f_a = midpoint, value
                    else:
                        self._state.b, self._state.f_b = midpoint, value
                self._check()
        self._record()

    cdef void _raise_callback_error(self) except *:
        if self._callback._error is not None:
            error = self._callback._error
            self._callback._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result())

    cdef object _snapshot(self, ScalarRecord record):
        reason = None if record.reason == RUNNING else _normalize_reason(record.reason)
        return ScalarStep(record.root, record.f_root, (record.a, record.b),
                          (record.f_a, record.f_b), record.iters, record.f_calls,
                          record.precision, record.error, reason, record.callback_code)

    def step(self):
        """Perform one iteration and return an immutable state snapshot."""
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    self._step()
                self._raise_callback_error()
        finally:
            self._leave()
        return self._snapshot(self._state)

    def run(self):
        """Use the remaining total iteration budget and return the result."""
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:
                        self._step()
                self._raise_callback_error()
        finally:
            self._leave()
        return self.result

    @property
    def result(self):
        """Copy the current state without advancing or finalizing the solve."""
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return self._copy_result()

    cdef object _copy_result(self):
        cdef ScalarRecord state = self._state
        reason = None if state.reason == RUNNING else state.reason
        return BracketingMethodsReturnType(
            state.root, state.f_root, state.iters, state.f_calls,
            (state.a, state.b), (state.f_a, state.f_b),
            state.precision, state.error, False, False, termination_reason=reason)

    @property
    def done(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return self._stop.reason != RUNNING

    @property
    def history(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        self._busy = True
        try:
            return tuple(self._snapshot(self._records[i]) for i in range(self._record_count))
        finally:
            self._busy = False

    @property
    def history_truncated(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return bool(self._history_truncated)

    @property
    def callback_code(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return self._state.callback_code

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


def bisect(function, a, b, **options):
    """Run native-preview bisection; use Bisection to retain history or step."""
    return Bisection(function, a, b, **options).run()


cdef class Newton:
    """Resumable scalar Newton with analytic or finite-difference derivatives.

    function and derivative accept the same ScalarCallback interface. Both are
    reserved together during setup/step/run; a single adapter may fill both
    roles. Pass None or a ScalarDerivativePlan for numerical derivatives. Stopping,
    history and total iteration limits match the native Bisection interface.
    """
    def __init__(self, function, derivative, x0, f_x0=None, df_x0=None,
                 etol=1e-10, ertol=8.881784197001252e-16,
                 ptol=1e-12, prtol=8.881784197001252e-16,
                 max_iter=200, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024):
        if self._constructing or self._function is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        cdef bint cached_f = f_x0 is not None, cached_df = df_x0 is not None
        cdef double initial, f_atol, f_rtol, x_atol, x_rtol
        try:
            self._function = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            if derivative is None:
                self._plan = ScalarDerivativePlan()
            elif isinstance(derivative, ScalarDerivativePlan):
                self._plan = derivative
            else:
                self._derivative = derivative if isinstance(derivative, ScalarCallback) else ScalarCallback(derivative)
            if self._plan is not None and not self._plan._initialized:
                raise RuntimeError('Scalar derivative plan is not initialized.')
            self._enter()
            try:
                # Conversions may run arbitrary Python; reserve the solver and
                # callback adapters before coercion, including x0/tolerances.
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                history_limit = _nonnegative_integer(history, 'history')
                history_every = _nonnegative_integer(history_every, 'history_every')
                history_bytes = _nonnegative_integer(history_bytes, 'history_bytes')
                if history_every == 0:
                    raise ValueError('history_every must be positive.')
                if history_limit > history_bytes // sizeof(NewtonRecord):
                    raise ValueError('Requested history exceeds history_bytes.')
                self._history_every = history_every
                self._max_iter = max_iter
                self._records.resize(history_limit)
                initial = x0
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                memset(&self._state, 0, sizeof(NewtonRecord))
                self._state.root = initial
                self._state.f_root = NAN if f_x0 is None else f_x0
                self._state.df_root = NAN if df_x0 is None else df_x0
                self._state.precision = INFINITY
                self._state.error = fabs(self._state.f_root)
                init_stop(&self._stop, 0., fabs(initial), f_atol, f_rtol, x_atol, x_rtol)
                with nogil:
                    self._initialize(cached_f, cached_df)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._leave()
        finally:
            self._constructing = False

    cdef void _enter(self) except *:
        if (self._function is None or self._function._function == NULL or
                (self._plan is None and
                 (self._derivative is None or self._derivative._function == NULL))):
            raise RuntimeError('Solver requires initialized function and derivative adapters.')
        if self._busy:
            raise RuntimeError('This solver is already running.')
        # Validate all adapters before reserving either; rollback is unnecessary
        # when a second adapter is already active. Aliases are reserved once.
        if self._function._active or (self._derivative is not None and self._derivative._active):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        self._function._active = True
        self._function._error = None
        if self._derivative is not None:
            self._derivative._active = True
            self._derivative._error = None

    cdef void _leave(self) noexcept:
        self._function._active = False
        if self._derivative is not None:
            self._derivative._active = False
        self._busy = False

    cdef int _evaluate(self, bint derivative, double x, double* out) noexcept nogil:
        cdef int code, status, sample_code = 0
        out[0] = NAN
        if derivative:
            self._state.df_calls += 1
            if self._plan is not None:
                status = self._plan.compute(
                    x, self._state.f_root, out,
                    <ScalarResidualFunction>self._function._function, self._function._context,
                    &self._state.f_calls, &sample_code)
                if status == SCALAR_DIFF_CALLBACK_FAILURE:
                    self._state.callback_code = sample_code
                    self._state.callback_stage = 2
                    self._stop.reason = CALLBACK_FAILURE
                elif status == SCALAR_DIFF_NONFINITE:
                    self._stop.reason = NONFINITE
                elif status == SCALAR_DIFF_BREAKDOWN:
                    self._stop.reason = BREAKDOWN
                return status
            code = self._derivative._function(x, out, self._derivative._context)
        else:
            self._state.f_calls += 1
            code = self._function._function(x, out, self._function._context)
        if code != 0:
            self._state.callback_code = code
            self._state.callback_stage = 2 if derivative else 1
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self, bint cached_f, bint cached_df) noexcept nogil:
        cdef double value = NAN
        if not isfinite(self._state.root):
            self._stop.reason = NONFINITE
            return
        if not cached_f:
            if self._evaluate(False, self._state.root, &value) != 0:
                return
            self._state.f_root = value
        self._state.error = fabs(self._state.f_root)
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, self._state.precision, False):
            return
        if not cached_df:
            if self._evaluate(True, self._state.root, &value) != 0:
                return
            self._state.df_root = value if isfinite(value) else NAN
        if not isfinite(self._state.df_root):
            self._stop.reason = NONFINITE
        elif self._state.df_root == 0:
            self._stop.reason = ZERO_DERIVATIVE

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop, self._state.error, self._state.precision,
                      self._state.iters > 0):
            return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS

    cdef void _record(self) noexcept nogil:
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:
            return
        if (self._record_count > 0 and
                self._records[self._record_count - 1].iters == self._state.iters):
            self._records[self._record_count - 1] = self._state
            return
        if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:
            return
        if self._record_count >= self._records.size():
            self._history_truncated = True
            return
        self._records[self._record_count] = self._state
        self._record_count += 1

    cdef void _step(self) noexcept nogil:
        if self._stop.reason != RUNNING:
            return
        cdef double delta = -self._state.f_root / self._state.df_root
        cdef double candidate = self._state.root + delta
        cdef double precision = fabs(candidate - self._state.root)
        cdef double value = NAN
        if not isfinite(delta) or not isfinite(candidate) or not isfinite(precision):
            self._stop.reason = NONFINITE
        else:
            self._state.iters += 1
            if self._evaluate(False, candidate, &value) == 0:
                if not isfinite(value):
                    self._stop.reason = NONFINITE
                else:
                    # Commit only a finite point/residual pair. Derivatives at
                    # the previous point cannot be labelled df_root here.
                    self._state.root, self._state.f_root = candidate, value
                    self._state.df_root = NAN
                    self._state.error = fabs(value)
                    self._state.precision = precision
                    advance_x(&self._stop, fabs(candidate))
                    self._check()
                    if self._stop.reason == RUNNING:
                        if self._evaluate(True, candidate, &value) == 0:
                            if not isfinite(value):
                                self._stop.reason = NONFINITE
                            else:
                                self._state.df_root = value
                                if value == 0:
                                    self._stop.reason = ZERO_DERIVATIVE
        self._record()

    cdef void _raise_callback_error(self) except *:
        # Detach error state before releasing adapter reservations at the
        # Python boundary. Only one callback failure can end a native run.
        if self._function._error is not None:
            error = self._function._error
            self._function._error = None
            raise error
        if self._derivative is not None and self._derivative._error is not None:
            error = self._derivative._error
            self._derivative._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            stage = 'function' if self._state.callback_stage == 1 else 'derivative'
            raise NativeCallbackError(self._state.callback_code, self._copy_result(), stage)

    cdef object _snapshot(self, NewtonRecord record):
        reason = None if record.reason == RUNNING else _normalize_reason(record.reason)
        stage = ('function' if record.callback_stage == 1 else
                 'derivative' if record.callback_stage == 2 else None)
        return NewtonStep(record.root, record.f_root, record.df_root, record.iters,
                          (record.f_calls, record.df_calls), record.precision,
                          record.error, reason, record.callback_code, stage)

    def step(self):
        """Attempt one Newton iteration and return an immutable state copy."""
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    self._step()
                self._raise_callback_error()
            return self._snapshot(self._state)
        finally:
            self._leave()

    def run(self):
        """Use the remaining total iteration budget through the step engine."""
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:
                        self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:
            self._leave()

    @property
    def result(self):
        """Copy the current result without evaluating or finalizing the solve."""
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return self._copy_result()

    cdef object _copy_result(self):
        cdef NewtonRecord state = self._state
        reason = None if state.reason == RUNNING else state.reason
        return NewtonMethodsReturnType(
            state.root, state.f_root, state.df_root,
            state.iters, (state.f_calls, state.df_calls),
            state.precision, state.error, False, False, termination_reason=reason)

    @property
    def done(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return self._stop.reason != RUNNING

    @property
    def history(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        self._busy = True
        try:
            return tuple(self._snapshot(self._records[i]) for i in range(self._record_count))
        finally:
            self._busy = False

    @property
    def history_truncated(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return bool(self._history_truncated)

    @property
    def callback_code(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return self._state.callback_code

    @property
    def callback_stage(self):
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')
        return ('function' if self._state.callback_stage == 1 else
                'derivative' if self._state.callback_stage == 2 else None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


def newton(function, derivative, x0, **options):
    """Run native-preview Newton; use Newton to retain history or step."""
    return Newton(function, derivative, x0, **options).run()
