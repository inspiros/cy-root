# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native secant state machine with shared scalar callbacks and copied history."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.math cimport fabs, fmax, isfinite, NAN, INFINITY
from libc.string cimport memset

from .scalar import NativeCallbackError, _nonnegative_integer
from cyroot._check_args import _check_stop_cond_args
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, RUNNING, RESIDUAL_TOLERANCE,
    MAX_ITERATIONS, BREAKDOWN, NONFINITE, CALLBACK_FAILURE,
)
from cyroot.return_types cimport RootReturnType
from cyroot.return_types import _normalize_reason


class SecantStep(NamedTuple):
    """Copied immutable state; termination_reason is None while running."""
    root: float
    f_root: float
    previous_root: float
    previous_f_root: float
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int


cdef class Secant:
    """Resumable scalar secant using Python or native residual callbacks.

    Two finite starting points need no ordering or bracket. Tolerances,
    iteration limits and bounded history follow native Newton conventions.
    Python callbacks acquire the GIL only for their calls; native callbacks
    permit initialization and complete numerical runs outside the GIL.
    """
    def __init__(self, function, x0, x1, f_x0=None, f_x1=None, *,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024):
        cdef double first, second, value0, value1
        cdef double f_atol, f_rtol, x_atol, x_rtol
        cdef bint cached_f0, cached_f1
        if self._constructing or self._callback is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            self._callback = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            self._enter()
            try:
                # All coercion can execute Python. Both construction and
                # callback guards precede it, including x0/x1 conversion.
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                limit = _nonnegative_integer(history, 'history')
                every = _nonnegative_integer(history_every, 'history_every')
                byte_limit = _nonnegative_integer(history_bytes, 'history_bytes')
                if every == 0:
                    raise ValueError('history_every must be positive.')
                if limit > byte_limit // sizeof(SecantRecord):
                    raise ValueError('Requested history exceeds history_bytes.')
                first, second = x0, x1
                cached_f0, cached_f1 = f_x0 is not None, f_x1 is not None
                value0 = NAN if f_x0 is None else f_x0
                value1 = NAN if f_x1 is None else f_x1
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                self._max_iter = max_iter
                self._history_every = every
                self._records.resize(limit)
                memset(&self._state, 0, sizeof(SecantRecord))
                self._state.root = self._state.previous_root = first
                self._state.f_root = self._state.previous_f_root = value0
                self._state.precision = INFINITY
                self._state.error = fabs(value0)
                init_stop(&self._stop, 0., fabs(first), f_atol, f_rtol, x_atol, x_rtol)
                with nogil:
                    self._initialize(second, value1, cached_f0, cached_f1)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._release()
        finally:
            self._constructing = False

    cdef void _ready(self) except *:
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')

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

    cdef void _release(self) noexcept:
        self._callback._active = False
        self._busy = False

    cdef int _evaluate(self, double x, double* out) noexcept nogil:
        cdef int code
        self._state.f_calls += 1
        out[0] = NAN
        code = self._callback._function(x, out, self._callback._context)
        if code != 0:
            self._state.callback_code = code
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self, double x1, double f_x1,
                          bint cached_f0, bint cached_f1) noexcept nogil:
        cdef double value = self._state.f_root
        if not isfinite(self._state.root):
            self._stop.reason = NONFINITE
            return
        if not cached_f0:
            if self._evaluate(self._state.root, &value) != 0:
                return
        if not isfinite(value):
            self._stop.reason = NONFINITE
            return
        self._state.f_root = self._state.previous_f_root = value
        self._state.error = fabs(value)
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, INFINITY, False):
            return
        # An already accepted first root does not need the second point/value.
        if not isfinite(x1):
            self._stop.reason = NONFINITE
            return
        value = f_x1
        if not cached_f1:
            if self._evaluate(x1, &value) != 0:
                return
        if not isfinite(value):
            self._stop.reason = NONFINITE
            return
        self._state.root, self._state.f_root = x1, value
        self._state.error = fabs(value)
        self._stop.f_scale = fmax(fabs(self._state.previous_f_root), fabs(value))
        self._stop.x_scale = self._stop.previous_x_norm = fabs(x1)
        self._check()

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
        if self._record_count > 0 and self._records[self._record_count - 1].iters == self._state.iters:
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
        cdef double scale, f0, f1, denominator, distance, factor, correction
        cdef double candidate, precision, value
        if self._stop.reason != RUNNING:
            return
        if self._state.f_root == self._state.previous_f_root:
            self._stop.reason = BREAKDOWN
            self._record()
            return
        scale = fmax(fabs(self._state.f_root), fabs(self._state.previous_f_root))
        f0, f1 = self._state.previous_f_root / scale, self._state.f_root / scale
        denominator = f1 - f0
        if denominator == 0.:
            self._stop.reason = BREAKDOWN
            self._record()
            return
        distance = self._state.root - self._state.previous_root
        factor = f1 / denominator
        correction = distance * factor
        candidate = self._state.root - correction
        precision = fabs(candidate - self._state.root)
        if (not isfinite(distance) or not isfinite(factor) or not isfinite(correction) or
                not isfinite(candidate) or not isfinite(precision)):
            self._stop.reason = NONFINITE
            self._record()
            return
        self._state.iters += 1
        if self._evaluate(candidate, &value) == 0:
            if not isfinite(value):
                self._stop.reason = NONFINITE
            else:
                self._state.previous_root, self._state.previous_f_root = self._state.root, self._state.f_root
                self._state.root, self._state.f_root = candidate, value
                self._state.precision, self._state.error = precision, fabs(value)
                advance_x(&self._stop, fabs(candidate))
                self._check()
        self._record()

    cdef void _raise_callback_error(self) except *:
        if self._callback._error is not None:
            error = self._callback._error
            self._callback._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result(), 'function')

    cdef object _snapshot(self, SecantRecord state):
        reason = None if state.reason == RUNNING else _normalize_reason(state.reason)
        return SecantStep(state.root, state.f_root, state.previous_root,
                          state.previous_f_root, state.iters, state.f_calls,
                          state.precision, state.error, reason, state.callback_code)

    cdef object _copy_result(self):
        cdef SecantRecord state = self._state
        return RootReturnType(state.root, state.f_root, state.iters, state.f_calls,
                              state.precision, state.error, False, False,
                              termination_reason=None if state.reason == RUNNING else state.reason)

    def step(self):
        """Attempt one finite candidate evaluation and return an immutable copy."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    self._step()
                self._raise_callback_error()
            return self._snapshot(self._state)
        finally:
            self._release()

    def run(self):
        """Use the remaining total iteration budget through the same step engine."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:
                        self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:
            self._release()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:
            return self._copy_result()
        finally:
            self._release()

    @property
    def history(self):
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i]) for i in range(self._record_count)])
        finally:
            self._release()

    @property
    def done(self):
        self._ready()
        return self._stop.reason != RUNNING

    @property
    def history_truncated(self):
        self._ready()
        return bool(self._history_truncated)

    @property
    def callback_code(self):
        self._ready()
        return self._state.callback_code

    @property
    def callback_stage(self):
        self._ready()
        return 'function' if self._stop.reason == CALLBACK_FAILURE else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


def secant(function, x0, x1, **options):
    """Convenience full run; use Secant to inspect immutable steps/history."""
    return Secant(function, x0, x1, **options).run()
