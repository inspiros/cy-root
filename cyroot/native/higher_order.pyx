# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Resumable Halley family with analytic or native numerical derivatives."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from numbers import Complex, Real
from libc.math cimport fabs, isfinite, NAN, INFINITY
from libc.string cimport memset

from .scalar import NativeCallbackError, _nonnegative_integer
from .scalar_derivatives cimport (
    ScalarResidualFunction, SCALAR_DIFF_CALLBACK_FAILURE,
    SCALAR_DIFF_NONFINITE, SCALAR_DIFF_BREAKDOWN,
)
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types cimport NewtonMethodsReturnType
from cyroot.return_types import _normalize_reason
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, RUNNING, MAX_ITERATIONS,
    BREAKDOWN, NONFINITE, CALLBACK_FAILURE, ZERO_DERIVATIVE,
)


cdef enum:
    HALLEY = 0
    CHEBYSHEV = 1
    MODIFIED_HALLEY = 2
    FUNCTION = 1
    DERIVATIVE = 2
    SECOND_DERIVATIVE = 3


class HigherOrderStep(NamedTuple):
    """Copied immutable point, derivatives, counters and diagnostics."""
    root: float
    f_root: float
    df_root: float
    d2f_root: float
    iters: int
    f_calls: tuple
    precision: float
    error: float
    termination_reason: object
    callback_code: int
    callback_stage: object


cdef object _stage_name(int stage):
    if stage == FUNCTION:
        return 'function'
    if stage == DERIVATIVE:
        return 'derivative'
    if stage == SECOND_DERIVATIVE:
        return 'second_derivative'
    return None


cdef class _ThirdOrderNewton:
    """Common guarded callbacks, stopping state, history and native step loop."""

    cdef void _setup(self, object function, object derivative,
                      object second_derivative, object x0, object f_x0,
                      object df_x0, object d2f_x0, object etol, object ertol,
                      object ptol, object prtol, object max_iter, object history,
                      object history_every, object history_bytes, int method,
                      object alpha) except *:
        cdef double initial, f_atol, f_rtol, x_atol, x_rtol
        cdef bint cached_f, cached_df, cached_d2f
        if self._constructing or self._function is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            # Validate all derivative roles even if cached residual convergence
            # will later make their numerical evaluations unnecessary.
            if isinstance(derivative, ScalarSecondDerivativePlan):
                raise TypeError('First derivative requires a first derivative plan or callback.')
            if isinstance(second_derivative, ScalarDerivativePlan):
                raise TypeError('Second derivative requires a second derivative plan or callback.')
            self._function = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            if derivative is None:
                self._plan = ScalarDerivativePlan()
            elif isinstance(derivative, ScalarDerivativePlan):
                self._plan = derivative
            else:
                self._derivative = derivative if isinstance(derivative, ScalarCallback) else ScalarCallback(derivative)
            if second_derivative is None:
                self._second_plan = ScalarSecondDerivativePlan()
            elif isinstance(second_derivative, ScalarSecondDerivativePlan):
                self._second_plan = second_derivative
            else:
                self._second_derivative = (second_derivative if isinstance(second_derivative, ScalarCallback)
                                           else ScalarCallback(second_derivative))
            if ((self._plan is not None and not self._plan._initialized) or
                    (self._second_plan is not None and not self._second_plan._initialized)):
                raise RuntimeError('Scalar derivative plans must be initialized.')
            self._enter()
            try:
                # Every user coercion, allocation and callback occurs while all
                # callback roles and this solver are reserved transactionally.
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                limit = _nonnegative_integer(history, 'history')
                every = _nonnegative_integer(history_every, 'history_every')
                byte_limit = _nonnegative_integer(history_bytes, 'history_bytes')
                if every == 0:
                    raise ValueError('history_every must be positive.')
                if limit > byte_limit // sizeof(HigherOrderRecord) or limit > self._records.max_size():
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                if method == MODIFIED_HALLEY:
                    if (isinstance(alpha, (str, bytes, bytearray, bool)) or
                            (isinstance(alpha, Complex) and not isinstance(alpha, Real)) or
                            getattr(getattr(alpha, 'dtype', None), 'kind', None) in ('b', 'c')):
                        raise TypeError('alpha must be a finite real scalar.')
                    try:
                        len(alpha)
                    except TypeError:
                        pass
                    else:
                        raise TypeError('alpha must be a finite real scalar.')
                    self._alpha = alpha
                    if not isfinite(self._alpha):
                        raise ValueError('alpha must be finite.')
                    # These family members have exactly the existing arithmetic
                    # and failure policy, including the Halley ratio form.
                    if self._alpha == 0.:
                        method = CHEBYSHEV
                    elif self._alpha == 0.5:
                        method = HALLEY
                initial = x0
                cached_f, cached_df, cached_d2f = f_x0 is not None, df_x0 is not None, d2f_x0 is not None
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                self._method = method
                self._max_iter = max_iter
                self._history_every = every
                self._records.resize(limit)
                memset(&self._state, 0, sizeof(HigherOrderRecord))
                self._state.root = initial
                self._state.f_root = NAN if f_x0 is None else f_x0
                self._state.df_root = NAN if df_x0 is None else df_x0
                self._state.d2f_root = NAN if d2f_x0 is None else d2f_x0
                self._state.precision = INFINITY
                self._state.error = fabs(self._state.f_root)
                init_stop(&self._stop, 0., fabs(initial), f_atol, f_rtol, x_atol, x_rtol)
                with nogil:
                    self._initialize(cached_f, cached_df, cached_d2f)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._leave()
        finally:
            self._constructing = False

    cdef void _ready(self) except *:
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if (self._function is None or self._function._function == NULL or
                (self._plan is None and
                 (self._derivative is None or self._derivative._function == NULL)) or
                (self._second_plan is None and
                 (self._second_derivative is None or self._second_derivative._function == NULL))):
            raise RuntimeError('Solver requires initialized residual and derivative providers.')
        if self._busy:
            raise RuntimeError('This solver is already running.')
        # Inspect every role before reserving any. Repeated aliases are safe;
        # an active third role cannot leave either earlier role reserved.
        if (self._function._active or
                (self._derivative is not None and self._derivative._active) or
                (self._second_derivative is not None and self._second_derivative._active)):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        self._function._active = True
        self._function._error = None
        if self._derivative is not None:
            self._derivative._active = True
            self._derivative._error = None
        if self._second_derivative is not None:
            self._second_derivative._active = True
            self._second_derivative._error = None

    cdef void _leave(self) noexcept:
        self._function._active = False
        if self._derivative is not None:
            self._derivative._active = False
        if self._second_derivative is not None:
            self._second_derivative._active = False
        self._busy = False

    cdef int _evaluate(self, int stage, double x, double* out) noexcept nogil:
        cdef int code = 0, status = 0, sample_code = 0
        out[0] = NAN
        if stage == FUNCTION:
            self._state.f_calls += 1
            code = self._function._function(x, out, self._function._context)
        elif stage == DERIVATIVE:
            self._state.df_calls += 1
            if self._plan is not None:
                status = self._plan.compute(
                    x, self._state.f_root, out,
                    <ScalarResidualFunction>self._function._function, self._function._context,
                    &self._state.f_calls, &sample_code)
            else:
                code = self._derivative._function(x, out, self._derivative._context)
        else:
            self._state.d2f_calls += 1
            if self._second_plan is not None:
                status = self._second_plan.compute(
                    x, self._state.f_root, out,
                    <ScalarResidualFunction>self._function._function, self._function._context,
                    &self._state.f_calls, &sample_code)
            else:
                code = self._second_derivative._function(x, out, self._second_derivative._context)
        if ((stage == DERIVATIVE and self._plan is not None) or
                (stage == SECOND_DERIVATIVE and self._second_plan is not None)):
            if status == SCALAR_DIFF_CALLBACK_FAILURE:
                self._state.callback_code = sample_code
                self._state.callback_stage = stage
                self._stop.reason = CALLBACK_FAILURE
            elif status == SCALAR_DIFF_NONFINITE:
                self._stop.reason = NONFINITE
            elif status == SCALAR_DIFF_BREAKDOWN:
                self._stop.reason = BREAKDOWN
            return status
        if code != 0:
            self._state.callback_code = code
            self._state.callback_stage = stage
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self, bint cached_f, bint cached_df,
                           bint cached_d2f) noexcept nogil:
        cdef double value = NAN
        if not isfinite(self._state.root):
            self._stop.reason = NONFINITE
            return
        if not cached_f:
            if self._evaluate(FUNCTION, self._state.root, &value) != 0:
                return
            self._state.f_root = value
        self._state.error = fabs(self._state.f_root)
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, INFINITY, False):
            return
        self._derivatives(cached_df, cached_d2f)

    cdef void _derivatives(self, bint cached_df, bint cached_d2f) noexcept nogil:
        cdef double value = self._state.df_root
        if not cached_df:
            if self._evaluate(DERIVATIVE, self._state.root, &value) != 0:
                self._state.df_root = self._state.d2f_root = NAN
                return
        if not isfinite(value):
            self._state.df_root = self._state.d2f_root = NAN
            self._stop.reason = NONFINITE
            return
        self._state.df_root = value
        if value == 0.:
            self._state.d2f_root = NAN
            self._stop.reason = ZERO_DERIVATIVE
            return
        value = self._state.d2f_root
        if not cached_d2f:
            if self._evaluate(SECOND_DERIVATIVE, self._state.root, &value) != 0:
                self._state.d2f_root = NAN
                return
        if not isfinite(value):
            self._state.d2f_root = NAN
            self._stop.reason = NONFINITE
            return
        self._state.d2f_root = value

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
        cdef double t, ratio, q, factor, delta, candidate, precision, value
        cdef double alpha_q, twice_alpha_q, denominator, correction
        if self._stop.reason != RUNNING:
            return
        # Ratios preserve common residual scaling without forming df*df or
        # F*d2f. Conservative nonfinite intermediates stop before any trial.
        t = self._state.f_root / self._state.df_root
        ratio = self._state.d2f_root / self._state.df_root
        if not isfinite(t) or not isfinite(ratio):
            self._stop.reason = NONFINITE
            self._record()
            return
        q = (0.5 * t) * ratio
        if not isfinite(q):
            self._stop.reason = NONFINITE
            self._record()
            return
        if self._method == MODIFIED_HALLEY:
            alpha_q = self._alpha * q
            twice_alpha_q = 2. * alpha_q
            denominator = 1. - twice_alpha_q
            if (not isfinite(alpha_q) or not isfinite(twice_alpha_q) or
                    not isfinite(denominator)):
                self._stop.reason = NONFINITE
                self._record()
                return
            if denominator == 0.:
                self._stop.reason = BREAKDOWN
                self._record()
                return
            correction = q / denominator
            factor = 1. + correction
            if not isfinite(correction) or not isfinite(factor):
                self._stop.reason = NONFINITE
                self._record()
                return
            delta = -t * factor
        else:
            factor = 1. - q if self._method == HALLEY else 1. + q
            if not isfinite(factor):
                self._stop.reason = NONFINITE
                self._record()
                return
            if self._method == HALLEY:
                if factor == 0.:
                    self._stop.reason = BREAKDOWN
                    self._record()
                    return
                delta = -t / factor
            else:
                delta = -t * factor
        candidate = self._state.root + delta
        precision = fabs(candidate - self._state.root)
        if not isfinite(delta) or not isfinite(candidate) or not isfinite(precision):
            self._stop.reason = NONFINITE
            self._record()
            return
        self._state.iters += 1
        if self._evaluate(FUNCTION, candidate, &value) == 0:
            if not isfinite(value):
                self._stop.reason = NONFINITE
            else:
                # Commit only a finite point/residual. Both old derivatives
                # belong to the old point and cannot label the new state.
                self._state.root, self._state.f_root = candidate, value
                self._state.df_root = self._state.d2f_root = NAN
                self._state.error, self._state.precision = fabs(value), precision
                advance_x(&self._stop, fabs(candidate))
                self._check()
                if self._stop.reason == RUNNING:
                    self._derivatives(False, False)
        self._record()

    cdef void _raise_callback_error(self) except *:
        # Detach the original Python exception without adding attributes or
        # releasing reservations before exception construction/transfer.
        if self._function._error is not None:
            error = self._function._error
            self._function._error = None
            raise error
        if self._derivative is not None and self._derivative._error is not None:
            error = self._derivative._error
            self._derivative._error = None
            raise error
        if self._second_derivative is not None and self._second_derivative._error is not None:
            error = self._second_derivative._error
            self._second_derivative._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result(),
                                      _stage_name(self._state.callback_stage))

    cdef object _snapshot(self, HigherOrderRecord record):
        reason = None if record.reason == RUNNING else _normalize_reason(record.reason)
        return HigherOrderStep(
            record.root, record.f_root, record.df_root, record.d2f_root,
            record.iters, (record.f_calls, record.df_calls, record.d2f_calls),
            record.precision, record.error, reason, record.callback_code,
            _stage_name(record.callback_stage))

    cdef object _copy_result(self):
        cdef HigherOrderRecord state = self._state
        return NewtonMethodsReturnType(
            state.root, state.f_root, (state.df_root, state.d2f_root), state.iters,
            (state.f_calls, state.df_calls, state.d2f_calls), state.precision,
            state.error, False, False,
            termination_reason=None if state.reason == RUNNING else state.reason)

    def step(self):
        """Attempt one finite candidate residual and copy its resulting state."""
        self._ready()
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
        """Use the remaining total budget through the common native step loop."""
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
            self._leave()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:
            return self._copy_result()
        finally:
            self._leave()

    @property
    def history(self):
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i]) for i in range(self._record_count)])
        finally:
            self._leave()

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
        return _stage_name(self._state.callback_stage)

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


cdef class Halley(_ThirdOrderNewton):
    """Resumable Halley method; None selects native numerical derivatives."""
    def __init__(self, function, derivative, second_derivative, x0,
                 f_x0=None, df_x0=None, d2f_x0=None, *,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024):
        self._setup(function, derivative, second_derivative, x0,
                    f_x0, df_x0, d2f_x0, etol, ertol, ptol, prtol,
                    max_iter, history, history_every, history_bytes, HALLEY, None)


cdef class Chebyshev(_ThirdOrderNewton):
    """Resumable Chebyshev method; None selects native numerical derivatives."""
    def __init__(self, function, derivative, second_derivative, x0,
                 f_x0=None, df_x0=None, d2f_x0=None, *,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024):
        self._setup(function, derivative, second_derivative, x0,
                    f_x0, df_x0, d2f_x0, etol, ertol, ptol, prtol,
                    max_iter, history, history_every, history_bytes, CHEBYSHEV, None)


cdef class ModifiedHalley(_ThirdOrderNewton):
    """Parameterized Halley family with a finite real alpha (default 0.5)."""
    def __init__(self, function, derivative, second_derivative, x0,
                 f_x0=None, df_x0=None, d2f_x0=None, *, alpha=0.5,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024):
        self._setup(function, derivative, second_derivative, x0,
                    f_x0, df_x0, d2f_x0, etol, ertol, ptol, prtol,
                    max_iter, history, history_every, history_bytes,
                    MODIFIED_HALLEY, alpha)


cdef class SuperHalley(_ThirdOrderNewton):
    """Super-Halley, the alpha=1 member of the modified Halley family."""
    def __init__(self, function, derivative, second_derivative, x0,
                 f_x0=None, df_x0=None, d2f_x0=None, *,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024):
        self._setup(function, derivative, second_derivative, x0,
                    f_x0, df_x0, d2f_x0, etol, ertol, ptol, prtol,
                    max_iter, history, history_every, history_bytes,
                    MODIFIED_HALLEY, 1.)


cdef class TangentHyperbolas(Halley):
    """Scalar tangent hyperbolas, sharing Halley's initializer and update."""


def halley(function, derivative, second_derivative, x0, **options):
    """Run Halley; retain a Halley instance to inspect copied steps."""
    return Halley(function, derivative, second_derivative, x0, **options).run()


def chebyshev(function, derivative, second_derivative, x0, **options):
    """Run Chebyshev; retain an instance to inspect copied steps."""
    return Chebyshev(function, derivative, second_derivative, x0, **options).run()


def modified_halley(function, derivative, second_derivative, x0, *, alpha=0.5, **options):
    """Run the modified Halley family with a finite real alpha."""
    return ModifiedHalley(function, derivative, second_derivative, x0,
                          alpha=alpha, **options).run()


def super_halley(function, derivative, second_derivative, x0, **options):
    """Run Super-Halley; retain an instance to inspect copied steps."""
    return SuperHalley(function, derivative, second_derivative, x0, **options).run()


def tangent_hyperbolas(function, derivative, second_derivative, x0, **options):
    """Run scalar tangent hyperbolas through the Halley update engine."""
    return TangentHyperbolas(function, derivative, second_derivative, x0, **options).run()
