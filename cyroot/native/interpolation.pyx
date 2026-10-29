# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Resumable interpolation methods with native callback and workspace paths."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.math cimport fabs, fmax, isfinite, NAN, INFINITY
from cyroot._check_args import _check_stop_cond_args
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, complex_abs, residual_met,
    RUNNING, RESIDUAL_TOLERANCE, MAX_ITERATIONS, BREAKDOWN, NONFINITE,
    CALLBACK_FAILURE, ZERO_DERIVATIVE,
)
from cyroot.ops.scalar_ops cimport csqrt
from cyroot.return_types cimport RootReturnType
from cyroot.return_types import _normalize_reason
from .scalar import NativeCallbackError, _nonnegative_integer

cdef enum:
    SIDI = 0
    STEFFENSEN = 1
    INVERSE_QUADRATIC = 2
    HYPERBOLIC = 3
    MULLER = 4


class InterpolationStep(NamedTuple):
    """Copied immutable state; termination_reason is None while running."""
    root: object
    f_root: object
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int


cdef class _Interpolation:
    def __init__(self, function, points, values, int method, aitken=True,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024,
                 workspace_bytes=8 * 1024 * 1024):
        cdef Py_ssize_t i, j, n
        cdef double f_atol, f_rtol, x_atol, x_rtol
        if self._constructing or self._callback is not None or self._complex_callback is not None or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        if method < SIDI or method > MULLER:
            raise ValueError('Unknown interpolation method.')
        self._constructing = True
        self._method = method
        try:
            if method == MULLER:
                self._complex_callback = function if isinstance(function, ComplexScalarCallback) else ComplexScalarCallback(function)
            else:
                self._callback = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
            self._enter()
            try:
                # Reserve the adapter before iterating/converting user data.
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                limit = _nonnegative_integer(history, 'history')
                every = _nonnegative_integer(history_every, 'history_every')
                byte_limit = _nonnegative_integer(history_bytes, 'history_bytes')
                work_limit = _nonnegative_integer(workspace_bytes, 'workspace_bytes')
                if every == 0:
                    raise ValueError('history_every must be positive.')
                if limit > byte_limit // sizeof(InterpolationRecord):
                    raise ValueError('Requested history exceeds history_bytes.')
                points = tuple(points)
                n = len(points)
                if (method == SIDI and n < 2) or (method == STEFFENSEN and n != 1) or (method >= INVERSE_QUADRATIC and n != 3):
                    raise ValueError('Invalid number of initial interpolation points.')
                if <size_t>n > work_limit // (3 * sizeof(double complex) + sizeof(unsigned char)):
                    raise ValueError('Interpolation workspace exceeds workspace_bytes.')
                values = (None,) * n if values is None else tuple(values)
                if len(values) != n:
                    raise ValueError('Initial points and values must have the same length.')
                self._xs.resize(n)
                self._fs.resize(n)
                self._work.resize(n)
                self._cached.resize(n)
                self._records.resize(limit)
                self._aitken = bool(aitken)
                for i in range(n):
                    if method == MULLER:
                        self._xs[i] = complex(points[i])
                        self._fs[i] = NAN if values[i] is None else complex(values[i])
                    else:
                        self._xs[i] = float(points[i])
                        self._fs[i] = NAN if values[i] is None else float(values[i])
                    self._cached[i] = values[i] is not None
                    for j in range(i):
                        if self._xs[i] == self._xs[j]:
                            raise ValueError('Initial interpolation points must be distinct.')
                self._max_iter = max_iter
                self._history_every = every
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                self._state.iters = self._state.f_calls = 0
                self._state.reason = RUNNING
                self._state.callback_code = 0
                self._state.root = self._xs[0]
                self._state.f_root = NAN
                self._state.error = INFINITY
                self._state.precision = INFINITY
                init_stop(&self._stop, 0., 0., f_atol, f_rtol, x_atol, x_rtol)
                with nogil:
                    self._initialize()
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
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if self._method == MULLER:
            if self._complex_callback is None or self._complex_callback._function == NULL:
                raise RuntimeError('Solver requires an initialized callback adapter.')
            if self._complex_callback._active:
                raise RuntimeError('This callback adapter is already active.')
            self._complex_callback._active = True
            self._complex_callback._error = None
        else:
            if self._callback is None or self._callback._function == NULL:
                raise RuntimeError('Solver requires an initialized callback adapter.')
            if self._callback._active:
                raise RuntimeError('This callback adapter is already active.')
            self._callback._active = True
            self._callback._error = None
        self._busy = True

    cdef void _release(self) noexcept:
        if self._method == MULLER:
            self._complex_callback._active = False
        else:
            self._callback._active = False
        self._busy = False

    cdef int _evaluate(self, double complex x, double complex* out) noexcept nogil:
        cdef double real = NAN, imag = NAN
        cdef int code
        self._state.f_calls += 1
        if self._method == MULLER:
            code = self._complex_callback._function(x.real, x.imag, &real, &imag,
                                                     self._complex_callback._context)
        else:
            imag = 0.
            code = self._callback._function(x.real, &real, self._callback._context)
        out[0] = real + imag * 1j
        if code != 0:
            self._state.callback_code = code
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self) noexcept nogil:
        cdef Py_ssize_t i, j, n = self._xs.size()
        cdef double error, xscale = 0.
        cdef double complex x, value
        for i in range(n):
            if not isfinite(complex_abs(self._xs[i])):
                self._stop.reason = NONFINITE
                return
            value = self._fs[i]
            if not self._cached[i] and self._evaluate(self._xs[i], &value) != 0:
                return
            if not isfinite(complex_abs(value)):
                self._stop.reason = NONFINITE
                return
            self._fs[i] = value
            error = complex_abs(value)
            self._stop.f_scale = fmax(self._stop.f_scale, error)
            xscale = fmax(xscale, complex_abs(self._xs[i]))
            if error < self._state.error:
                self._state.root, self._state.f_root = self._xs[i], value
                self._state.error = error
            # Exact/accepted roots can return without evaluating unused starts.
            if residual_met(&self._stop, self._state.error):
                self._stop.reason = RESIDUAL_TOLERANCE
                return
        self._stop.x_scale = self._stop.previous_x_norm = xscale
        if self._method == SIDI:
            # Descending initial errors put the best point last, like Sidi's
            # original generalized secant formulation. Stable insertion sort
            # avoids allocations in the native initialization path.
            for i in range(1, n):
                x, value = self._xs[i], self._fs[i]
                j = i
                while j > 0 and complex_abs(self._fs[j - 1]) < complex_abs(value):
                    self._xs[j], self._fs[j] = self._xs[j - 1], self._fs[j - 1]
                    j -= 1
                self._xs[j], self._fs[j] = x, value
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

    cdef bint _candidate(self, double complex* out) noexcept nogil:
        cdef Py_ssize_t i, j, n = self._xs.size()
        cdef double scale = 0., x0, x1, x2, f0, f1, f2, d01, d02, d12, denominator
        cdef double p, derivative, difference
        cdef double complex c01, c02, c12, a, b, delta, denominator1, denominator2
        cdef double complex probe, value
        for i in range(n):
            scale = fmax(scale, complex_abs(self._fs[i]))
        # A zero residual scale would already have terminated at initialization
        # or the preceding accepted step.
        if scale == 0.:
            self._stop.reason = BREAKDOWN
            return False
        if self._method == SIDI:
            for i in range(n):
                self._work[i] = self._fs[i] / scale
            # Newton divided differences in O(n) owned workspace.
            for j in range(1, n):
                for i in range(n - 1, j - 1, -1):
                    difference = (self._xs[i] - self._xs[i - j]).real
                    if not isfinite(difference):
                        self._stop.reason = NONFINITE
                        return False
                    if difference == 0.:
                        self._stop.reason = BREAKDOWN
                        return False
                    self._work[i] = (self._work[i] - self._work[i - 1]) / difference
                    if not isfinite(self._work[i].real):
                        self._stop.reason = NONFINITE
                        return False
            p, derivative = self._work[n - 1].real, 0.
            for i in range(n - 2, -1, -1):
                difference = (self._xs[n - 1] - self._xs[i]).real
                derivative = derivative * difference + p
                p = p * difference + self._work[i].real
            if not isfinite(derivative):
                self._stop.reason = NONFINITE
                return False
            if derivative == 0.:
                self._stop.reason = ZERO_DERIVATIVE
                return False
            out[0] = self._xs[n - 1].real - (self._fs[n - 1].real / scale) / derivative
        elif self._method == STEFFENSEN:
            x0, f0 = self._xs[0].real, self._fs[0].real
            probe = x0 + f0
            if not isfinite(probe.real):
                self._stop.reason = NONFINITE
                return False
            if self._evaluate(probe, &value) != 0:
                return False
            if not isfinite(value.real):
                self._stop.reason = NONFINITE
                return False
            f1 = value.real
            denominator = f1 - f0
            if not isfinite(denominator):
                self._stop.reason = NONFINITE
                return False
            if denominator == 0.:
                self._stop.reason = BREAKDOWN
                return False
            if self._aitken:
                out[0] = x0 - f0 * (f0 / denominator)
            else:
                out[0] = probe.real + f1 - f1 * (f1 / denominator)
        elif self._method == INVERSE_QUADRATIC or self._method == HYPERBOLIC:
            x0, x1, x2 = self._xs[0].real, self._xs[1].real, self._xs[2].real
            if not isfinite(x0 - x1) or not isfinite(x0 - x2) or not isfinite(x1 - x2):
                self._stop.reason = NONFINITE
                return False
            f0, f1, f2 = self._fs[0].real / scale, self._fs[1].real / scale, self._fs[2].real / scale
            d01, d02, d12 = f0 - f1, f0 - f2, f1 - f2
            if self._method == INVERSE_QUADRATIC:
                if d01 == 0. or d02 == 0. or d12 == 0.:
                    self._stop.reason = BREAKDOWN
                    return False
                # Offset form avoids summing three large absolute abscissas.
                out[0] = x2 + (x0 - x2) * (f1 / d01) * (f2 / d02) - (x1 - x2) * (f0 / d01) * (f2 / d12)
            else:
                if x0 == x1 or x1 == x2:
                    self._stop.reason = BREAKDOWN
                    return False
                denominator = f0 * (d12 / (x1 - x2)) - f2 * (d01 / (x0 - x1))
                if not isfinite(denominator):
                    self._stop.reason = NONFINITE
                    return False
                if denominator == 0.:
                    self._stop.reason = BREAKDOWN
                    return False
                out[0] = x1 - f1 * (d02 / denominator)
        else:
            c01 = self._xs[0] - self._xs[1]
            c02 = self._xs[0] - self._xs[2]
            c12 = self._xs[1] - self._xs[2]
            if not isfinite(complex_abs(c01)) or not isfinite(complex_abs(c02)) or not isfinite(complex_abs(c12)):
                self._stop.reason = NONFINITE
                return False
            if c01 == 0. or c02 == 0. or c12 == 0.:
                self._stop.reason = BREAKDOWN
                return False
            c01 = (self._fs[0] / scale - self._fs[1] / scale) / c01
            c02 = (self._fs[0] / scale - self._fs[2] / scale) / c02
            c12 = (self._fs[1] / scale - self._fs[2] / scale) / c12
            b = c12 + c02 - c01
            a = (c01 - c12) / (self._xs[0] - self._xs[2])
            delta = csqrt(b * b - 4 * a * (self._fs[2] / scale))
            denominator1, denominator2 = b + delta, b - delta
            if complex_abs(denominator1) < complex_abs(denominator2):
                denominator1 = denominator2
            if not isfinite(complex_abs(denominator1)):
                self._stop.reason = NONFINITE
                return False
            if denominator1 == 0.:
                self._stop.reason = BREAKDOWN
                return False
            out[0] = self._xs[2] - 2 * (self._fs[2] / scale) / denominator1
        if not isfinite(complex_abs(out[0])):
            self._stop.reason = NONFINITE
            return False
        return True

    cdef void _step(self) noexcept nogil:
        cdef double complex candidate, value
        cdef double precision
        cdef size_t i, n = self._xs.size()
        if self._stop.reason != RUNNING:
            return
        if not self._candidate(&candidate):
            self._record()
            return
        precision = complex_abs(candidate - self._xs[n - 1])
        if not isfinite(precision):
            self._stop.reason = NONFINITE
            self._record()
            return
        self._state.iters += 1
        if self._evaluate(candidate, &value) == 0:
            if not isfinite(complex_abs(value)):
                self._stop.reason = NONFINITE
            else:
                for i in range(1, n):
                    self._xs[i - 1], self._fs[i - 1] = self._xs[i], self._fs[i]
                self._xs[n - 1], self._fs[n - 1] = candidate, value
                self._state.root, self._state.f_root = candidate, value
                self._state.precision, self._state.error = precision, complex_abs(value)
                advance_x(&self._stop, complex_abs(candidate))
                self._check()
        self._record()

    cdef void _raise_callback_error(self) except *:
        error = self._complex_callback._error if self._method == MULLER else self._callback._error
        if error is not None:
            if self._method == MULLER:
                self._complex_callback._error = None
            else:
                self._callback._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result(), 'function')

    cdef object _snapshot(self, InterpolationRecord state):
        cdef object root, value
        reason = None if state.reason == RUNNING else _normalize_reason(state.reason)
        if self._method == MULLER:
            root, value = state.root, state.f_root
        else:
            root, value = state.root.real, state.f_root.real
        return InterpolationStep(root, value, state.iters, state.f_calls,
                                 state.precision, state.error, reason, state.callback_code)

    cdef object _copy_result(self):
        cdef InterpolationRecord state = self._state
        cdef object root, value
        if self._method == MULLER:
            root, value = state.root, state.f_root
        else:
            root, value = state.root.real, state.f_root.real
        return RootReturnType(root, value, state.iters, state.f_calls,
                              state.precision, state.error, False, False,
                              termination_reason=None if state.reason == RUNNING else state.reason)

    def step(self):
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


class Sidi(_Interpolation):
    """Generalized secant using the derivative of a Newton interpolant.

    Initial points and values are copied into owned native workspace. History
    follows the common bounded native solver contract. No Python or allocation
    is used in the numerical loop with a native ScalarCallback.
    """
    def __init__(self, function, x0s, f_x0s=None, **options):
        super().__init__(function, x0s, f_x0s, SIDI, **options)


class Steffensen(_Interpolation):
    """Steffensen iteration, optionally using Aitken's delta-squared form."""
    def __init__(self, function, x0, f_x0=None, *, aitken=True, **options):
        super().__init__(function, (x0,), (f_x0,), STEFFENSEN, aitken=aitken, **options)


class InverseQuadraticInterpolation(_Interpolation):
    """Three-point inverse quadratic interpolation without a bracket."""
    def __init__(self, function, x0, x1, x2, f_x0=None, f_x1=None, f_x2=None, **options):
        super().__init__(function, (x0, x1, x2), (f_x0, f_x1, f_x2), INVERSE_QUADRATIC, **options)


class HyperbolicInterpolation(_Interpolation):
    """Three-point hyperbolic interpolation without a bracket."""
    def __init__(self, function, x0, x1, x2, f_x0=None, f_x1=None, f_x2=None, **options):
        super().__init__(function, (x0, x1, x2), (f_x0, f_x1, f_x2), HYPERBOLIC, **options)


class Muller(_Interpolation):
    """Complex Muller iteration; accepts ComplexScalarCallback or Python callable."""
    def __init__(self, function, x0, x1, x2, f_x0=None, f_x1=None, f_x2=None, **options):
        super().__init__(function, (x0, x1, x2), (f_x0, f_x1, f_x2), MULLER, **options)


def sidi(function, x0s, **options):
    return Sidi(function, x0s, **options).run()


def steffensen(function, x0, **options):
    return Steffensen(function, x0, **options).run()


def inverse_quadratic_interp(function, x0, x1, x2, **options):
    return InverseQuadraticInterpolation(function, x0, x1, x2, **options).run()


def hyperbolic_interp(function, x0, x1, x2, **options):
    return HyperbolicInterpolation(function, x0, x1, x2, **options).run()


def muller(function, x0, x1, x2, **options):
    return Muller(function, x0, x1, x2, **options).run()
