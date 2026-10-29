# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Shared residual-only state engine for Broyden and Klement updates."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport fabs, fmax, isfinite, NAN, INFINITY
from libc.string cimport memcpy, memmove, memset
from .buffers cimport OwnedDoubleBuffer, VectorView, MatrixView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .vector cimport VectorCallback
from .derivatives cimport (
    JacobianPlan, ResidualFunction, DIFF_CALLBACK_FAILURE, DIFF_NONFINITE, DIFF_BREAKDOWN,
)
from .hessians cimport HessianPlan
from .tensor_callbacks cimport HessianCallback
from .linalg cimport square_factor, square_solve_factored
from .tensor_ops cimport matrix_vector
from .broyden_ops cimport broyden_update, klement_update
from .barnes_ops cimport barnes_direction, barnes_update
from .scalar import NativeCallbackError
from .higher_derivatives import _checked_integer
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types import _normalize_reason
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, RUNNING, MAX_ITERATIONS,
    BREAKDOWN, NONFINITE, CALLBACK_FAILURE, RESIDUAL_TOLERANCE,
)


class BroydenResult(NamedTuple):
    """Copied secant approximation and its last updated accepted iteration."""
    root: tuple
    f_root: tuple
    approximation: tuple
    approximation_kind: str
    approximation_iteration: int
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int
    callback_stage: object
    linear_status: int
    phase: object
    approximation_valid: bool
    initialization_calls: int

    @property
    def success(self):
        return self.termination_reason == 'residual_tolerance'

    @property
    def converged(self):
        return self.success

    @property
    def optimal(self):
        return self.success


BroydenStep = BroydenResult


cdef void _fill(double* data, size_t n, double value) noexcept nogil:
    cdef size_t i
    for i in range(n):data[i] = value


cdef double _norm(const double* data, size_t n) noexcept nogil:
    cdef size_t i
    cdef double value = 0.
    for i in range(n):
        if not isfinite(data[i]):return INFINITY
        value = fmax(value, fabs(data[i]))
    return value


cdef void _pack_vector(object source, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != n:raise ValueError('Vector length must match the solver dimension.')
        for i in range(n):out[i] = view.get(i)
    else:
        if len(source) != <Py_ssize_t>n:raise ValueError('Vector length must match the solver dimension.')
        for i in range(n):out[i] = source[i]


cdef void _pack_matrix(object source, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef MatrixView view
    cdef size_t i, j
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a square numeric initial approximation.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.matrix_view()
        if view.rows() != n or view.columns() != n:
            raise ValueError('Initial approximation must be square with the solver dimension.')
        for j in range(n):
            for i in range(n):out[i+j*n] = view.get(i, j)
    else:
        if len(source) != <Py_ssize_t>n:
            raise ValueError('Initial approximation must be square with the solver dimension.')
        for i in range(n):
            row = source[i]
            if isinstance(row, (str, bytes, bytearray)) or len(row) != <Py_ssize_t>n:
                raise ValueError('Initial approximation must be square with the solver dimension.')
            for j in range(n):out[i+j*n] = row[j]


cdef object _phase(int value):
    if value == 1:return 'linear'
    if value == 2:return 'trial'
    if value == 3:return 'update'
    if value == 4:return 'initial_jacobian'
    if value == 5:return 'initial_inverse'
    return None


cdef class BroydenGood:
    """Direct good Broyden, with an initial Jacobian matrix or native plan."""
    def __init__(self, function, jacobian0, x0, f_x0=None, *, etol=ETOL,
                 ertol=ERTOL, ptol=PTOL,
                 prtol=PRTOL, max_iter=MAX_ITER, history=0,
                 history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024):
        self._setup(function, jacobian0, x0, f_x0, etol, ertol, ptol, prtol,
                    max_iter, history, history_every, history_bytes, workspace_bytes, False, None, False)

    cdef void _setup(self, object function, object approximation, object x0,
                     object f_x0, object etol, object ertol, object ptol,
                     object prtol, object max_iter, object history,
                     object history_every, object history_bytes,
                     object workspace_bytes, bint from_jacobian,
                     object algo, bint dispatch) except *:
        cdef DoubleBuffer guard
        cdef size_t n
        cdef double* data
        cdef bint numerical
        if (self._constructing or self._initialized or self._storage != NULL or
                self._function is not None or self._plan is not None):
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            self._inverse = isinstance(self, BroydenBad)
            self._klement = isinstance(self, Klement)
            self._barnes = isinstance(self, Barnes)
            if ((self._inverse and self._klement) or
                    (self._barnes and (self._inverse or self._klement))):
                raise TypeError('Barnes, Klement and inverse Broyden cannot be combined by inheritance.')
            if isinstance(approximation, (VectorCallback, HessianPlan, HessianCallback)):
                raise TypeError('Initial approximation requires a matrix or JacobianPlan.')
            if isinstance(approximation, JacobianPlan):
                self._plan = approximation
                if not self._plan._initialized:
                    raise RuntimeError('Jacobian plan is not initialized.')
            numerical = approximation is None or self._plan is not None
            # Only native extension-type checks occur before reservations.
            if isinstance(function, (JacobianPlan, HessianPlan, HessianCallback)):
                raise TypeError('function requires a residual callback.')
            if isinstance(function, VectorCallback):
                self._function = function
                if self._function._function == NULL:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._function._jacobian:
                    raise ValueError('function requires a residual-kind callback.')
            self._enter()
            try:
                if dispatch:
                    # Exact builtin types prevent arbitrary equality/index hooks.
                    # Mode selection and all subsequent coercion share one reservation.
                    if type(algo) is int:
                        if algo != 1 and algo != 2:
                            raise ValueError("algo must be 1/'good' or 2/'bad'.")
                        self._inverse = algo == 2
                    elif type(algo) is str:
                        if algo != 'good' and algo != 'bad':
                            raise ValueError("algo must be 1/'good' or 2/'bad'.")
                        self._inverse = algo == 'bad'
                    else:
                        raise TypeError('algo requires an exact built-in int or str.')
                    from_jacobian = self._inverse
                self._from_jacobian = from_jacobian
                if self._inverse and not from_jacobian:
                    if numerical:
                        raise TypeError('inverse0 requires a matrix; use BroydenBad.from_jacobian for a plan.')
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(
                    etol, ertol, ptol, prtol, max_iter)
                limit = _checked_integer(history, 'history')
                every = _checked_integer(history_every, 'history_every')
                byte_limit = _checked_integer(history_bytes, 'history_bytes')
                workspace_limit = _checked_integer(workspace_bytes, 'workspace_bytes')
                if every == 0:raise ValueError('history_every must be positive.')
                if is_native_double_buffer(x0):
                    guard = DoubleBuffer(x0)
                    dimension = int(guard.vector_view().size())
                else:
                    if isinstance(x0, (str, bytes, bytearray)):
                        raise TypeError('Expected a numeric point vector.')
                    dimension = len(x0)
                if dimension == 0 or dimension > INT_MAX:
                    raise ValueError('Dimension must be positive and fit the LAPACK C int ABI.')
                if self._function is not None and self._function._dimension != dimension:
                    raise ValueError('Residual callback dimension must match the point.')
                # All products remain Python integers through capacity/byte checks.
                square = dimension*dimension
                workspace_count = 6*dimension+(3 if not self._inverse or from_jacobian else 2)*square
                if numerical:workspace_count += 5*dimension
                if self._barnes:workspace_count += 2*square
                plan_steps = 0 if self._plan is None else int(self._plan._step_count)
                if self._plan is not None and self._plan._dimension != dimension:
                    raise ValueError('Jacobian plan dimension must match the solver.')
                stride = 2*dimension+square
                pivot_count = dimension if not self._inverse or from_jacobian else 0
                record_bytes = sizeof(BroydenRecord)+sizeof(double)*stride
                if (workspace_count > OwnedDoubleBuffer.max_size() or
                        pivot_count > self._pivots.max_size() or
                        (workspace_count+plan_steps)*sizeof(double)+pivot_count*sizeof(int) > workspace_limit):
                    raise ValueError('Requested workspace exceeds workspace_bytes or native limits.')
                if (limit > self._records.max_size() or
                        limit*stride > OwnedDoubleBuffer.max_size() or
                        limit > byte_limit//record_bytes):
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                if numerical and self._plan is None:
                    self._plan = JacobianPlan(dimension)
                self._n = n = dimension
                self._square = square
                self._stride = stride
                self._max_iter = max_iter
                self._history_every = every
                if self._function is None:
                    self._function = VectorCallback(function, n)
                    self._function._active = True
                self._storage = new OwnedDoubleBuffer(workspace_count)
                data = self._storage.data()
                self._x = data
                self._f = data+n
                self._approx = data+2*n
                self._trial = self._approx+self._square
                self._trial_f = self._trial+n
                self._delta = self._trial_f+n
                self._change = self._delta+n
                if self._inverse and not from_jacobian:
                    self._lu = NULL
                    self._next = self._change+n
                else:
                    self._lu = self._change+n
                    self._next = self._lu+self._square
                data = self._next+self._square
                if numerical:
                    self._diff_work = data
                    data += 5*n
                if self._barnes:
                    self._directions = data
                    self._basis = self._directions+n*(n-1)
                    self._projected = self._basis+self._square
                self._history_data = new OwnedDoubleBuffer(limit*stride)
                self._pivots.resize(pivot_count)
                self._records.resize(limit)
                _fill(self._f, n, NAN)
                _pack_vector(x0, self._x, n)
                _fill(self._approx, self._square, NAN)
                if not numerical:
                    _pack_matrix(approximation, self._next if from_jacobian else self._approx, n)
                if f_x0 is not None:_pack_vector(f_x0, self._f, n)
                memset(&self._state, 0, sizeof(BroydenRecord))
                self._state.precision = INFINITY
                self._state.error = _norm(self._f, n)
                self._state.approximation_valid = (not numerical and not from_jacobian and
                                                   isfinite(_norm(self._approx, self._square)))
                init_stop(&self._stop, 0., _norm(self._x, n), etol, ertol, ptol, prtol)
                with nogil:
                    self._initialize(f_x0 is not None)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                self._leave()
        finally:
            self._constructing = False

    def __dealloc__(self):
        if self._storage != NULL:del self._storage
        if self._history_data != NULL:del self._history_data

    cdef void _ready(self) except *:
        if self._busy or self._constructing:raise RuntimeError('This solver is already active.')
        if not self._initialized:raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._busy:raise RuntimeError('This solver is already active.')
        if self._function is not None:
            if self._function._active:raise RuntimeError('This callback adapter is already active.')
            self._function._active = True
        self._busy = True
        if self._function is not None:self._function._error = None

    cdef void _leave(self) noexcept:
        if self._function is not None:self._function._active = False
        self._busy = False

    cdef int _evaluate(self, const double* point, double* out) noexcept nogil:
        cdef int code
        _fill(out, self._n, NAN)
        self._state.f_calls += 1
        code = self._function._function(point, self._n, out, self._function._context)
        if code != 0:
            self._state.callback_code = code
            self._state.callback_stage = 1
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _initialize(self, bint cached) noexcept nogil:
        if not isfinite(_norm(self._x, self._n)):
            self._stop.reason = NONFINITE
            return
        if not cached:
            if self._evaluate(self._x, self._trial_f) != 0:return
            if not isfinite(_norm(self._trial_f, self._n)):
                self._stop.reason = NONFINITE
                return
            memcpy(self._f, self._trial_f, self._n*sizeof(double))
        self._state.error = _norm(self._f, self._n)
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, INFINITY, False):return
        self._initial_approximation()

    cdef void _initial_approximation(self) noexcept nogil:
        cdef size_t i, j
        cdef int status, sample_code = 0
        if self._plan is not None:
            self._state.phase = 4
            self._state.initialization_calls += 1
            status = self._plan.compute(self._x, self._f, self._next, self._diff_work,
                <ResidualFunction>self._function._function, self._function._context,
                &self._state.f_calls, &sample_code)
            if status != 0:
                if status == DIFF_CALLBACK_FAILURE:
                    self._state.callback_code = sample_code
                    self._state.callback_stage = 2
                    self._stop.reason = CALLBACK_FAILURE
                else:
                    self._stop.reason = NONFINITE if status == DIFF_NONFINITE else BREAKDOWN
                return
        if self._from_jacobian:
            self._state.phase = 5
            self._state.linear_status = 0
            if not isfinite(_norm(self._next, self._square)):
                self._stop.reason = NONFINITE
                return
            memcpy(self._lu, self._next, self._square*sizeof(double))
            self._state.linear_status = square_factor(<int>self._n, self._lu, self._pivots.data())
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
                return
            if not isfinite(_norm(self._lu, self._square)):
                self._stop.reason = NONFINITE
                return
            for j in range(self._n):
                _fill(self._delta, self._n, 0.)
                self._delta[j] = 1.
                self._state.linear_status = square_solve_factored(<int>self._n, self._lu,
                                                                  self._delta, self._pivots.data())
                if self._state.linear_status != 0:
                    self._stop.reason = BREAKDOWN
                    return
                if not isfinite(_norm(self._delta, self._n)):
                    self._stop.reason = NONFINITE
                    return
                for i in range(self._n):self._next[i+j*self._n] = self._delta[i]
        if self._plan is not None or self._from_jacobian:
            if not isfinite(_norm(self._next, self._square)):
                self._stop.reason = NONFINITE
                return
            memcpy(self._approx, self._next, self._square*sizeof(double))
            self._state.approximation_valid = True
        elif not self._state.approximation_valid:
            self._stop.reason = NONFINITE

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop, self._state.error, self._state.precision,
                      self._state.iters > 0):return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS

    cdef void _record(self) noexcept nogil:
        cdef size_t slot
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:return
        if self._record_count > 0 and self._records[self._record_count-1].iters == self._state.iters:
            slot = self._record_count-1
        else:
            if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:return
            if self._record_count >= self._records.size():
                self._history_truncated = True
                return
            slot = self._record_count
            self._record_count += 1
        self._records[slot] = self._state
        memcpy(self._history_data.data()+slot*self._stride, self._x, self._stride*sizeof(double))

    cdef void _step(self) noexcept nogil:
        cdef size_t i
        cdef int status
        cdef double difference, precision = 0.
        if self._stop.reason != RUNNING:return
        self._state.phase = 1
        self._state.linear_status = 0
        if self._inverse:
            status = matrix_vector(self._approx, self._f, self._n, self._delta)
            if status != 0:
                self._stop.reason = BREAKDOWN if status == 2 else NONFINITE
                self._record()
                return
            for i in range(self._n):self._delta[i] = -self._delta[i]
        else:
            memcpy(self._lu, self._approx, self._square*sizeof(double))
            for i in range(self._n):self._delta[i] = -self._f[i]
            self._state.linear_status = square_factor(<int>self._n, self._lu, self._pivots.data())
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
                self._record()
                return
            if not isfinite(_norm(self._lu, self._square)):
                self._stop.reason = NONFINITE
                self._record()
                return
            self._state.linear_status = square_solve_factored(<int>self._n, self._lu,
                                                              self._delta, self._pivots.data())
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
                self._record()
                return
        for i in range(self._n):
            self._trial[i] = self._x[i]+self._delta[i]
            # Replace unrounded direction by the actual representable change.
            difference = self._trial[i]-self._x[i]
            if (not isfinite(self._delta[i]) or not isfinite(self._trial[i]) or
                    not isfinite(difference)):
                self._stop.reason = NONFINITE
                self._record()
                return
            self._delta[i] = difference
            precision = fmax(precision, fabs(difference))
        self._state.phase = 2
        self._state.iters += 1
        if self._evaluate(self._trial, self._trial_f) != 0:
            self._record()
            return
        if not isfinite(_norm(self._trial_f, self._n)):
            self._stop.reason = NONFINITE
            self._record()
            return
        # Retain old F until after commit/stopping, so an unused residual
        # difference need not be representable at a terminal accepted point.
        memcpy(self._change, self._f, self._n*sizeof(double))
        memcpy(self._x, self._trial, self._n*sizeof(double))
        memcpy(self._f, self._trial_f, self._n*sizeof(double))
        self._state.precision = precision
        self._state.error = _norm(self._f, self._n)
        advance_x(&self._stop, _norm(self._x, self._n))
        self._check()
        if self._stop.reason != RUNNING:
            self._record()
            return
        self._state.phase = 3
        for i in range(self._n):self._change[i] = self._f[i]-self._change[i]
        if not isfinite(_norm(self._change, self._n)):
            self._stop.reason = NONFINITE
            self._record()
            return
        if self._inverse:
            status = broyden_update(self._approx, self._change, self._delta, self._n, self._next)
        elif self._klement:
            status = klement_update(self._approx, self._delta, self._change, self._n, self._next)
        elif self._barnes:
            status = barnes_direction(self._directions, self._direction_count,
                self._delta, self._n, self._basis,
                self._basis+self._n*(self._n-1), self._projected)
            if status == 0:
                status = barnes_update(self._approx, self._delta, self._change,
                    self._projected, self._n, self._next)
        else:
            status = broyden_update(self._approx, self._delta, self._change, self._n, self._next)
        if status != 0:
            self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
            self._record()
            return
        memcpy(self._approx, self._next, self._square*sizeof(double))
        if self._barnes and self._n > 1:
            if self._direction_count == self._n-1:
                memmove(self._directions, self._directions+self._n,
                        (self._n-2)*self._n*sizeof(double))
                memcpy(self._directions+(self._n-2)*self._n, self._delta,
                       self._n*sizeof(double))
            else:
                memcpy(self._directions+self._direction_count*self._n,
                       self._delta, self._n*sizeof(double))
                self._direction_count += 1
        self._state.approximation_iteration = self._state.iters
        self._record()

    cdef void _raise_callback_error(self) except *:
        if self._function._error is not None:
            error = self._function._error
            self._function._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code,
                                      self._snapshot(self._state, self._x),
                                      'initial_jacobian' if self._state.callback_stage == 2 else 'function')

    cdef object _snapshot(self, BroydenRecord record, const double* data):
        cdef size_t i, j
        return BroydenResult(
            tuple([data[i] for i in range(self._n)]),
            tuple([data[self._n+i] for i in range(self._n)]),
            tuple([tuple([data[2*self._n+i+j*self._n] for j in range(self._n)])
                   for i in range(self._n)]),
            'inverse_jacobian' if self._inverse else 'jacobian',
            record.approximation_iteration, record.iters, record.f_calls,
            record.precision, record.error,
            None if record.reason == RUNNING else _normalize_reason(record.reason),
            record.callback_code,
            ('initial_jacobian' if record.callback_stage == 2 else 'function') if record.callback_code else None,
            record.linear_status, _phase(record.phase), bool(record.approximation_valid),
            record.initialization_calls)

    def step(self):
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:self._step()
                self._raise_callback_error()
            return self._snapshot(self._state, self._x)
        finally:self._leave()

    def run(self):
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:self._step()
                self._raise_callback_error()
            return self._snapshot(self._state, self._x)
        finally:self._leave()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:return self._snapshot(self._state, self._x)
        finally:self._leave()

    @property
    def history(self):
        cdef size_t i
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i], self._history_data.data()+i*self._stride)
                          for i in range(self._record_count)])
        finally:self._leave()

    @property
    def done(self):
        self._ready()
        return self._stop.reason != RUNNING

    @property
    def history_truncated(self):
        self._ready()
        return bool(self._history_truncated)

    @property
    def method(self):
        self._ready()
        if self._barnes:return 'barnes'
        if self._klement:return 'klement'
        return 'broyden_bad' if self._inverse else 'broyden_good'

    def __iter__(self):return self

    def __next__(self):
        if self.done:raise StopIteration
        return self.step()


cdef class BroydenBad(BroydenGood):
    """Inverse bad Broyden; inverse0 must already be an inverse approximation."""
    @classmethod
    def from_jacobian(cls, function, jacobian0, x0, f_x0=None, *, etol=ETOL,
                      ertol=ERTOL, ptol=PTOL,
                      prtol=PRTOL, max_iter=MAX_ITER, history=0,
                      history_every=1, history_bytes=8*1024*1024,
                      workspace_bytes=64*1024*1024):
        """Initialize an inverse from a matrix or native numerical Jacobian plan."""
        cdef BroydenGood solver = BroydenBad.__new__(cls)
        solver._setup(function, jacobian0, x0, f_x0, etol, ertol, ptol, prtol,
                      max_iter, history, history_every, history_bytes, workspace_bytes, True, None, False)
        return solver

    def __init__(self, function, inverse0, x0, f_x0=None, *, etol=ETOL,
                 ertol=ERTOL, ptol=PTOL,
                 prtol=PRTOL, max_iter=MAX_ITER, history=0,
                 history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024):
        self._setup(function, inverse0, x0, f_x0, etol, ertol, ptol, prtol,
                    max_iter, history, history_every, history_bytes, workspace_bytes, False, None, False)


def broyden_good(function, jacobian0, x0, **options):
    return BroydenGood(function, jacobian0, x0, **options).run()


cdef class Klement(BroydenGood):
    """Square Klement using Broyden's state with a row-weighted secant update.

    Constructor and result/history fields match BroydenGood. jacobian0 is a
    matrix, matching JacobianPlan, or None for automatic initialization.
    """
    pass


cdef class Barnes(BroydenGood):
    """Square Barnes with deterministic projection against recent secants.

    Constructor, initialization and result/history fields match BroydenGood.
    Keeps at most n-1 previous directions; dependent updates report breakdown.
    """
    @property
    def secant_history_size(self):
        self._ready()
        return self._direction_count


def barnes(function, jacobian0, x0, **options):
    """Run native Barnes; retain Barnes for copied intermediate history."""
    return Barnes(function, jacobian0, x0, **options).run()


def klement(function, jacobian0, x0, **options):
    """Run native Klement; retain Klement for step/history inspection."""
    return Klement(function, jacobian0, x0, **options).run()


def broyden_bad(function, inverse0, x0, **options):
    return BroydenBad(function, inverse0, x0, **options).run()


def _broyden(function, x0, f_x0, jacobian0, algo, etol, ertol, ptol, prtol,
             max_iter, history, history_every, history_bytes, workspace_bytes):
    """Private guarded dispatcher for the established public entry point."""
    cdef BroydenGood solver = BroydenGood.__new__(BroydenGood)
    solver._setup(function, jacobian0, x0, f_x0, etol, ertol, ptol, prtol,
                  max_iter, history, history_every, history_bytes, workspace_bytes,
                  False, algo, True)
    return solver.run()
