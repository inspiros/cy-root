# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Resumable Newton with native Armijo backtracking on 1/2 ||F||_2^2."""
from libc.math cimport fabs, fmax, isfinite, NAN, INFINITY
from libc.string cimport memcpy
from .scalar import _nonnegative_integer
from .linalg cimport square_solve
from cyroot._stopping cimport (
    RUNNING, BREAKDOWN, NONFINITE, ZERO_DERIVATIVE, STAGNATION,
    LINE_SEARCH_FAILURE, advance_x,
)

cdef extern from "damped_newton.hpp" namespace "cyroot::native" nogil:
    int newton_merit_slope(const double*, const double*, const double*, size_t,
                          size_t, double, double*) noexcept
    int bfgs_change(const double*, const double*, size_t, double, double*) noexcept


def _line_controls(options):
    # Invoked only after the base solver has reserved itself and its adapters.
    c1, contraction, limit = options
    c1, contraction = float(c1), float(contraction)
    limit = _nonnegative_integer(limit, 'max_line_search')
    if not isfinite(c1) or not 0. < c1 < 1.:
        raise ValueError('c1 must be finite and strictly between zero and one.')
    if not isfinite(contraction) or not 0. < contraction < 1.:
        raise ValueError('contraction must be finite and strictly between zero and one.')
    if limit == 0:
        raise ValueError('max_line_search must be positive.')
    return c1, contraction, limit


cdef double _norm(const double* values, size_t count) noexcept nogil:
    cdef size_t i
    cdef double result = 0.
    for i in range(count):
        if not isfinite(values[i]):
            return INFINITY
        result = fmax(result, fabs(values[i]))
    return result


cdef void _clear(double* values, size_t count) noexcept nogil:
    cdef size_t i
    for i in range(count):
        values[i] = NAN


cdef class DampedNewton:
    """Scalar Newton with bounded Armijo backtracking.

    Uses Newton's Python/native callbacks, numerical derivative plans, copied
    history and total max_iter budget. Each iteration tries at most
    max_line_search residual evaluations, starting with the full Newton step.
    Nonfinite trial points/residuals are rejected; callback exceptions terminate
    and propagate. Only accepted point/residual pairs replace the current state.
    History contains accepted states and terminal diagnostics, not trial points.
    step_scale and backtracks describe the most recent line search.
    """
    def __init__(self, function, derivative, x0, f_x0=None, df_x0=None, *,
                 c1=1e-4, contraction=.5, max_line_search=32, **options):
        if self._line_options is not None or self._constructing or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._line_options = (c1, contraction, max_line_search)
        Newton.__init__(self, function, derivative, x0, f_x0=f_x0,
                        df_x0=df_x0, **options)

    cdef void _enter(self) except *:
        Newton._enter(self)
        try:
            if self._line_options is None:
                raise RuntimeError('Damped Newton options are not initialized.')
            if self._line_options:
                self._c1, self._contraction, self._max_line_search = _line_controls(self._line_options)
                self._line_options = ()
        except BaseException:
            Newton._leave(self)
            raise

    cdef void _step(self) noexcept nogil:
        cdef double delta, candidate, precision, value, change, slope
        cdef double alpha = 1.
        cdef size_t attempt
        if self._stop.reason != RUNNING:
            return
        self._step_scale, self._backtracks = 0., 0
        delta = -self._state.f_root / self._state.df_root
        if not isfinite(delta) or newton_merit_slope(
                &self._state.f_root, &self._state.df_root, &delta, 1, 1,
                self._state.error, &slope) != 0:
            self._stop.reason = NONFINITE
        elif slope >= 0.:
            self._stop.reason = LINE_SEARCH_FAILURE
        else:
            self._state.iters += 1
            for attempt in range(self._max_line_search):
                candidate = self._state.root + alpha*delta
                precision = fabs(candidate-self._state.root)
                if isfinite(candidate) and isfinite(precision):
                    if precision == 0.:
                        self._state.precision = 0.
                        self._stop.reason = STAGNATION
                        break
                    if self._evaluate(False, candidate, &value) != 0:
                        break
                    if (isfinite(value) and bfgs_change(&self._state.f_root,
                            &value, 1, self._state.error, &change) == 0 and
                            change < 0. and change <= self._c1*alpha*slope):
                        self._step_scale = alpha
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
                                    if value == 0.:
                                        self._stop.reason = ZERO_DERIVATIVE
                        break
                self._backtracks += 1
                alpha *= self._contraction
            else:
                self._stop.reason = LINE_SEARCH_FAILURE
        self._record()

    @property
    def step_scale(self):
        if self._busy or not self._initialized:
            raise RuntimeError('Solver is running or not initialized.')
        return self._step_scale

    @property
    def backtracks(self):
        if self._busy or not self._initialized:
            raise RuntimeError('Solver is running or not initialized.')
        return self._backtracks


cdef class VectorDampedNewton:
    """Vector Newton with the scalar Armijo policy and inferred system shape.

    Square systems use the existing LU backend. Rectangular systems use its
    rank-aware least-squares backend. Acceptance uses the actual slope F^T J p,
    not the square-system identity Jp=-F. A non-descent direction or exhausted
    search reports line_search_failure, never root success. Workspace, history,
    callbacks and numerical Jacobian plans follow VectorNewton's contract.
    """
    def __init__(self, function, jacobian, x0, f_x0=None, jac_x0=None, *,
                 c1=1e-4, contraction=.5, max_line_search=32, **options):
        if self._line_options is not None or self._constructing or self._initialized:
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._line_options = (c1, contraction, max_line_search)
        VectorNewton.__init__(self, function, jacobian, x0, f_x0=f_x0,
                              jac_x0=jac_x0, **options)

    cdef void _enter(self) except *:
        VectorNewton._enter(self)
        try:
            if self._line_options is None:
                raise RuntimeError('Damped Newton options are not initialized.')
            if self._line_options:
                self._c1, self._contraction, self._max_line_search = _line_controls(self._line_options)
                self._line_options = ()
        except BaseException:
            VectorNewton._leave(self)
            raise

    cdef void _step(self) noexcept nogil:
        cdef size_t i, attempt
        cdef int status
        cdef double precision, difference, change, slope
        cdef double alpha = 1.
        cdef bint finite_trial
        if self._stop.reason != RUNNING:
            return
        self._step_scale, self._backtracks = 0., 0
        if self._linear_plan is None:
            memcpy(self._lu, self._jac, self._square*sizeof(double))
            for i in range(self._n):
                self._delta[i] = -self._f[i]
            self._state.linear_status = square_solve(
                <int>self._n, self._lu, self._delta, &self._pivots[0])
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
            elif not isfinite(_norm(self._lu, self._square)):
                self._stop.reason = NONFINITE
        else:
            for i in range(self._m):
                self._rhs[i] = -self._f[i]
            status = self._linear_plan.compute(
                self._jac, self._rhs, self._delta, self._linear_work,
                &self._pivots[0], &self._state.linear_rank, &self._state.linear_status)
            if status != 0:
                self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
        if self._stop.reason != RUNNING:
            self._record()
            return
        if not isfinite(_norm(self._delta, self._n)) or newton_merit_slope(
                self._f, self._jac, self._delta, self._n, self._m,
                self._state.error, &slope) != 0:
            self._stop.reason = NONFINITE
        elif slope >= 0.:
            self._stop.reason = LINE_SEARCH_FAILURE
        else:
            self._state.iters += 1
            for attempt in range(self._max_line_search):
                precision, finite_trial = 0., True
                for i in range(self._n):
                    self._trial[i] = self._x[i] + alpha*self._delta[i]
                    difference = fabs(self._trial[i]-self._x[i])
                    if not isfinite(self._trial[i]) or not isfinite(difference):
                        finite_trial = False
                    precision = fmax(precision, difference)
                if finite_trial:
                    if precision == 0.:
                        self._state.precision = 0.
                        self._stop.reason = STAGNATION
                        break
                    if self._evaluate(False, self._trial, self._trial_f) != 0:
                        break
                    if (isfinite(_norm(self._trial_f, self._m)) and bfgs_change(
                            self._f, self._trial_f, self._m, self._state.error,
                            &change) == 0 and change < 0. and
                            change <= self._c1*alpha*slope):
                        self._step_scale = alpha
                        memcpy(self._x, self._trial, self._n*sizeof(double))
                        memcpy(self._f, self._trial_f, self._m*sizeof(double))
                        _clear(self._jac, self._square)
                        self._state.error = _norm(self._f, self._m)
                        self._state.precision = precision
                        advance_x(&self._stop, _norm(self._x, self._n))
                        self._check()
                        if self._stop.reason == RUNNING:
                            if self._evaluate(True, self._x, self._jac) != 0:
                                _clear(self._jac, self._square)
                            elif not isfinite(_norm(self._jac, self._square)):
                                _clear(self._jac, self._square)
                                self._stop.reason = NONFINITE
                        break
                self._backtracks += 1
                alpha *= self._contraction
            else:
                self._stop.reason = LINE_SEARCH_FAILURE
        self._record()

    @property
    def step_scale(self):
        self._ready()
        return self._step_scale

    @property
    def backtracks(self):
        self._ready()
        return self._backtracks


def damped_newton(function, derivative, x0, **options):
    """Solve a scalar equation with Armijo-backtracked Newton."""
    return DampedNewton(function, derivative, x0, **options).run()


def vector_damped_newton(function, jacobian, x0, **options):
    """Solve a nonlinear system with Armijo-backtracked Newton."""
    return VectorDampedNewton(function, jacobian, x0, **options).run()
