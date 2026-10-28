# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Shared vector Chebyshev and square matrix modified-Halley state engine."""
from typing import NamedTuple
from numbers import Complex, Real
from cpython.buffer cimport PyObject_CheckBuffer
from libc.limits cimport INT_MAX
from libc.math cimport fabs, fma, fmax, isfinite, NAN, INFINITY
from libc.string cimport memcpy, memset

from .buffers cimport VectorView, MatrixView
from ._buffers cimport DoubleBuffer
from .derivatives cimport DIFF_NONFINITE, DIFF_BREAKDOWN, DIFF_CALLBACK_FAILURE
from .hessians cimport compute_hessian, compute_rectangular_hessian
from .linalg cimport square_factor, square_solve_factored
from .tensor_ops cimport (
    contract_hessian, contract_rectangular_hessian,
    contract_rectangular_hessian_direction, negative_half_sum, rectangular_matrix_vector,
)
from .tensor_callbacks cimport _pack_tensor, _pack_rectangular_tensor
from .higher_derivatives import _checked_integer
from .scalar import NativeCallbackError
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types cimport NewtonMethodsReturnType
from cyroot.return_types import _normalize_reason
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, RUNNING, MAX_ITERATIONS,
    BREAKDOWN, NONFINITE, CALLBACK_FAILURE,
)


cdef enum:
    FUNCTION = 1
    JACOBIAN = 2
    HESSIAN = 3
    CHEBYSHEV = 0
    HALLEY = 1
    SUPER_HALLEY = 2
    MODIFIED_HALLEY = 3
    GENERAL_MODIFIED = 3


class VectorChebyshevStep(NamedTuple):
    """Copied immutable point, Jacobian/Hessian, counters and diagnostics."""
    root: tuple
    f_root: tuple
    df_root: tuple
    iters: int
    f_calls: tuple
    precision: float
    error: float
    termination_reason: object
    callback_code: int
    callback_stage: object
    linear_status: int
    linear_stage: object
    linear_rank: object


# All methods export the same copied state type. Keep the original
# spelling as an alias for consumers compiled against this preview API.
VectorHigherOrderStep = VectorChebyshevStep


cdef object _stage_name(int stage):
    if stage == FUNCTION:
        return 'function'
    if stage == JACOBIAN:
        return 'jacobian'
    if stage == HESSIAN:
        return 'hessian'
    return None


cdef object _linear_stage_name(int stage):
    if stage == 1:
        return 'jacobian'
    if stage == 2:
        return 'secondary'
    if stage == 3:
        return 'correction'
    return None


cdef void _fill(double* data, size_t count, double value) noexcept nogil:
    cdef size_t i
    for i in range(count):
        data[i] = value


cdef double _norm(const double* data, size_t count) noexcept nogil:
    cdef size_t i
    cdef double result = 0., value
    for i in range(count):
        value = data[i]
        if not isfinite(value):
            return INFINITY
        result = fmax(result, fabs(value))
    return result


cdef void _pack_vector(object source, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector.')
    if PyObject_CheckBuffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != n:
            raise ValueError('Vector length must match the solver dimension.')
        for i in range(n):
            out[i] = view.get(i)
    else:
        if len(source) != n:
            raise ValueError('Vector length must match the solver dimension.')
        for i in range(n):
            out[i] = source[i]


cdef void _pack_matrix(object source, double* out, size_t n, size_t m) except *:
    cdef DoubleBuffer guard
    cdef MatrixView view
    cdef size_t i, j
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric Jacobian matrix.')
    if PyObject_CheckBuffer(source):
        guard = DoubleBuffer(source)
        view = guard.matrix_view()
        if view.rows() != m or view.columns() != n:
            raise ValueError('Jacobian shape must match the solver dimension.')
        for j in range(n):
            for i in range(m):
                out[i+j*m] = view.get(i, j)
    else:
        if len(source) != m:
            raise ValueError('Jacobian shape must match the solver dimension.')
        for i in range(m):
            row = source[i]
            if isinstance(row, (str, bytes, bytearray)) or len(row) != n:
                raise ValueError('Jacobian shape must match the solver dimension.')
            for j in range(n):
                out[i+j*m] = row[j]


cdef object _vector_tuple(const double* data, size_t n):
    return tuple([data[i] for i in range(n)])


cdef object _matrix_tuple(const double* data, size_t n, size_t m):
    return tuple([tuple([data[i+j*m] for j in range(n)]) for i in range(m)])


cdef object _tensor_tuple(const double* data, size_t n, size_t m):
    return tuple([tuple([tuple([data[i+m*(j+n*k)] for k in range(n)])
                        for j in range(n)]) for i in range(m)])


cdef object _vector_length(object source):
    cdef DoubleBuffer guard
    if isinstance(source, (str, bytes, bytearray)):raise TypeError('Expected a numeric vector.')
    if PyObject_CheckBuffer(source):
        guard = DoubleBuffer(source)
        return int(guard.vector_view().size())
    return len(source)


cdef int _rectangular_residual(const double* x, size_t n, size_t m,
                              double* out, void* context) noexcept nogil:
    if (<VectorChebyshev><object>context)._rect_function_ptr != NULL:
        return (<VectorChebyshev><object>context)._rect_function_ptr(
            x, n, m, out, (<VectorChebyshev><object>context)._function_context)
    return (<VectorChebyshev><object>context)._function_ptr(
        x, n, out, (<VectorChebyshev><object>context)._function_context)


cdef int _square_residual(const double* x, size_t n, double* out,
                          void* context) noexcept nogil:
    return _rectangular_residual(x, n, (<VectorChebyshev><object>context)._m, out, context)


cdef class VectorChebyshev:
    """Shared resumable solver with analytic or numerical J and H.

    Square Chebyshev uses one Jacobian factorization and two solves;
    rectangles use rank-aware least-squares corrections. Matrix Halley
    variants form and solve a secondary system; general modified Halley also
    retains the Jacobian factors for its correction. None providers select
    native central plans. Derived classes select their native update formula.
    """

    def __init__(self, function, jacobian, hessian, x0,
                 f_x0=None, jac_x0=None, hess_x0=None, *, etol=1e-10,
                 ertol=8.881784197001252e-16, ptol=1e-12,
                 prtol=8.881784197001252e-16, max_iter=200, history=0,
                 history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024):
        self._setup(function, jacobian, hessian, x0, f_x0, jac_x0, hess_x0,
                    etol, ertol, ptol, prtol, max_iter, history, history_every,
                    history_bytes, workspace_bytes, None)

    cdef void _setup(self, object function, object jacobian, object hessian,
                      object x0, object f_x0, object jac_x0, object hess_x0,
                      object etol, object ertol, object ptol, object prtol,
                      object max_iter, object history, object history_every,
                      object history_bytes, object workspace_bytes,
                      object alpha_option) except *:
        cdef size_t n, m
        cdef double* data
        cdef OwnedDoubleBuffer* bootstrap = NULL
        cdef bint discovered = False
        cdef double f_atol, f_rtol, x_atol, x_rtol
        if (self._constructing or self._initialized or self._storage != NULL or
                self._function is not None or self._jacobian is not None or
                self._hessian is not None or self._jplan is not None or self._hplan is not None or
                self._rect_function is not None or self._rect_jacobian is not None or
                self._rect_hessian is not None or self._rect_jplan is not None or
                self._rect_hplan is not None or self._linear_plan is not None):
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            if isinstance(self, VectorModifiedHalley):self._method = MODIFIED_HALLEY
            elif isinstance(self, VectorSuperHalley):self._method = SUPER_HALLEY
            elif isinstance(self, VectorHalley):self._method = HALLEY
            else:self._method = CHEBYSHEV
            if isinstance(function, (JacobianPlan, RectangularJacobianPlan, HessianPlan,
                                     RectangularHessianPlan, HessianCallback, RectangularHessianCallback)):
                raise TypeError('function requires a residual callback.')
            if isinstance(jacobian, (HessianPlan, RectangularHessianPlan, HessianCallback, RectangularHessianCallback)):
                raise TypeError('jacobian requires a Jacobian provider.')
            if isinstance(hessian, (JacobianPlan, RectangularJacobianPlan, VectorCallback, RectangularCallback)):
                raise TypeError('hessian requires a Hessian provider.')
            if isinstance(function, VectorCallback):
                self._function = function
                if self._function._function == NULL or self._function._constructing:raise RuntimeError('Residual callback is not initialized.')
                if self._function._jacobian:raise ValueError('function requires a residual-kind callback.')
            elif isinstance(function, RectangularCallback):
                self._rect_function = function
                if self._rect_function._function == NULL or self._rect_function._constructing:raise RuntimeError('Residual callback is not initialized.')
                if self._rect_function._jacobian:raise ValueError('function requires a residual-kind callback.')
            if isinstance(jacobian, VectorCallback):
                self._jacobian = jacobian
                if self._jacobian._function == NULL or self._jacobian._constructing:raise RuntimeError('Jacobian callback is not initialized.')
                if not self._jacobian._jacobian:raise ValueError('jacobian requires a Jacobian-kind callback.')
            elif isinstance(jacobian, RectangularCallback):
                self._rect_jacobian = jacobian
                if self._rect_jacobian._function == NULL or self._rect_jacobian._constructing:raise RuntimeError('Jacobian callback is not initialized.')
                if not self._rect_jacobian._jacobian:raise ValueError('jacobian requires a Jacobian-kind callback.')
            elif isinstance(jacobian, JacobianPlan):
                self._jplan = jacobian
                if not self._jplan._initialized or self._jplan._constructing:raise RuntimeError('Jacobian plan is not initialized.')
            elif isinstance(jacobian, RectangularJacobianPlan):
                self._rect_jplan = jacobian
                if not self._rect_jplan._initialized or self._rect_jplan._constructing:raise RuntimeError('Jacobian plan is not initialized.')
            if isinstance(hessian, HessianCallback):
                self._hessian = hessian
                if self._hessian._function == NULL or self._hessian._constructing:raise RuntimeError('Hessian callback is not initialized.')
            elif isinstance(hessian, RectangularHessianCallback):
                self._rect_hessian = hessian
                if self._rect_hessian._function == NULL or self._rect_hessian._constructing:raise RuntimeError('Hessian callback is not initialized.')
            elif isinstance(hessian, HessianPlan):
                self._hplan = hessian
                if not self._hplan._initialized or self._hplan._constructing:raise RuntimeError('Hessian plan is not initialized.')
            elif isinstance(hessian, RectangularHessianPlan):
                self._rect_hplan = hessian
                if not self._rect_hplan._initialized or self._rect_hplan._constructing:raise RuntimeError('Hessian plan is not initialized.')
            self._enter()
            try:
                self._path = self._method
                if self._method == MODIFIED_HALLEY:
                    if (isinstance(alpha_option, (str, bytes, bytearray, bool)) or
                            (isinstance(alpha_option, Complex) and not isinstance(alpha_option, Real)) or
                            getattr(getattr(alpha_option, 'dtype', None), 'kind', None) in ('b', 'c')):
                        raise TypeError('alpha must be a finite real scalar.')
                    try:len(alpha_option)
                    except TypeError:pass
                    else:raise TypeError('alpha must be a finite real scalar.')
                    self._alpha = alpha_option
                    if not isfinite(self._alpha):raise ValueError('alpha must be finite.')
                    if self._alpha == 0.:self._path = CHEBYSHEV
                    elif self._alpha == .5:self._path = HALLEY
                    elif self._alpha == 1.:self._path = SUPER_HALLEY
                    else:self._path = GENERAL_MODIFIED
                workspace_limit = _checked_integer(workspace_bytes, 'workspace_bytes')
                limit = _checked_integer(history, 'history')
                every = _checked_integer(history_every, 'history_every')
                history_limit = _checked_integer(history_bytes, 'history_bytes')
                if every == 0:raise ValueError('history_every must be positive.')
                max_iter = 200 if max_iter is None else _checked_integer(max_iter, 'max_iter')
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)
                f_atol, f_rtol, x_atol, x_rtol = etol, ertol, ptol, prtol
                dimension = _vector_length(x0)
                if dimension == 0 or dimension > INT_MAX:raise ValueError('Dimension must be positive and fit LAPACK C int.')
                n = dimension
                shapes = []
                if self._function is not None:shapes.append((int(self._function._dimension), int(self._function._dimension)))
                if self._jacobian is not None:shapes.append((int(self._jacobian._dimension), int(self._jacobian._dimension)))
                if self._hessian is not None:shapes.append((int(self._hessian._dimension), int(self._hessian._dimension)))
                if self._jplan is not None:shapes.append((int(self._jplan._dimension), int(self._jplan._dimension)))
                if self._hplan is not None:shapes.append((int(self._hplan._dimension), int(self._hplan._dimension)))
                if self._rect_function is not None:shapes.append((int(self._rect_function._n), int(self._rect_function._m)))
                if self._rect_jacobian is not None:shapes.append((int(self._rect_jacobian._n), int(self._rect_jacobian._m)))
                if self._rect_hessian is not None:shapes.append((int(self._rect_hessian._n), int(self._rect_hessian._m)))
                if self._rect_jplan is not None:shapes.append((int(self._rect_jplan._axes._dimension), int(self._rect_jplan._m)))
                if self._rect_hplan is not None:shapes.append((int(self._rect_hplan._n), int(self._rect_hplan._m)))
                output_dimension = None
                for input_size, output_size in shapes:
                    if input_size != dimension or (output_dimension is not None and output_size != output_dimension):
                        raise ValueError('Provider dimensions must match point and residual dimensions.')
                    output_dimension = output_size
                if self._function is None and self._rect_function is None and not callable(function):raise TypeError('function must be callable.')
                if (jacobian is not None and self._jacobian is None and self._rect_jacobian is None and
                        self._jplan is None and self._rect_jplan is None and not callable(jacobian)):
                    raise TypeError('jacobian must be callable or a native plan.')
                if (hessian is not None and self._hessian is None and self._rect_hessian is None and
                        self._hplan is None and self._rect_hplan is None and not callable(hessian)):
                    raise TypeError('hessian must be callable or a native plan.')
                if f_x0 is not None:
                    cached_dimension = _vector_length(f_x0)
                    if output_dimension is not None and cached_dimension != output_dimension:raise ValueError('Cached residual shape must match providers.')
                    output_dimension = cached_dimension
                if (dimension > OwnedDoubleBuffer.max_size() or dimension*sizeof(double) > workspace_limit or
                        limit*(sizeof(VectorChebyshevRecord)+sizeof(double)*(dimension+1+dimension+dimension*dimension)) > history_limit):
                    raise ValueError('Requested initial storage exceeds workspace_bytes or history_bytes.')
                bootstrap = new OwnedDoubleBuffer(n)
                _pack_vector(x0, bootstrap.data(), n)
                point = _vector_tuple(bootstrap.data(), n)
                finite_point = isfinite(_norm(bootstrap.data(), n))
                del bootstrap
                bootstrap = NULL
                base_value, discovery_error = f_x0, None
                if output_dimension is None:
                    if finite_point:
                        try:
                            base_value = function(point)
                            output_dimension = _vector_length(base_value)
                        except BaseException as failure:
                            discovery_error, base_value, output_dimension = failure, None, dimension
                        discovered = True
                    else:output_dimension = dimension
                if output_dimension == 0 or output_dimension > INT_MAX:raise ValueError('Residual dimension must be positive and fit LAPACK C int.')
                m = output_dimension
                square, cube = dimension*output_dimension, dimension*dimension*output_dimension
                minimum_work = 5*dimension+3*output_dimension+square+cube
                if n == m:minimum_work += square
                else:
                    self._linear_plan = LeastSquaresPlan(output_dimension, dimension, workspace_bytes=workspace_limit)
                    minimum_work += int(self._linear_plan._work_size)
                if self._path != CHEBYSHEV:minimum_work += square
                pivot_count = dimension
                if self._path == GENERAL_MODIFIED:
                    minimum_work += square
                    if n == m:pivot_count += dimension
                if (minimum_work > OwnedDoubleBuffer.max_size() or
                        minimum_work*sizeof(double)+pivot_count*sizeof(int) > workspace_limit or
                        dimension > self._pivots.max_size() or
                        (self._path == GENERAL_MODIFIED and n == m and dimension > self._secondary_pivots.max_size())):
                    raise ValueError('Requested workspace exceeds workspace_bytes or native limits.')
                if jacobian is None:
                    if n == m:self._jplan = JacobianPlan(n)
                    else:self._rect_jplan = RectangularJacobianPlan(n, m, memory_bytes=workspace_limit)
                if hessian is None:
                    if n == m:self._hplan = HessianPlan(n, memory_bytes=workspace_limit)
                    else:self._rect_hplan = RectangularHessianPlan(n, m, memory_bytes=workspace_limit)
                diff_size, plan_steps = 0, 0
                if self._jplan is not None:
                    diff_size = 5*dimension
                    plan_steps += int(self._jplan._step_count)
                if self._rect_jplan is not None:
                    diff_size = max(diff_size, int(self._rect_jplan._work))
                    plan_steps += int(self._rect_jplan._axes._step_count)
                if self._hplan is not None:
                    diff_size = max(diff_size, int(self._hplan._work))
                    plan_steps += int(self._hplan._step_count)
                if self._rect_hplan is not None:
                    diff_size = max(diff_size, int(self._rect_hplan._work))
                    plan_steps += int(self._rect_hplan._step_count)
                workspace_count = minimum_work+diff_size
                if (workspace_count > OwnedDoubleBuffer.max_size() or
                        (workspace_count+plan_steps)*sizeof(double)+pivot_count*sizeof(int) > workspace_limit):
                    raise ValueError('Requested workspace exceeds workspace_bytes or native limits.')
                stride = dimension+output_dimension+square+cube
                record_bytes = sizeof(VectorChebyshevRecord)+stride*sizeof(double)
                if (limit*record_bytes > history_limit or limit*stride > OwnedDoubleBuffer.max_size() or limit > self._records.max_size()):
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                if self._function is None and self._rect_function is None:
                    if n == m:
                        self._function = VectorCallback(function, n)
                        self._function._active = True
                    else:
                        self._rect_function = RectangularCallback(function, n, m)
                        self._rect_function._active = True
                if jacobian is not None and self._jacobian is None and self._rect_jacobian is None and self._jplan is None and self._rect_jplan is None:
                    if n == m:
                        self._jacobian = VectorCallback(jacobian, n, jacobian=True)
                        self._jacobian._active = True
                    else:
                        self._rect_jacobian = RectangularCallback(jacobian, n, m, jacobian=True)
                        self._rect_jacobian._active = True
                if hessian is not None and self._hessian is None and self._rect_hessian is None and self._hplan is None and self._rect_hplan is None:
                    if n == m:
                        self._hessian = HessianCallback(hessian, n)
                        self._hessian._active = True
                    else:
                        self._rect_hessian = RectangularHessianCallback(hessian, n, m)
                        self._rect_hessian._active = True
                self._n, self._m, self._square, self._cube = n, m, square, cube
                self._record_stride, self._diff_size = stride, diff_size
                self._max_iter, self._history_every = max_iter, every
                if self._function is not None:self._function_ptr, self._function_context = self._function._function, self._function._context
                else:self._rect_function_ptr, self._function_context = self._rect_function._function, self._rect_function._context
                if self._jacobian is not None:self._jacobian_ptr, self._jacobian_context = self._jacobian._function, self._jacobian._context
                if self._rect_jacobian is not None:self._rect_jacobian_ptr, self._jacobian_context = self._rect_jacobian._function, self._rect_jacobian._context
                if self._hessian is not None:self._hessian_ptr, self._hessian_context = self._hessian._function, self._hessian._context
                if self._rect_hessian is not None:self._rect_hessian_ptr, self._hessian_context = self._rect_hessian._function, self._rect_hessian._context
                if self._hplan is not None:self._hplan.copy_config(&self._hconfig)
                if self._rect_hplan is not None:self._rect_hplan.copy_config(&self._rhconfig)
                self._storage = new OwnedDoubleBuffer(workspace_count)
                data = self._storage.data()
                self._x = data
                self._f = self._x+n
                self._jac = self._f+m
                self._hess = self._jac+self._square
                self._trial = self._hess+self._cube
                self._trial_f = self._trial+n
                data = self._trial_f+m
                if n == m:
                    self._lu = data
                    data += self._square
                self._s = data
                self._q = self._s+n
                self._t = self._q+m
                self._delta = self._t+n
                self._diff_work = self._delta+n
                data = self._diff_work+self._diff_size
                if self._linear_plan is not None:
                    self._linear_work = data
                    data += self._linear_plan._work_size
                if self._path != CHEBYSHEV:
                    self._secondary = data
                    data += self._square
                if self._path == GENERAL_MODIFIED:self._directional = data
                self._history_data = new OwnedDoubleBuffer(limit*stride)
                self._pivots.resize(n)
                if self._path == GENERAL_MODIFIED and n == m:self._secondary_pivots.resize(n)
                self._records.resize(limit)
                _fill(self._f, m, NAN)
                _fill(self._jac, self._square, NAN)
                _fill(self._hess, self._cube, NAN)
                _pack_vector(point, self._x, n)
                if base_value is not None:
                    if discovered:
                        try:_pack_vector(base_value, self._f, m)
                        except BaseException as failure:
                            discovery_error = failure
                            _fill(self._f, m, NAN)
                    else:_pack_vector(base_value, self._f, m)
                if jac_x0 is not None:_pack_matrix(jac_x0, self._jac, n, m)
                if hess_x0 is not None:_pack_rectangular_tensor(hess_x0, self._hess, n, m)
                memset(&self._state, 0, sizeof(VectorChebyshevRecord))
                self._state.linear_rank = -1
                self._state.f_calls = 1 if discovered else 0
                self._state.precision, self._state.error = INFINITY, _norm(self._f, m)
                init_stop(&self._stop, 0., _norm(self._x, n), f_atol, f_rtol, x_atol, x_rtol)
                if discovery_error is not None:
                    if self._function is not None:self._function._error = discovery_error
                    else:self._rect_function._error = discovery_error
                    self._state.callback_code, self._state.callback_stage = 1, FUNCTION
                    self._stop.reason = CALLBACK_FAILURE
                with nogil:
                    if self._stop.reason != CALLBACK_FAILURE:
                        self._initialize(base_value is not None, jac_x0 is not None, hess_x0 is not None)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                if bootstrap != NULL:del bootstrap
                self._leave()
        finally:self._constructing = False

    def __dealloc__(self):
        if self._storage != NULL:
            del self._storage
        if self._history_data != NULL:
            del self._history_data

    cdef void _ready(self) except *:
        if self._busy:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._busy:raise RuntimeError('This solver is already running.')
        if (not self._constructing and
                ((self._function is None and self._rect_function is None) or
                 (self._jplan is None and self._rect_jplan is None and self._jacobian is None and self._rect_jacobian is None) or
                 (self._hplan is None and self._rect_hplan is None and self._hessian is None and self._rect_hessian is None))):
            raise RuntimeError('Solver requires initialized derivative providers.')
        if ((self._function is not None and self._function._function == NULL) or
                (self._jacobian is not None and self._jacobian._function == NULL) or
                (self._hessian is not None and self._hessian._function == NULL) or
                (self._rect_function is not None and self._rect_function._function == NULL) or
                (self._rect_jacobian is not None and self._rect_jacobian._function == NULL) or
                (self._rect_hessian is not None and self._rect_hessian._function == NULL)):
            raise RuntimeError('Solver requires initialized callback adapters.')
        if ((self._function is not None and self._function._active) or
                (self._jacobian is not None and self._jacobian._active) or
                (self._hessian is not None and self._hessian._active) or
                (self._rect_function is not None and self._rect_function._active) or
                (self._rect_jacobian is not None and self._rect_jacobian._active) or
                (self._rect_hessian is not None and self._rect_hessian._active)):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        if self._function is not None:self._function._active, self._function._error = True, None
        if self._jacobian is not None:self._jacobian._active, self._jacobian._error = True, None
        if self._hessian is not None:self._hessian._active, self._hessian._error = True, None
        if self._rect_function is not None:self._rect_function._active, self._rect_function._error = True, None
        if self._rect_jacobian is not None:self._rect_jacobian._active, self._rect_jacobian._error = True, None
        if self._rect_hessian is not None:self._rect_hessian._active, self._rect_hessian._error = True, None

    cdef void _leave(self) noexcept:
        if self._function is not None:self._function._active = False
        if self._jacobian is not None:self._jacobian._active = False
        if self._hessian is not None:self._hessian._active = False
        if self._rect_function is not None:self._rect_function._active = False
        if self._rect_jacobian is not None:self._rect_jacobian._active = False
        if self._rect_hessian is not None:self._rect_hessian._active = False
        self._busy = False

    cdef int _evaluate(self, int stage, const double* x, double* out) noexcept nogil:
        cdef int code = 0, status = 0, sample_code = 0
        if stage == FUNCTION:
            _fill(out, self._m, NAN)
            self._state.f_calls += 1
            code = _rectangular_residual(x, self._n, self._m, out, <void*>self)
        elif stage == JACOBIAN:
            _fill(out, self._square, NAN)
            self._state.jac_calls += 1
            if self._jplan is not None:
                status = self._jplan.compute(x, self._f, out, self._diff_work,
                    _square_residual, <void*>self,
                    &self._state.f_calls, &sample_code)
            elif self._rect_jplan is not None:
                status = self._rect_jplan.compute(x, self._f, out, self._diff_work,
                    _rectangular_residual, <void*>self, &self._state.f_calls, &sample_code)
            elif self._rect_jacobian_ptr != NULL:
                code = self._rect_jacobian_ptr(x, self._n, self._m, out, self._jacobian_context)
            else:
                code = self._jacobian_ptr(x, self._n, out, self._jacobian_context)
        else:
            _fill(out, self._cube, NAN)
            self._state.hess_calls += 1
            if self._hplan is not None:
                status = compute_hessian(&self._hconfig, x, self._f, out, self._cube,
                    self._diff_work, self._diff_size, _square_residual,
                    <void*>self, &self._state.f_calls, &sample_code)
            elif self._rect_hplan is not None:
                status = compute_rectangular_hessian(&self._rhconfig, x, self._f, out, self._cube,
                    self._diff_work, self._diff_size, _rectangular_residual, <void*>self,
                    &self._state.f_calls, &sample_code)
            elif self._rect_hessian_ptr != NULL:
                code = self._rect_hessian_ptr(x, self._n, self._m, out, self._hessian_context)
            else:
                code = self._hessian_ptr(x, self._n, out, self._hessian_context)
        if status == DIFF_CALLBACK_FAILURE:
            code = sample_code
        elif status == DIFF_NONFINITE:
            self._stop.reason = NONFINITE
            return status
        elif status == DIFF_BREAKDOWN:
            self._stop.reason = BREAKDOWN
            return status
        if code != 0:
            self._state.callback_code = code
            self._state.callback_stage = stage
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef void _derivatives(self, bint cached_jac, bint cached_hess) noexcept nogil:
        if not cached_jac:
            if self._evaluate(JACOBIAN, self._x, self._jac) != 0:
                _fill(self._jac, self._square, NAN)
                _fill(self._hess, self._cube, NAN)
                return
        if not isfinite(_norm(self._jac, self._square)):
            _fill(self._jac, self._square, NAN)
            _fill(self._hess, self._cube, NAN)
            self._stop.reason = NONFINITE
            return
        if not cached_hess:
            if self._evaluate(HESSIAN, self._x, self._hess) != 0:
                _fill(self._hess, self._cube, NAN)
                return
        if not isfinite(_norm(self._hess, self._cube)):
            _fill(self._hess, self._cube, NAN)
            self._stop.reason = NONFINITE

    cdef void _initialize(self, bint cached_f, bint cached_jac,
                           bint cached_hess) noexcept nogil:
        if not isfinite(_norm(self._x, self._n)):
            self._stop.reason = NONFINITE
            return
        if not cached_f:
            if self._evaluate(FUNCTION, self._x, self._trial_f) != 0:
                return
            if not isfinite(_norm(self._trial_f, self._m)):
                self._stop.reason = NONFINITE
                return
            memcpy(self._f, self._trial_f, self._m*sizeof(double))
        self._state.error = _norm(self._f, self._m)
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, INFINITY, False):
            return
        self._derivatives(cached_jac, cached_hess)

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop, self._state.error, self._state.precision,
                      self._state.iters > 0):
            return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS

    cdef void _record(self) noexcept nogil:
        cdef size_t slot
        cdef double* data
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:
            return
        if self._record_count > 0 and self._records[self._record_count-1].iters == self._state.iters:
            slot = self._record_count-1
        else:
            if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:
                return
            if self._record_count >= self._records.size():
                self._history_truncated = True
                return
            slot = self._record_count
            self._record_count += 1
        self._records[slot] = self._state
        data = self._history_data.data()+slot*self._record_stride
        memcpy(data, self._x, self._record_stride*sizeof(double))

    cdef void _step(self) noexcept nogil:
        cdef size_t i
        cdef int status
        cdef int* secondary_pivots
        cdef double* directional
        cdef double precision = 0., difference, alpha
        if self._stop.reason != RUNNING:
            return
        self._state.linear_stage = 1
        self._state.linear_status = 0
        if self._linear_plan is not None:
            memcpy(self._q, self._f, self._m*sizeof(double))
            status = self._linear_plan.compute(self._jac, self._q, self._s,
                self._linear_work, self._pivots.data(), &self._state.linear_rank, &self._state.linear_status)
            if status != 0:
                self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
                self._record()
                return
        else:
            memcpy(self._lu, self._jac, self._square*sizeof(double))
            memcpy(self._s, self._f, self._n*sizeof(double))
            self._state.linear_status = square_factor(<int>self._n, self._lu, self._pivots.data())
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
                self._record()
                return
            # LAPACK may return info=0 after finite inputs overflow during LU.
            # The triangular solve can then produce a finite but incorrect step.
            if not isfinite(_norm(self._lu, self._square)):
                self._stop.reason = NONFINITE
                self._record()
                return
            self._state.linear_status = square_solve_factored(<int>self._n, self._lu,
                                                              self._s, self._pivots.data())
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
                self._record()
                return
            if not isfinite(_norm(self._s, self._n)):
                self._stop.reason = NONFINITE
                self._record()
                return
        if self._path == CHEBYSHEV:
            status = contract_rectangular_hessian(self._hess, self._s, self._n, self._m, self._q)
            if status != 0 or not isfinite(_norm(self._q, self._m)):
                self._stop.reason = BREAKDOWN if status == 2 else NONFINITE
                self._record()
                return
            if self._linear_plan is not None:
                status = self._linear_plan.compute(self._jac, self._q, self._t,
                    self._linear_work, self._pivots.data(), &self._state.linear_rank, &self._state.linear_status)
                if status != 0:
                    self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
                    self._record()
                    return
            else:
                memcpy(self._t, self._q, self._n*sizeof(double))
                self._state.linear_status = square_solve_factored(<int>self._n, self._lu,
                                                                  self._t, self._pivots.data())
                if self._state.linear_status != 0:
                    self._stop.reason = BREAKDOWN
                    self._record()
                    return
        else:
            self._state.linear_stage = 2
            self._state.linear_status = 0
            if self._path == GENERAL_MODIFIED:
                directional = self._directional
                secondary_pivots = self._secondary_pivots.data() if self._linear_plan is None else self._pivots.data()
                alpha = self._alpha
            else:
                directional = self._secondary
                secondary_pivots = self._pivots.data()
                alpha = .5 if self._path == HALLEY else 1.
            status = contract_rectangular_hessian_direction(self._hess, self._s, self._n, self._m, directional)
            if status != 0 or not isfinite(_norm(directional, self._square)):
                self._stop.reason = BREAKDOWN if status == 2 else NONFINITE
                self._record()
                return
            for i in range(self._square):
                self._secondary[i] = fma(-alpha, directional[i], self._jac[i])
            if not isfinite(_norm(self._secondary, self._square)):
                self._stop.reason = NONFINITE
                self._record()
                return
            if self._linear_plan is not None:
                memcpy(self._q, self._f, self._m*sizeof(double))
                status = self._linear_plan.compute(self._secondary, self._q, self._t,
                    self._linear_work, self._pivots.data(), &self._state.linear_rank, &self._state.linear_status)
                if status != 0:
                    self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
                    self._record()
                    return
            else:
                memcpy(self._t, self._f, self._n*sizeof(double))
                self._state.linear_status = square_factor(<int>self._n, self._secondary,
                                                           secondary_pivots)
                if self._state.linear_status != 0:
                    self._stop.reason = BREAKDOWN
                    self._record()
                    return
                if not isfinite(_norm(self._secondary, self._square)):
                    self._stop.reason = NONFINITE
                    self._record()
                    return
                self._state.linear_status = square_solve_factored(<int>self._n, self._secondary,
                                                                  self._t, secondary_pivots)
                if self._state.linear_status != 0:
                    self._stop.reason = BREAKDOWN
                    self._record()
                    return
                if not isfinite(_norm(self._t, self._n)):
                    self._stop.reason = NONFINITE
                    self._record()
                    return
            if self._path == GENERAL_MODIFIED:
                self._state.linear_stage = 3
                self._state.linear_status = 0
                status = rectangular_matrix_vector(self._directional, self._t, self._n, self._m, self._q)
                if status != 0 or not isfinite(_norm(self._q, self._m)):
                    self._stop.reason = BREAKDOWN if status == 2 else NONFINITE
                    self._record()
                    return
                if self._linear_plan is not None:
                    status = self._linear_plan.compute(self._jac, self._q, self._t,
                        self._linear_work, self._pivots.data(), &self._state.linear_rank, &self._state.linear_status)
                    if status != 0:
                        self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
                        self._record()
                        return
                else:
                    memcpy(self._t, self._q, self._n*sizeof(double))
                    self._state.linear_status = square_solve_factored(<int>self._n, self._lu,
                                                                      self._t, self._pivots.data())
                    if self._state.linear_status != 0:
                        self._stop.reason = BREAKDOWN
                        self._record()
                        return
        for i in range(self._n):
            if self._path == CHEBYSHEV:
                self._delta[i] = -self._s[i]-.5*self._t[i]
            elif self._path == GENERAL_MODIFIED:
                # Keep a subnormal half-correction until the final rounding;
                # Chebyshev's established special-path arithmetic is retained.
                self._delta[i] = fma(-.5, self._t[i], -self._s[i])
            elif self._path == HALLEY:
                self._delta[i] = -self._t[i]
            else:
                self._delta[i] = negative_half_sum(self._s[i], self._t[i])
            self._trial[i] = self._x[i]+self._delta[i]
            difference = fabs(self._trial[i]-self._x[i])
            if (not isfinite(self._t[i]) or not isfinite(self._delta[i]) or
                    not isfinite(self._trial[i]) or not isfinite(difference)):
                self._stop.reason = NONFINITE
                self._record()
                return
            precision = fmax(precision, difference)
        self._state.iters += 1
        if self._evaluate(FUNCTION, self._trial, self._trial_f) == 0:
            if not isfinite(_norm(self._trial_f, self._m)):
                self._stop.reason = NONFINITE
            else:
                memcpy(self._x, self._trial, self._n*sizeof(double))
                memcpy(self._f, self._trial_f, self._m*sizeof(double))
                _fill(self._jac, self._square, NAN)
                _fill(self._hess, self._cube, NAN)
                self._state.error = _norm(self._f, self._m)
                self._state.precision = precision
                advance_x(&self._stop, _norm(self._x, self._n))
                self._check()
                if self._stop.reason == RUNNING:
                    self._derivatives(False, False)
        self._record()

    cdef void _raise_callback_error(self) except *:
        if self._function is not None and self._function._error is not None:
            error = self._function._error
            self._function._error = None
            raise error
        if self._jacobian is not None and self._jacobian._error is not None:
            error = self._jacobian._error
            self._jacobian._error = None
            raise error
        if self._hessian is not None and self._hessian._error is not None:
            error = self._hessian._error
            self._hessian._error = None
            raise error
        if self._rect_function is not None and self._rect_function._error is not None:
            error = self._rect_function._error
            self._rect_function._error = None
            raise error
        if self._rect_jacobian is not None and self._rect_jacobian._error is not None:
            error = self._rect_jacobian._error
            self._rect_jacobian._error = None
            raise error
        if self._rect_hessian is not None and self._rect_hessian._error is not None:
            error = self._rect_hessian._error
            self._rect_hessian._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result(),
                                      _stage_name(self._state.callback_stage))

    cdef object _snapshot(self, VectorChebyshevRecord record, const double* data):
        return VectorChebyshevStep(
            _vector_tuple(data, self._n), _vector_tuple(data+self._n, self._m),
            (_matrix_tuple(data+self._n+self._m, self._n, self._m),
             _tensor_tuple(data+self._n+self._m+self._square, self._n, self._m)),
            record.iters, (record.f_calls, record.jac_calls, record.hess_calls),
            record.precision, record.error,
            None if record.reason == RUNNING else _normalize_reason(record.reason),
            record.callback_code, _stage_name(record.callback_stage), record.linear_status,
            _linear_stage_name(record.linear_stage),
            None if record.linear_rank < 0 else record.linear_rank)

    cdef object _copy_result(self):
        cdef VectorChebyshevRecord record = self._state
        return NewtonMethodsReturnType(
            _vector_tuple(self._x, self._n), _vector_tuple(self._f, self._m),
            (_matrix_tuple(self._jac, self._n, self._m), _tensor_tuple(self._hess, self._n, self._m)),
            record.iters, (record.f_calls, record.jac_calls, record.hess_calls),
            record.precision, record.error, False, False,
            termination_reason=None if record.reason == RUNNING else record.reason)

    def step(self):
        """Attempt one finite candidate residual and copy the resulting state."""
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    self._step()
                self._raise_callback_error()
            return self._snapshot(self._state, self._x)
        finally:
            self._leave()

    def run(self):
        """Use the remaining total iteration budget through the native loop."""
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
        cdef size_t i
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i],
                self._history_data.data()+i*self._record_stride)
                for i in range(self._record_count)])
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

    @property
    def linear_status(self):
        self._ready()
        return self._state.linear_status

    @property
    def linear_rank(self):
        self._ready()
        return None if self._state.linear_rank < 0 else self._state.linear_rank

    @property
    def input_dimension(self):self._ready();return self._n

    @property
    def output_dimension(self):self._ready();return self._m

    @property
    def linear_stage(self):
        self._ready()
        return _linear_stage_name(self._state.linear_stage)

    @property
    def method(self):
        self._ready()
        if self._method == MODIFIED_HALLEY:
            return 'modified_halley'
        if self._method == HALLEY:
            return 'halley'
        if self._method == SUPER_HALLEY:
            return 'super_halley'
        return 'chebyshev'

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


def vector_chebyshev(function, jacobian, hessian, x0, **options):
    """Run native Chebyshev; retain an instance for copied history and rank."""
    return VectorChebyshev(function, jacobian, hessian, x0, **options).run()


cdef class VectorHalley(VectorChebyshev):
    """Fixed matrix Halley: solve (J-.5*H[s,.])*v=F, then delta=-v."""


cdef class VectorSuperHalley(VectorChebyshev):
    """Fixed matrix Super-Halley: solve (J-H[s,.])*v=F, delta=-(s+v)/2."""


cdef class VectorModifiedHalley(VectorChebyshev):
    """Parameterized matrix modified Halley with exact fixed special paths."""

    def __init__(self, function, jacobian, hessian, x0,
                 f_x0=None, jac_x0=None, hess_x0=None, *, alpha=.5,
                 etol=1e-10, ertol=8.881784197001252e-16,
                 ptol=1e-12, prtol=8.881784197001252e-16,
                 max_iter=200, history=0, history_every=1,
                 history_bytes=8*1024*1024, workspace_bytes=64*1024*1024):
        self._setup(function, jacobian, hessian, x0, f_x0, jac_x0, hess_x0,
                    etol, ertol, ptol, prtol, max_iter, history, history_every,
                    history_bytes, workspace_bytes, alpha)

    @property
    def alpha(self):
        self._ready()
        return self._alpha


def vector_halley(function, jacobian, hessian, x0, **options):
    """Run fixed square matrix Halley; retain an instance for copied history."""
    return VectorHalley(function, jacobian, hessian, x0, **options).run()


def vector_super_halley(function, jacobian, hessian, x0, **options):
    """Run fixed square matrix Super-Halley with copied state diagnostics."""
    return VectorSuperHalley(function, jacobian, hessian, x0, **options).run()


def vector_modified_halley(function, jacobian, hessian, x0, **options):
    """Run parameterized square matrix modified Halley with guarded alpha."""
    return VectorModifiedHalley(function, jacobian, hessian, x0, **options).run()
