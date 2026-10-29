# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native vector Newton, with Python conversion only at its boundaries."""
from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from typing import NamedTuple
from libc.limits cimport INT_MAX
from libc.math cimport fabs, fmax, isfinite, NAN, INFINITY
from libc.string cimport memcpy, memmove, memset
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

from .buffers cimport OwnedDoubleBuffer, VectorView, MatrixView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .linalg cimport square_solve, square_factor, square_solve_factored
from .robinson_ops cimport robinson_nodes
from .wolfe_bittner_ops cimport wolfe_bittner_candidate
from .derivatives cimport (
    JacobianPlan, RectangularJacobianPlan, DIFF_NONFINITE, DIFF_BREAKDOWN,
    DIFF_CALLBACK_FAILURE, _steffensen_jacobian,
)
from .hessians cimport HessianPlan
from .tensor_callbacks cimport HessianCallback
from .scalar import NativeCallbackError, _nonnegative_integer
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types cimport NewtonMethodsReturnType
from cyroot.return_types import _normalize_reason
from cyroot._stopping cimport (
    init_stop, advance_x, check_stop, RUNNING, MAX_ITERATIONS,
    BREAKDOWN, NONFINITE, CALLBACK_FAILURE,
)


class VectorNewtonStep(NamedTuple):
    """Immutable copied point, residual, Jacobian and diagnostic metrics."""
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
    linear_rank: object


cdef void _fill(double* data, size_t count, double value) noexcept nogil:
    cdef size_t i
    for i in range(count):
        data[i] = value


cdef double _norm(const double* data, size_t count) noexcept nogil:
    cdef size_t i
    cdef double value = 0.
    for i in range(count):
        if not isfinite(data[i]):
            return INFINITY
        value = fmax(value, fabs(data[i]))
    return value


cdef void _pack_vector(object source, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector or matrix.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != n:
            raise ValueError('Residual/vector length must match the solver dimension.')
        for i in range(n):
            out[i] = view.get(i)
    else:
        if len(source) != <Py_ssize_t>n:
            raise ValueError('Residual/vector length must match the solver dimension.')
        for i in range(n):
            out[i] = source[i]


cdef void _pack_matrix(object source, double* out, size_t m, size_t n) except *:
    cdef DoubleBuffer guard
    cdef MatrixView view
    cdef size_t i, j
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector or matrix.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.matrix_view()
        if view.rows() != m or view.columns() != n:
            raise ValueError('Jacobian shape must match (residual length, point length).')
        for j in range(n):
            for i in range(m):
                out[i + j * m] = view.get(i, j)
    else:
        if len(source) != <Py_ssize_t>m:
            raise ValueError('Jacobian shape must match (residual length, point length).')
        for i in range(m):
            row = source[i]
            if isinstance(row, (str, bytes, bytearray)) or len(row) != <Py_ssize_t>n:
                raise ValueError('Jacobian shape must match (residual length, point length).')
            for j in range(n):
                out[i + j * m] = row[j]


cdef object _vector_tuple(const double* data, size_t count):
    return tuple([data[i] for i in range(count)])


cdef object _matrix_tuple(const double* data, size_t m, size_t n):
    return tuple([tuple([data[i + j * m] for j in range(n)]) for i in range(m)])


cdef object _vector_length(object source):
    cdef DoubleBuffer guard
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        return int(guard.vector_view().size())
    return len(source)


cdef int _square_residual(const double* x, size_t n, double* out,
                          void* context) noexcept nogil:
    return (<VectorNewton><object>context)._residual(x, out)


cdef int _rectangular_residual(const double* x, size_t n, size_t m, double* out,
                               void* context) noexcept nogil:
    return (<VectorNewton><object>context)._residual(x, out)


cdef int _python_vector(const double* x, size_t n, double* out, void* context) noexcept nogil:
    with gil:
        callback = <VectorCallback><object>context
        try:
            value = callback._python_function(_vector_tuple(x, n))
            if callback._jacobian:
                _pack_matrix(value, out, n, n)
            else:
                _pack_vector(value, out, n)
        except BaseException as error:
            callback._error = error
            return 1
    return 0


cdef class VectorCallback:
    """A dimensioned residual or column-major square-Jacobian callback."""
    def __init__(self, function, dimension, *, jacobian=False):
        if self._function != NULL or self._constructing:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        self._constructing = True
        try:
            n = _nonnegative_integer(dimension, 'dimension')
            if n == 0 or n > INT_MAX:
                raise ValueError('dimension must be positive and fit the LAPACK C int ABI.')
            kind = bool(jacobian)
            if not callable(function):
                raise TypeError('function must be callable.')
            self._dimension = n
            self._jacobian = kind
            self._python_function = function
            self._context = <void*>self
            self._function = _python_vector
        finally:
            self._constructing = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_vector_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.vector_callback.v1'):
            raise ValueError('Expected a cyroot.vector_callback.v1 capsule.')
        descriptor = <cyroot_vector_descriptor*>PyCapsule_GetPointer(capsule, b'cyroot.vector_callback.v1')
        if (descriptor.struct_size < sizeof(cyroot_vector_descriptor) or
                descriptor.abi_version != 1 or descriptor.kind not in (1, 2)):
            raise ValueError('Unsupported vector callback descriptor size, version or kind.')
        return make_vector_callback(descriptor.function, descriptor.dimension,
                                    descriptor.context, (capsule, keepalive), descriptor.kind == 2)

    @property
    def native(self):
        return bool(self._native)

    @property
    def dimension(self):
        return self._dimension

    @property
    def jacobian(self):
        return bool(self._jacobian)


cdef VectorCallback make_vector_callback(VectorFunction function, size_t dimension,
                                         void* context, object keepalive=None, bint jacobian=False):
    if function == NULL:
        raise ValueError('Native vector callback function must not be NULL.')
    if dimension == 0 or dimension > INT_MAX:
        raise ValueError('dimension must be positive and fit the LAPACK C int ABI.')
    cdef VectorCallback callback = VectorCallback.__new__(VectorCallback)
    callback._dimension = dimension
    callback._jacobian = jacobian
    callback._function = function
    callback._context = context
    callback._keepalive = keepalive
    callback._native = True
    return callback


cdef class VectorNewton:
    """Newton with square LU or rectangular rank-aware least-squares steps."""
    def __init__(self, function, jacobian, x0, f_x0=None, jac_x0=None, *,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL,
                 max_iter=MAX_ITER, history=0, history_every=1,
                 history_bytes=8 * 1024 * 1024, workspace_bytes=64 * 1024 * 1024):
        self._setup(function, jacobian, x0, f_x0, jac_x0, etol, ertol, ptol,
            prtol, max_iter, history, history_every, history_bytes, workspace_bytes,
            None, None, None)

    cdef void _setup(self, object function, object jacobian, object x0,
            object f_x0, object jac_x0, object etol, object ertol, object ptol,
            object prtol, object max_iter, object history, object history_every,
            object history_bytes, object workspace_bytes, object previous,
            object previous_f, object orth) except *:
        cdef OwnedDoubleBuffer* point_storage = NULL
        cdef size_t n, m, i
        cdef double* data
        cdef bint discovered = False
        if (self._constructing or self._storage != NULL or self._initialized or
                self._function is not None or self._jacobian is not None or self._plan is not None or
                self._rect_function is not None or self._rect_jacobian is not None or
                self._rect_plan is not None or self._linear_plan is not None):
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            self._steffensen = isinstance(self, TraubSteffensen)
            self._robinson = isinstance(self, Robinson)
            if self._robinson and self._steffensen:
                raise TypeError('Robinson and TraubSteffensen cannot be combined by inheritance.')
            if self._robinson and previous is None:
                raise TypeError('Robinson requires two initial points.')
            if (self._steffensen or self._robinson) and (jacobian is not None or jac_x0 is not None):
                raise TypeError('This method constructs divided differences, not a Jacobian provider/cache.')
            # Native extension-type checks do not inspect Python metadata.
            if isinstance(function, (JacobianPlan, RectangularJacobianPlan, HessianPlan, HessianCallback)):
                raise TypeError('function requires a residual callback.')
            if isinstance(jacobian, (HessianPlan, HessianCallback)):
                raise TypeError('jacobian requires a Jacobian provider.')
            if isinstance(function, VectorCallback):
                self._function = function
                if self._function._function == NULL:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._function._jacobian:
                    raise ValueError('function requires a residual-kind callback.')
            elif isinstance(function, RectangularCallback):
                self._rect_function = function
                if self._rect_function._function == NULL or self._rect_function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._rect_function._jacobian:
                    raise ValueError('function requires a residual-kind callback.')
            if isinstance(jacobian, VectorCallback):
                self._jacobian = jacobian
                if self._jacobian._function == NULL:
                    raise RuntimeError('Jacobian callback is not initialized.')
                if not self._jacobian._jacobian:
                    raise ValueError('jacobian requires a Jacobian-kind callback.')
            elif isinstance(jacobian, RectangularCallback):
                self._rect_jacobian = jacobian
                if self._rect_jacobian._function == NULL or self._rect_jacobian._constructing:
                    raise RuntimeError('Jacobian callback is not initialized.')
                if not self._rect_jacobian._jacobian:
                    raise ValueError('jacobian requires a Jacobian-kind callback.')
            elif isinstance(jacobian, JacobianPlan):
                self._plan = jacobian
                if not self._plan._initialized or self._plan._constructing:
                    raise RuntimeError('Jacobian plan is not initialized.')
            elif isinstance(jacobian, RectangularJacobianPlan):
                self._rect_plan = jacobian
                if not self._rect_plan._initialized or self._rect_plan._constructing:
                    raise RuntimeError('Jacobian plan is not initialized.')
            self._enter()
            try:
                if self._robinson:
                    if type(orth) is int:
                        if orth not in (1, 2):
                            raise ValueError('orth must be identity/1 or efficient/2.')
                        self._rob_identity = orth == 1
                    elif type(orth) is str:
                        if orth not in ('identity', 'efficient'):
                            raise ValueError('orth must be identity/1 or efficient/2.')
                        self._rob_identity = orth == 'identity'
                    else:
                        raise TypeError('orth requires an exact built-in int or str.')
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)
                limit = _nonnegative_integer(history, 'history')
                every = _nonnegative_integer(history_every, 'history_every')
                byte_limit = _nonnegative_integer(history_bytes, 'history_bytes')
                workspace_limit = _nonnegative_integer(workspace_bytes, 'workspace_bytes')
                if every == 0:
                    raise ValueError('history_every must be positive.')
                dimension = _vector_length(x0)
                if dimension == 0 or dimension > INT_MAX:
                    raise ValueError('x0 must be nonempty with dimension fitting LAPACK C int.')
                n = dimension
                # Every explicit provider contributes a shape constraint. Read
                # C fields so Python subclass properties cannot override it.
                shapes = [(dimension, dimension)] if self._steffensen or self._robinson else []
                if self._function is not None:
                    shapes.append((int(self._function._dimension), int(self._function._dimension)))
                if self._jacobian is not None:
                    shapes.append((int(self._jacobian._dimension), int(self._jacobian._dimension)))
                if self._rect_function is not None:
                    shapes.append((int(self._rect_function._n), int(self._rect_function._m)))
                if self._rect_jacobian is not None:
                    shapes.append((int(self._rect_jacobian._n), int(self._rect_jacobian._m)))
                if self._plan is not None:
                    shapes.append((int(self._plan._dimension), int(self._plan._dimension)))
                if self._rect_plan is not None:
                    shapes.append((int(self._rect_plan._axes._dimension), int(self._rect_plan._m)))
                output_dimension = None
                for input_size, output_size in shapes:
                    if input_size != dimension or (output_dimension is not None and output_dimension != output_size):
                        raise ValueError('Provider dimensions must match the point and residual.')
                    output_dimension = output_size
                if self._function is None and self._rect_function is None and not callable(function):
                    raise TypeError('function must be callable.')
                numerical = jacobian is None or self._plan is not None or self._rect_plan is not None
                if not numerical and self._jacobian is None and self._rect_jacobian is None and not callable(jacobian):
                    raise TypeError('jacobian must be callable or a native Jacobian plan.')
                base_value = f_x0
                discovery_error = None
                if f_x0 is not None:
                    cached_dimension = _vector_length(f_x0)
                    if output_dimension is not None and output_dimension != cached_dimension:
                        raise ValueError('Cached residual length must match the provider dimensions.')
                    output_dimension = cached_dimension
                # A copied bootstrap point is bounded before allocation. It is
                # released before the persistent workspace is allocated. The
                # immutable tuple is a Python-boundary copy, not loop storage.
                if (dimension > OwnedDoubleBuffer.max_size() or dimension*sizeof(double) > workspace_limit or
                        (limit and byte_limit < limit*(sizeof(VectorRecord)+sizeof(double)*(2*dimension+1)))):
                    raise ValueError('Requested initial storage exceeds workspace_bytes or history_bytes.')
                point_storage = new OwnedDoubleBuffer(n)
                _pack_vector(x0, point_storage.data(), n)
                point = _vector_tuple(point_storage.data(), n)
                finite_point = isfinite(_norm(point_storage.data(), n))
                del point_storage
                point_storage = NULL
                if output_dimension is None:
                    if finite_point:
                        # No native provider declared m: infer it from exactly
                        # one Python residual call and reuse that base value.
                        try:
                            base_value = function(point)
                            output_dimension = _vector_length(base_value)
                        except BaseException as failure:
                            # Preserve a terminal inspectable state even when
                            # the first callback cannot reveal its output shape.
                            discovery_error = failure
                            base_value = None
                            output_dimension = dimension
                        discovered = True
                    else:
                        # No callback is run at an invalid initial point. With
                        # no declared/cached shape only n is known for reporting.
                        output_dimension = dimension
                if output_dimension == 0 or output_dimension > INT_MAX:
                    raise ValueError('Residual must be nonempty with dimension fitting LAPACK C int.')
                m = output_dimension
                plan_count = (int(self._plan._step_count) if self._plan is not None else
                              int(self._rect_plan._axes._step_count) if self._rect_plan is not None else 0)
                matrix_count = dimension*output_dimension
                # Accepted point/residual/J, trial point/residual, delta[n].
                workspace_count = 3*dimension+2*output_dimension+matrix_count
                stride = dimension+output_dimension+matrix_count
                if n == m:
                    workspace_count += matrix_count
                else:
                    # Plan query is allocation-free; all its reusable scratch
                    # is included in this solver's single owning allocation.
                    self._linear_plan = LeastSquaresPlan(output_dimension, dimension,
                                                          workspace_bytes=workspace_limit)
                    workspace_count += output_dimension+int(self._linear_plan._work_size)
                if self._robinson:
                    workspace_count += 3*dimension+2*matrix_count
                elif numerical:
                    workspace_count += 3*dimension+2*output_dimension
                record_bytes = sizeof(VectorRecord)+sizeof(double)*stride
                if (workspace_count > OwnedDoubleBuffer.max_size() or dimension > self._pivots.max_size() or
                        (workspace_count+plan_count)*sizeof(double)+dimension*sizeof(int) > workspace_limit):
                    raise ValueError('Requested native workspace exceeds workspace_bytes or native limits.')
                if (limit > byte_limit//record_bytes or limit*stride > OwnedDoubleBuffer.max_size() or
                        limit > self._records.max_size()):
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                if jacobian is None and not (self._steffensen or self._robinson):
                    if n == m:
                        self._plan = JacobianPlan(n)
                    else:
                        self._rect_plan = RectangularJacobianPlan(n, m, memory_bytes=workspace_limit)
                self._n, self._m = n, m
                self._square = matrix_count
                self._record_stride = stride
                self._max_iter = max_iter
                self._history_every = every
                if self._function is None and self._rect_function is None:
                    if n == m:
                        self._function = VectorCallback(function, n)
                        self._function._active = True
                    else:
                        self._rect_function = RectangularCallback(function, n, m)
                        self._rect_function._active = True
                if not numerical and self._jacobian is None and self._rect_jacobian is None:
                    if n == m:
                        self._jacobian = VectorCallback(jacobian, n, jacobian=True)
                        self._jacobian._active = True
                    else:
                        self._rect_jacobian = RectangularCallback(jacobian, n, m, jacobian=True)
                        self._rect_jacobian._active = True
                self._storage = new OwnedDoubleBuffer(workspace_count)
                data = self._storage.data()
                self._x = data
                self._f = data+n
                self._jac = self._f+m
                self._trial = self._jac+self._square
                self._trial_f = self._trial+n
                data = self._trial_f+m
                if n == m:
                    self._lu = data
                    self._delta = self._lu+self._square
                    data = self._delta+n
                else:
                    self._delta = data
                    self._rhs = self._delta+n
                    self._linear_work = self._rhs+m
                    data = self._linear_work+self._linear_plan._work_size
                if self._robinson:
                    self._rob_previous = data
                    self._rob_previous_f = data+n
                    self._rob_nodes = data+2*n
                    self._rob_orient = self._rob_nodes+self._square
                    self._rob_row = self._rob_orient+self._square
                elif numerical:self._diff_work = data
                self._history_data = new OwnedDoubleBuffer(limit*stride)
                self._pivots.resize(n)
                self._records.resize(limit)
                _fill(self._f, m, NAN)
                _fill(self._jac, self._square, NAN)
                _pack_vector(point, self._x, n)
                if self._robinson:
                    _pack_vector(previous, self._rob_previous, n)
                    same_point = True
                    for i in range(n):
                        if self._rob_previous[i] != self._x[i]:same_point = False
                    if same_point:
                        raise ValueError('Robinson initial points must be distinct.')
                    _fill(self._rob_previous_f, n, NAN)
                    if previous_f is not None:
                        _pack_vector(previous_f, self._rob_previous_f, n)
                if base_value is not None:
                    if discovered:
                        try:
                            _pack_vector(base_value, self._f, m)
                        except BaseException as failure:
                            discovery_error = failure
                            _fill(self._f, m, NAN)
                    else:
                        _pack_vector(base_value, self._f, m)
                if jac_x0 is not None:_pack_matrix(jac_x0, self._jac, m, n)
                memset(&self._state, 0, sizeof(VectorRecord))
                self._state.linear_rank = -1
                self._state.f_calls = 1 if discovered else 0
                self._state.precision = INFINITY
                self._state.error = _norm(self._f, m)
                init_stop(&self._stop, 0., _norm(self._x, n), etol, ertol, ptol, prtol)
                if discovery_error is not None:
                    if self._function is not None:self._function._error = discovery_error
                    else:self._rect_function._error = discovery_error
                    self._state.callback_code = 1
                    self._state.callback_stage = 1
                    self._stop.reason = CALLBACK_FAILURE
                with nogil:
                    if self._stop.reason != CALLBACK_FAILURE:
                        if self._robinson:
                            self._initialize_robinson(previous_f is not None, base_value is not None)
                        else:
                            self._initialize(base_value is not None, jac_x0 is not None)
                    self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                if point_storage != NULL:del point_storage
                self._leave()
        finally:
            self._constructing = False

    def __dealloc__(self):
        if self._storage != NULL:
            del self._storage
        if self._history_data != NULL:
            del self._history_data

    cdef void _ready(self) except *:
        if self._busy or self._constructing:
            raise RuntimeError('This solver is already running.')
        if not self._initialized:
            raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._busy:
            raise RuntimeError('This solver is already running.')
        # Check every explicit adapter before mutating any reservation.
        if ((self._function is not None and self._function._active) or
                (self._jacobian is not None and self._jacobian._active) or
                (self._rect_function is not None and self._rect_function._active) or
                (self._rect_jacobian is not None and self._rect_jacobian._active)):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        if self._function is not None:
            self._function._active = True
            self._function._error = None
        if self._jacobian is not None:
            self._jacobian._active = True
            self._jacobian._error = None
        if self._rect_function is not None:
            self._rect_function._active = True
            self._rect_function._error = None
        if self._rect_jacobian is not None:
            self._rect_jacobian._active = True
            self._rect_jacobian._error = None

    cdef void _leave(self) noexcept:
        if self._function is not None:
            self._function._active = False
        if self._jacobian is not None:
            self._jacobian._active = False
        if self._rect_function is not None:
            self._rect_function._active = False
        if self._rect_jacobian is not None:
            self._rect_jacobian._active = False
        self._busy = False

    cdef int _residual(self, const double* x, double* out) noexcept nogil:
        if self._rect_function is not None:
            return self._rect_function.evaluate_raw(x, out)
        return self._function._function(x, self._n, out, self._function._context)

    cdef int _evaluate(self, bint jacobian, const double* x, double* out) noexcept nogil:
        cdef int code, status, sample_code = 0
        if jacobian:
            _fill(out, self._square, NAN)
            self._state.jac_calls += 1
            if self._robinson or self._steffensen or self._plan is not None or self._rect_plan is not None:
                if self._robinson:
                    status = self._robinson_jacobian(x, out)
                    sample_code = self._state.callback_code
                elif self._steffensen:
                    status = _steffensen_jacobian(self._n, x, self._f, out,
                        self._diff_work, _square_residual, <void*>self,
                        &self._state.f_calls, &sample_code)
                elif self._plan is not None:
                    status = self._plan.compute(
                        x, self._f, out, self._diff_work, _square_residual, <void*>self,
                        &self._state.f_calls, &sample_code)
                else:
                    status = self._rect_plan.compute(
                        x, self._f, out, self._diff_work, _rectangular_residual, <void*>self,
                        &self._state.f_calls, &sample_code)
                if status == DIFF_CALLBACK_FAILURE:
                    self._state.callback_code = sample_code
                    self._state.callback_stage = 2
                    self._stop.reason = CALLBACK_FAILURE
                elif status == DIFF_NONFINITE:
                    self._stop.reason = NONFINITE
                elif status == DIFF_BREAKDOWN:
                    self._stop.reason = BREAKDOWN
                return status
            else:
                if self._rect_jacobian is not None:
                    code = self._rect_jacobian.evaluate_raw(x, out)
                else:
                    code = self._jacobian._function(x, self._n, out, self._jacobian._context)
        else:
            _fill(out, self._m, NAN)
            self._state.f_calls += 1
            code = self._residual(x, out)
        if code != 0:
            self._state.callback_code = code
            self._state.callback_stage = 2 if jacobian else 1
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef int _robinson_jacobian(self, const double* x, double* out) noexcept nogil:
        cdef size_t i, j
        cdef int status, code
        cdef double difference
        status = robinson_nodes(self._rob_previous, x, self._n, self._rob_identity,
            self._rob_nodes, self._rob_orient, self._rob_row)
        if status != 0:
            return DIFF_NONFINITE if status == 1 else DIFF_BREAKDOWN
        # The columns of S are the ACTUAL rounded probe displacements.
        # Factor S^T once and solve its system for each row of D to obtain J.
        self._state.linear_status = square_factor(<int>self._n, self._rob_orient,
                                                   self._pivots.data())
        if self._state.linear_status != 0:return DIFF_BREAKDOWN
        if not isfinite(_norm(self._rob_orient, self._square)):return DIFF_NONFINITE
        for j in range(self._n):
            _fill(self._trial_f, self._n, NAN)
            self._state.f_calls += 1
            code = self._residual(self._rob_nodes+j*self._n, self._trial_f)
            if code != 0:
                self._state.callback_code = code
                _fill(out, self._square, NAN)
                return DIFF_CALLBACK_FAILURE
            for i in range(self._n):
                if not isfinite(self._trial_f[i]):
                    _fill(out, self._square, NAN)
                    return DIFF_NONFINITE
                difference = self._trial_f[i]-self._f[i]
                if not isfinite(difference):
                    _fill(out, self._square, NAN)
                    return DIFF_NONFINITE
                out[i+j*self._n] = difference
        for i in range(self._n):
            for j in range(self._n):self._rob_row[j] = out[i+j*self._n]
            self._state.linear_status = square_solve_factored(<int>self._n,
                self._rob_orient, self._rob_row, self._pivots.data())
            if self._state.linear_status != 0:
                _fill(out, self._square, NAN)
                return DIFF_BREAKDOWN
            if not isfinite(_norm(self._rob_row, self._n)):
                _fill(out, self._square, NAN)
                return DIFF_NONFINITE
            for j in range(self._n):out[i+j*self._n] = self._rob_row[j]
        return 0

    cdef void _initialize_robinson(self, bint cached_previous, bint cached_f) noexcept nogil:
        cdef size_t i
        cdef double current_error, previous_error, separation = 0., difference
        if (not isfinite(_norm(self._x, self._n)) or
                not isfinite(_norm(self._rob_previous, self._n))):
            self._stop.reason = NONFINITE
            return
        for i in range(self._n):
            difference = fabs(self._x[i]-self._rob_previous[i])
            separation = fmax(separation, difference)
        self._state.precision = separation
        if not cached_previous:
            if self._evaluate(False, self._rob_previous, self._rob_previous_f) != 0:return
        previous_error = _norm(self._rob_previous_f, self._n)
        if not isfinite(previous_error):
            self._stop.reason = NONFINITE
            return
        if not cached_f:
            if self._evaluate(False, self._x, self._trial_f) != 0:return
            if not isfinite(_norm(self._trial_f, self._n)):
                self._stop.reason = NONFINITE
                return
            memcpy(self._f, self._trial_f, self._n*sizeof(double))
        current_error = _norm(self._f, self._n)
        self._state.error = current_error
        if not isfinite(current_error):
            self._stop.reason = NONFINITE
            return
        self._stop.f_scale = fmax(current_error, previous_error)
        self._stop.x_scale = fmax(_norm(self._x, self._n), _norm(self._rob_previous, self._n))
        # Either initial point may already converge; otherwise x1 is active.
        if check_stop(&self._stop,
                previous_error if previous_error < current_error else current_error,
                separation, False):
            if previous_error < current_error:
                memcpy(self._x, self._rob_previous, self._n*sizeof(double))
                memcpy(self._f, self._rob_previous_f, self._n*sizeof(double))
                self._state.error = previous_error
            return
        if self._evaluate(True, self._x, self._jac) != 0:
            _fill(self._jac, self._square, NAN)

    cdef void _initialize(self, bint cached_f, bint cached_jac) noexcept nogil:
        if not isfinite(_norm(self._x, self._n)):
            self._stop.reason = NONFINITE
            return
        if not cached_f:
            if self._evaluate(False, self._x, self._trial_f) != 0:
                return
            if not isfinite(_norm(self._trial_f, self._m)):
                self._stop.reason = NONFINITE
                return
            memcpy(self._f, self._trial_f, self._m * sizeof(double))
        self._state.error = _norm(self._f, self._m)
        self._stop.f_scale = self._state.error
        if check_stop(&self._stop, self._state.error, INFINITY, False):
            return
        if not cached_jac:
            if self._evaluate(True, self._x, self._jac) != 0:
                _fill(self._jac, self._square, NAN)
                return
        if not isfinite(_norm(self._jac, self._square)):
            _fill(self._jac, self._square, NAN)
            self._stop.reason = NONFINITE

    cdef void _check(self) noexcept nogil:
        if check_stop(&self._stop, self._state.error, self._state.precision, self._state.iters > 0):
            return
        if self._max_iter != 0 and self._state.iters >= self._max_iter:
            self._stop.reason = MAX_ITERATIONS

    cdef void _record(self) noexcept nogil:
        cdef size_t slot
        cdef double* data
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:
            return
        if self._record_count > 0 and self._records[self._record_count - 1].iters == self._state.iters:
            slot = self._record_count - 1
        else:
            if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:
                return
            if self._record_count >= self._records.size():
                self._history_truncated = True
                return
            slot = self._record_count
            self._record_count += 1
        self._records[slot] = self._state
        data = self._history_data.data() + slot * self._record_stride
        memcpy(data, self._x, self._n * sizeof(double))
        memcpy(data + self._n, self._f, self._m * sizeof(double))
        memcpy(data + self._n + self._m, self._jac, self._square * sizeof(double))

    cdef void _step(self) noexcept nogil:
        cdef size_t i
        cdef int status
        cdef double precision = 0., difference
        if self._stop.reason != RUNNING:
            return
        if self._linear_plan is None:
            memcpy(self._lu, self._jac, self._square * sizeof(double))
            for i in range(self._n):
                self._delta[i] = -self._f[i]
            self._state.linear_status = square_solve(<int>self._n, self._lu, self._delta, &self._pivots[0])
            if self._state.linear_status != 0:
                self._stop.reason = BREAKDOWN
                self._record()
                return
            # LAPACK success alone does not certify finite factors.
            if not isfinite(_norm(self._lu, self._square)):
                self._stop.reason = NONFINITE
                self._record()
                return
        else:
            for i in range(self._m):self._rhs[i] = -self._f[i]
            status = self._linear_plan.compute(
                self._jac, self._rhs, self._delta, self._linear_work, &self._pivots[0],
                &self._state.linear_rank, &self._state.linear_status)
            if status != 0:
                self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
                self._record()
                return
        for i in range(self._n):
            self._trial[i] = self._x[i] + self._delta[i]
            difference = fabs(self._trial[i] - self._x[i])
            if (not isfinite(self._delta[i]) or not isfinite(self._trial[i]) or
                    not isfinite(difference)):
                self._stop.reason = NONFINITE
                self._record()
                return
            precision = fmax(precision, difference)
        self._state.iters += 1
        if self._evaluate(False, self._trial, self._trial_f) == 0:
            if not isfinite(_norm(self._trial_f, self._m)):
                self._stop.reason = NONFINITE
            else:
                if self._robinson:
                    memcpy(self._rob_previous, self._x, self._n*sizeof(double))
                    memcpy(self._rob_previous_f, self._f, self._n*sizeof(double))
                memcpy(self._x, self._trial, self._n * sizeof(double))
                memcpy(self._f, self._trial_f, self._m * sizeof(double))
                _fill(self._jac, self._square, NAN)
                self._state.error = _norm(self._f, self._m)
                self._state.precision = precision
                advance_x(&self._stop, _norm(self._x, self._n))
                self._check()
                if self._stop.reason == RUNNING:
                    if self._evaluate(True, self._x, self._jac) != 0:
                        _fill(self._jac, self._square, NAN)
                    elif not isfinite(_norm(self._jac, self._square)):
                        _fill(self._jac, self._square, NAN)
                        self._stop.reason = NONFINITE
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
        if self._rect_function is not None and self._rect_function._error is not None:
            error = self._rect_function._error
            self._rect_function._error = None
            raise error
        if self._rect_jacobian is not None and self._rect_jacobian._error is not None:
            error = self._rect_jacobian._error
            self._rect_jacobian._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            stage = 'function' if self._state.callback_stage == 1 else 'jacobian'
            raise NativeCallbackError(self._state.callback_code, self._copy_result(), stage)

    cdef object _snapshot(self, VectorRecord record, const double* data):
        reason = None if record.reason == RUNNING else _normalize_reason(record.reason)
        stage = 'function' if record.callback_stage == 1 else 'jacobian' if record.callback_stage == 2 else None
        return VectorNewtonStep(
            _vector_tuple(data, self._n), _vector_tuple(data + self._n, self._m),
            _matrix_tuple(data + self._n + self._m, self._m, self._n), record.iters,
            (record.f_calls, record.jac_calls), record.precision, record.error,
            reason, record.callback_code, stage, record.linear_status,
            None if record.linear_rank < 0 else record.linear_rank)

    cdef object _copy_result(self):
        cdef VectorRecord record = self._state
        root = _vector_tuple(self._x, self._n)
        residual = _vector_tuple(self._f, self._m)
        jacobian = _matrix_tuple(self._jac, self._m, self._n)
        return NewtonMethodsReturnType(
            root, residual, jacobian, record.iters, (record.f_calls, record.jac_calls),
            record.precision, record.error, False, False,
            termination_reason=None if record.reason == RUNNING else record.reason)

    def step(self):
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
            return tuple([self._snapshot(self._records[i], self._history_data.data() + i * self._record_stride)
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
        return ('function' if self._state.callback_stage == 1 else
                'jacobian' if self._state.callback_stage == 2 else None)

    @property
    def linear_status(self):
        self._ready()
        return self._state.linear_status

    @property
    def linear_rank(self):
        self._ready()
        return None if self._state.linear_rank < 0 else self._state.linear_rank

    @property
    def input_dimension(self):
        self._ready()
        return self._n

    @property
    def output_dimension(self):
        self._ready()
        return self._m

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        return self.step()


cdef class Robinson(VectorNewton):
    """Square two-point Robinson with deterministic identity/Householder probes."""
    def __init__(self, function, x0, x1, f_x0=None, f_x1=None, *, orth='efficient',
                 etol=ETOL, ertol=ERTOL, ptol=PTOL,
                 prtol=PRTOL, max_iter=MAX_ITER, history=0,
                 history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024):
        self._setup(function, None, x1, f_x1, None, etol, ertol, ptol, prtol,
            max_iter, history, history_every, history_bytes, workspace_bytes,
            x0, f_x0, orth)


def robinson(function, x0, x1, **options):
    """Run native Robinson; retain Robinson for step/history inspection."""
    return Robinson(function, x0, x1, **options).run()


cdef class TraubSteffensen(VectorNewton):
    """Square Traub-Steffensen with signed residual-dependent divided differences.

    Uses the native Newton state/history and strict LU. Zero or collapsed
    perturbations report breakdown; there is no automatic-step fallback.
    """
    def __init__(self, function, x0, f_x0=None, *, etol=ETOL,
                 ertol=ERTOL, ptol=PTOL,
                 prtol=PRTOL, max_iter=MAX_ITER, history=0,
                 history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024):
        VectorNewton.__init__(self, function, None, x0, f_x0=f_x0,
            etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter,
            history=history, history_every=history_every, history_bytes=history_bytes,
            workspace_bytes=workspace_bytes)


def traub_steffensen(function, x0, **options):
    """Run native Traub-Steffensen; retain TraubSteffensen for step/history."""
    return TraubSteffensen(function, x0, **options).run()


def vector_newton(function, jacobian, x0, **options):
    """Run native Newton; use VectorNewton for retained history and rank."""
    return VectorNewton(function, jacobian, x0, **options).run()


class WolfeBittnerResult(NamedTuple):
    """Copied best and active states, with latest affine-solve diagnostics."""
    root: tuple
    f_root: tuple
    active_root: tuple
    active_f_root: tuple
    iters: int
    f_calls: int
    precision: float
    error: float
    termination_reason: object
    callback_code: int
    callback_stage: object
    linear_status: int
    linear_rank: object
    affine_error: float

    @property
    def success(self):return self.termination_reason == 'residual_tolerance'
    @property
    def converged(self):return self.success
    @property
    def optimal(self):return self.success


WolfeBittnerStep = WolfeBittnerResult


cdef class WolfeBittner:
    """Bounded multipoint affine interpolation with rank-aware native solves."""
    def __init__(self, function, x0s, f_x0s=None, *, rcond=1e-12,
                 etol=ETOL, ertol=ERTOL,
                 ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                 history=0, history_every=1, history_bytes=8*1024*1024,
                 workspace_bytes=64*1024*1024):
        cdef OwnedDoubleBuffer* bootstrap = NULL
        cdef double* data
        cdef size_t i, j, source, target
        cdef double temporary
        cdef bint discovered = False
        if (self._constructing or self._initialized or self._storage != NULL or
                self._function is not None or self._rect_function is not None):
            raise RuntimeError('Solvers cannot be reinitialized.')
        self._constructing = True
        try:
            if isinstance(function, VectorCallback):
                self._function = function
                if self._function._function == NULL or self._function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._function._jacobian:raise ValueError('function requires a residual-kind callback.')
            elif isinstance(function, RectangularCallback):
                self._rect_function = function
                if self._rect_function._function == NULL or self._rect_function._constructing:
                    raise RuntimeError('Residual callback is not initialized.')
                if self._rect_function._jacobian:raise ValueError('function requires a residual-kind callback.')
            elif isinstance(function, (JacobianPlan, RectangularJacobianPlan, HessianPlan,
                                      HessianCallback)):
                raise TypeError('function requires a residual callback.')
            elif not callable(function):raise TypeError('function must be callable.')
            self._enter()
            try:
                etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)
                limit = _nonnegative_integer(history, 'history')
                every = _nonnegative_integer(history_every, 'history_every')
                byte_limit = _nonnegative_integer(history_bytes, 'history_bytes')
                workspace_limit = _nonnegative_integer(workspace_bytes, 'workspace_bytes')
                if every == 0:raise ValueError('history_every must be positive.')
                if isinstance(x0s, (str, bytes, bytearray)):
                    raise TypeError('x0s must be a nonempty matrix of points.')
                pool_count = len(x0s)
                if pool_count == 0 or pool_count > 32767:
                    raise ValueError('The point count must be between 1 and 32767.')
                dimension = _vector_length(x0s[0])
                if dimension == 0 or dimension > INT_MAX:
                    raise ValueError('Points must be nonempty with dimension fitting C int.')
                output_dimension = None
                if self._function is not None:
                    if self._function._dimension != dimension:raise ValueError('Callback input dimension must match points.')
                    output_dimension = int(self._function._dimension)
                elif self._rect_function is not None:
                    if self._rect_function._n != dimension:raise ValueError('Callback input dimension must match points.')
                    output_dimension = int(self._rect_function._m)
                if f_x0s is not None:
                    if isinstance(f_x0s, (str, bytes, bytearray)) or len(f_x0s) != pool_count:
                        raise ValueError('Cached residuals must have one row per point.')
                    cached_dimension = _vector_length(f_x0s[0])
                    if output_dimension is not None and cached_dimension != output_dimension:
                        raise ValueError('Cached residual dimensions must match the callback.')
                    output_dimension = cached_dimension
                # The bootstrap contains ALL coordinates, so no callback sees
                # an early valid row when a later point is invalid.
                bootstrap_count = pool_count*dimension
                if (bootstrap_count > OwnedDoubleBuffer.max_size() or bootstrap_count* sizeof(double) > workspace_limit or
                        (limit and limit*(sizeof(VectorRecord)+sizeof(double)*(2*(dimension+1)+1)) > byte_limit)):
                    raise ValueError('Initial storage exceeds workspace_bytes or history_bytes.')
                self._n, self._k = dimension, pool_count
                bootstrap = new OwnedDoubleBuffer(bootstrap_count)
                for i in range(self._k):_pack_vector(x0s[i], bootstrap.data()+i*self._n, self._n)
                points = tuple([_vector_tuple(bootstrap.data()+i*self._n, self._n) for i in range(self._k)])
                finite_points = isfinite(_norm(bootstrap.data(), bootstrap_count))
                del bootstrap
                bootstrap = NULL
                base_value = None
                discovery_error = None
                if output_dimension is None:
                    if finite_points:
                        try:
                            base_value = function(points[0])
                            output_dimension = _vector_length(base_value)
                        except BaseException as failure:
                            discovery_error = failure
                            base_value = None
                            output_dimension = dimension
                        discovered = True
                    else:output_dimension = dimension
                if output_dimension == 0 or output_dimension >= 32767:
                    raise ValueError('Residual dimension must be between 1 and 32766.')
                self._m, self._rows = output_dimension, output_dimension+1
                self._linear_plan = LeastSquaresPlan(output_dimension+1, pool_count,
                    rcond=rcond, workspace_bytes=workspace_limit)
                affine_count = (output_dimension+1)*pool_count
                workspace_count = (pool_count*(dimension+output_dimension)+3*(dimension+output_dimension)+
                    affine_count+output_dimension+1+pool_count+int(self._linear_plan._work_size))
                stride = 2*(dimension+output_dimension)
                record_bytes = sizeof(VectorRecord)+sizeof(double)*(stride+1)
                if (workspace_count > OwnedDoubleBuffer.max_size() or pool_count > self._pivots.max_size() or
                        workspace_count*sizeof(double)+pool_count*sizeof(int) > workspace_limit):
                    raise ValueError('Requested native workspace exceeds workspace_bytes or native limits.')
                if (limit > byte_limit//record_bytes or limit*stride > OwnedDoubleBuffer.max_size() or
                        limit > self._records.max_size() or limit > self._affine_records.max_size()):
                    raise ValueError('Requested history exceeds history_bytes or native limits.')
                self._stride, self._max_iter, self._history_every = stride, max_iter, every
                if self._function is None and self._rect_function is None:
                    if self._n == self._m:
                        self._function = VectorCallback(function, self._n)
                        self._function._active = True
                    else:
                        self._rect_function = RectangularCallback(function, self._n, self._m)
                        self._rect_function._active = True
                self._storage = new OwnedDoubleBuffer(workspace_count)
                data = self._storage.data()
                self._points = data
                self._values = self._points+self._k*self._n
                self._x = self._values+self._k*self._m
                self._f = self._x+self._n
                self._active_x = self._f+self._m
                self._active_f = self._active_x+self._n
                self._trial = self._active_f+self._m
                self._trial_f = self._trial+self._n
                self._affine = self._trial_f+self._m
                self._rhs = self._affine+self._rows*self._k
                self._weights = self._rhs+self._rows
                self._work = self._weights+self._k
                self._history_data = new OwnedDoubleBuffer(limit*stride)
                self._pivots.resize(self._k)
                self._records.resize(limit)
                self._affine_records.resize(limit)
                for i in range(self._k):_pack_vector(points[i], self._points+i*self._n, self._n)
                _fill(self._values, self._k*self._m, NAN)
                if f_x0s is not None:
                    for i in range(self._k):_pack_vector(f_x0s[i], self._values+i*self._m, self._m)
                elif base_value is not None:
                    try:_pack_vector(base_value, self._values, self._m)
                    except BaseException as failure:
                        discovery_error = failure
                        _fill(self._values, self._m, NAN)
                memcpy(self._x, self._points, self._n*sizeof(double))
                memcpy(self._active_x, self._points, self._n*sizeof(double))
                _fill(self._f, self._m, NAN)
                _fill(self._active_f, self._m, NAN)
                _fill(self._rhs, self._rows, 0.)
                self._rhs[0] = 1.
                self._affine_error = NAN
                memset(&self._state, 0, sizeof(VectorRecord))
                self._state.linear_rank = -1
                self._state.f_calls = 1 if discovered else 0
                self._state.error, self._state.precision = INFINITY, INFINITY
                init_stop(&self._stop, 0., _norm(self._points, self._k*self._n), etol, ertol, ptol, prtol)
                if discovery_error is not None:
                    if self._function is not None:self._function._error = discovery_error
                    else:self._rect_function._error = discovery_error
                    self._state.callback_code, self._state.callback_stage = 1, 1
                    self._stop.reason = CALLBACK_FAILURE
                with nogil:
                    if self._stop.reason != CALLBACK_FAILURE:self._initialize(f_x0s is not None, discovered)
                if self._stop.reason == RUNNING:
                    # Stable initial worst-to-best order; Python boundary only.
                    errors = [_norm(self._values+i*self._m, self._m) for i in range(self._k)]
                    ordering = sorted(range(self._k), key=errors.__getitem__, reverse=True)
                    positions, at = list(range(self._k)), list(range(self._k))
                    for target in range(self._k):
                        source = positions[ordering[target]]
                        if source == target:continue
                        for j in range(self._n):
                            temporary = self._points[target*self._n+j]
                            self._points[target*self._n+j] = self._points[source*self._n+j]
                            self._points[source*self._n+j] = temporary
                        for j in range(self._m):
                            temporary = self._values[target*self._m+j]
                            self._values[target*self._m+j] = self._values[source*self._m+j]
                            self._values[source*self._m+j] = temporary
                        positions[at[target]], positions[ordering[target]] = source, target
                        at[source], at[target] = at[target], ordering[target]
                with nogil:self._record()
                self._initialized = True
                self._raise_callback_error()
            finally:
                if bootstrap != NULL:del bootstrap
                self._leave()
        finally:self._constructing = False

    def __dealloc__(self):
        if self._storage != NULL:del self._storage
        if self._history_data != NULL:del self._history_data

    cdef void _ready(self) except *:
        if self._busy or self._constructing:raise RuntimeError('This solver is already running.')
        if not self._initialized:raise RuntimeError('This solver is not initialized.')

    cdef void _enter(self) except *:
        if self._busy:raise RuntimeError('This solver is already running.')
        if ((self._function is not None and self._function._active) or
                (self._rect_function is not None and self._rect_function._active)):
            raise RuntimeError('This callback adapter is already active.')
        self._busy = True
        if self._function is not None:
            self._function._active, self._function._error = True, None
        if self._rect_function is not None:
            self._rect_function._active, self._rect_function._error = True, None

    cdef void _leave(self) noexcept:
        if self._function is not None:self._function._active = False
        if self._rect_function is not None:self._rect_function._active = False
        self._busy = False

    cdef int _evaluate(self, const double* point, double* out) noexcept nogil:
        cdef int code
        _fill(out, self._m, NAN)
        self._state.f_calls += 1
        if self._rect_function is not None:code = self._rect_function.evaluate_raw(point, out)
        else:code = self._function._function(point, self._n, out, self._function._context)
        if code != 0:
            _fill(out, self._m, NAN)
            self._state.callback_code, self._state.callback_stage = code, 1
            self._stop.reason = CALLBACK_FAILURE
        return code

    cdef double _geometry(self) noexcept nogil:
        cdef size_t i, j
        cdef double lower, upper, width = 0., difference
        for i in range(self._n):
            lower = upper = self._points[i]
            for j in range(1, self._k):
                if self._points[j*self._n+i] < lower:lower = self._points[j*self._n+i]
                if self._points[j*self._n+i] > upper:upper = self._points[j*self._n+i]
            difference = upper-lower
            if not isfinite(difference):return INFINITY
            width = fmax(width, difference)
        return width

    cdef void _initialize(self, bint cached, bint discovered) noexcept nogil:
        cdef size_t i
        cdef double error
        if not isfinite(_norm(self._points, self._k*self._n)):
            self._stop.reason = NONFINITE
            return
        self._state.precision = self._geometry()
        for i in range(self._k):
            if not cached and not (discovered and i == 0):
                if self._evaluate(self._points+i*self._n, self._trial_f) != 0:return
                if not isfinite(_norm(self._trial_f, self._m)):
                    self._stop.reason = NONFINITE
                    return
                memcpy(self._values+i*self._m, self._trial_f, self._m*sizeof(double))
            error = _norm(self._values+i*self._m, self._m)
            if not isfinite(error):
                self._stop.reason = NONFINITE
                return
            self._stop.f_scale = fmax(self._stop.f_scale, error)
            if i == 0 or error < self._state.error:
                memcpy(self._x, self._points+i*self._n, self._n*sizeof(double))
                memcpy(self._f, self._values+i*self._m, self._m*sizeof(double))
                memcpy(self._active_x, self._x, self._n*sizeof(double))
                memcpy(self._active_f, self._f, self._m*sizeof(double))
                self._state.error = error
        self._stop.previous_x_norm = _norm(self._active_x, self._n)
        check_stop(&self._stop, self._state.error, self._state.precision, False)

    cdef void _record(self) noexcept nogil:
        cdef size_t slot
        cdef double* data
        self._state.reason = <int>self._stop.reason
        if self._records.size() == 0:return
        if self._record_count and self._records[self._record_count-1].iters == self._state.iters:
            slot = self._record_count-1
        else:
            if self._state.iters % self._history_every != 0 and self._stop.reason == RUNNING:return
            if self._record_count >= self._records.size():
                self._history_truncated = True
                return
            slot = self._record_count
            self._record_count += 1
        self._records[slot] = self._state
        self._affine_records[slot] = self._affine_error
        data = self._history_data.data()+slot*self._stride
        memcpy(data, self._x, self._n*sizeof(double))
        memcpy(data+self._n, self._f, self._m*sizeof(double))
        memcpy(data+self._n+self._m, self._active_x, self._n*sizeof(double))
        memcpy(data+2*self._n+self._m, self._active_f, self._m*sizeof(double))

    cdef void _step(self) noexcept nogil:
        cdef size_t i, j
        cdef int status
        cdef double error
        if self._stop.reason != RUNNING:return
        for j in range(self._k):
            self._affine[j*self._rows] = 1.
            for i in range(self._m):self._affine[1+i+j*self._rows] = self._values[i+j*self._m]
        self._affine_error = NAN
        status = self._linear_plan.compute(self._affine, self._rhs, self._weights,
            self._work, self._pivots.data(), &self._state.linear_rank, &self._state.linear_status)
        if status == 0:
            status = wolfe_bittner_candidate(self._affine, self._weights, self._points,
                self._rows, self._k, self._n, self._trial, &self._affine_error)
        if status != 0:
            self._stop.reason = NONFINITE if status == 1 else BREAKDOWN
            self._record()
            return
        self._state.iters += 1
        if self._evaluate(self._trial, self._trial_f) != 0:
            self._record()
            return
        error = _norm(self._trial_f, self._m)
        if not isfinite(error):
            self._stop.reason = NONFINITE
            self._record()
            return
        memmove(self._points, self._points+self._n, (self._k-1)*self._n*sizeof(double))
        memmove(self._values, self._values+self._m, (self._k-1)*self._m*sizeof(double))
        memcpy(self._points+(self._k-1)*self._n, self._trial, self._n*sizeof(double))
        memcpy(self._values+(self._k-1)*self._m, self._trial_f, self._m*sizeof(double))
        memcpy(self._active_x, self._trial, self._n*sizeof(double))
        memcpy(self._active_f, self._trial_f, self._m*sizeof(double))
        if error < self._state.error:
            memcpy(self._x, self._trial, self._n*sizeof(double))
            memcpy(self._f, self._trial_f, self._m*sizeof(double))
            self._state.error = error
        self._state.precision = self._geometry()
        advance_x(&self._stop, _norm(self._active_x, self._n))
        if not check_stop(&self._stop, self._state.error, self._state.precision, True):
            if self._max_iter and self._state.iters >= self._max_iter:self._stop.reason = MAX_ITERATIONS
        self._record()

    cdef object _snapshot(self, VectorRecord record, const double* data, double affine_error):
        return WolfeBittnerResult(_vector_tuple(data, self._n), _vector_tuple(data+self._n, self._m),
            _vector_tuple(data+self._n+self._m, self._n), _vector_tuple(data+2*self._n+self._m, self._m),
            record.iters, record.f_calls, record.precision, record.error,
            None if record.reason == RUNNING else _normalize_reason(record.reason), record.callback_code,
            'function' if record.callback_stage == 1 else None, record.linear_status,
            None if record.linear_rank < 0 else record.linear_rank, affine_error)

    cdef object _copy_result(self):return self._snapshot(self._state, self._x, self._affine_error)

    cdef void _raise_callback_error(self) except *:
        if self._function is not None and self._function._error is not None:
            error = self._function._error
            self._function._error = None
            raise error
        if self._rect_function is not None and self._rect_function._error is not None:
            error = self._rect_function._error
            self._rect_function._error = None
            raise error
        if self._stop.reason == CALLBACK_FAILURE:
            raise NativeCallbackError(self._state.callback_code, self._copy_result(), 'function')

    def step(self):
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:self._leave()

    def run(self):
        self._ready()
        self._enter()
        try:
            if self._stop.reason == RUNNING:
                with nogil:
                    while self._stop.reason == RUNNING:self._step()
                self._raise_callback_error()
            return self._copy_result()
        finally:self._leave()

    @property
    def result(self):
        self._ready()
        self._enter()
        try:return self._copy_result()
        finally:self._leave()
    @property
    def history(self):
        self._ready()
        self._enter()
        try:
            return tuple([self._snapshot(self._records[i], self._history_data.data()+i*self._stride,
                                        self._affine_records[i]) for i in range(self._record_count)])
        finally:self._leave()
    @property
    def done(self):self._ready();return self._stop.reason != RUNNING
    @property
    def history_truncated(self):self._ready();return bool(self._history_truncated)
    @property
    def input_dimension(self):self._ready();return self._n
    @property
    def output_dimension(self):self._ready();return self._m
    @property
    def point_count(self):self._ready();return self._k
    @property
    def linear_status(self):self._ready();return self._state.linear_status
    @property
    def linear_rank(self):self._ready();return None if self._state.linear_rank < 0 else self._state.linear_rank
    @property
    def callback_code(self):self._ready();return self._state.callback_code
    @property
    def callback_stage(self):self._ready();return 'function' if self._state.callback_stage == 1 else None
    def __iter__(self):return self
    def __next__(self):
        if self.done:raise StopIteration
        return self.step()


def wolfe_bittner(function, x0s, **options):
    """Run the native multipoint state; retain WolfeBittner for inspection."""
    return WolfeBittner(function, x0s, **options).run()
