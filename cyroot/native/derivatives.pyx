# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Immutable finite-difference plans with caller-owned native scratch space."""
from cpython.buffer cimport PyObject_CheckBuffer
from libc.float cimport DBL_EPSILON
from libc.limits cimport INT_MAX
from libc.math cimport cbrt, fabs, fmax, isfinite, NAN
from libc.string cimport memcpy

from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer
from .scalar import _nonnegative_integer, NativeCallbackError
from .rectangular cimport RectangularCallback, RectangularFunction
from typing import NamedTuple


cdef enum:
    FORWARD = 0
    BACKWARD = 1
    CENTRAL = 2


cdef void _fill(double* data, size_t count, double value) noexcept nogil:
    cdef size_t i
    for i in range(count):
        data[i] = value


cdef int _failure(double* out, size_t square, double* point,
                  const double* x, size_t n, int status) noexcept nogil:
    _fill(out, square, NAN)
    memcpy(point, x, n * sizeof(double))
    return status


cdef int _compute_jacobian(size_t n, size_t m, int scheme, const double* steps,
        const double* x, const double* f0, double* out, double* work,
        ResidualFunction square_function, RectangularFunction rectangular_function,
        void* context, size_t* calls, int* callback_code) noexcept nogil:
    cdef size_t i, j
    cdef int code
    cdef double h, a, b = 0., scale, a_scaled, b_scaled
    cdef double plus_weight = 0., minus_weight = 0., plus_secant, minus_secant
    cdef double difference, term_plus, term_minus, value
    cdef double* point = work
    cdef double* fplus = work + n
    cdef double* fminus = work + n + m
    cdef double* positive = work + n + 2 * m
    cdef double* negative = work + 2 * n + 2 * m

    callback_code[0] = 0
    _fill(out, n*m, NAN)
    memcpy(point, x, n * sizeof(double))

    # Validate the complete stencil before making any callback attempt.
    # Actual rounded displacements, not requested h, define the secants.
    for i in range(m):
        if not isfinite(f0[i]):return DIFF_NONFINITE
    for j in range(n):
        if not isfinite(x[j]):
            return DIFF_NONFINITE
        if steps != NULL:
            h = steps[j]
        elif scheme == CENTRAL:
            h = fmax(1., fabs(x[j])) * cbrt(DBL_EPSILON)
        else:
            h = fmax(1., fabs(x[j])) * 1.4901161193847656e-8
        if not isfinite(h):
            return DIFF_NONFINITE
        positive[j] = negative[j] = x[j]
        if scheme != BACKWARD:
            positive[j] = x[j] + h
            a = positive[j] - x[j]
            if not isfinite(positive[j]) or not isfinite(a):
                return DIFF_NONFINITE
            if a == 0.:
                return DIFF_BREAKDOWN
        if scheme != FORWARD:
            negative[j] = x[j] - h
            b = negative[j] - x[j]
            if not isfinite(negative[j]) or not isfinite(b):
                return DIFF_NONFINITE
            if b == 0.:
                return DIFF_BREAKDOWN

    for j in range(n):
        if scheme == BACKWARD:
            point[j] = negative[j]
            a = negative[j] - x[j]
        else:
            point[j] = positive[j]
            a = positive[j] - x[j]
        _fill(fplus, m, NAN)
        calls[0] += 1
        if square_function != NULL:
            code = square_function(point, n, fplus, context)
        else:
            code = rectangular_function(point, n, m, fplus, context)
        point[j] = x[j]
        if code != 0:
            callback_code[0] = code
            return _failure(out, n*m, point, x, n, DIFF_CALLBACK_FAILURE)
        for i in range(m):
            if not isfinite(fplus[i]):
                return _failure(out, n*m, point, x, n, DIFF_NONFINITE)

        if scheme == CENTRAL:
            b = negative[j] - x[j]
            point[j] = negative[j]
            _fill(fminus, m, NAN)
            calls[0] += 1
            if square_function != NULL:
                code = square_function(point, n, fminus, context)
            else:
                code = rectangular_function(point, n, m, fminus, context)
            point[j] = x[j]
            if code != 0:
                callback_code[0] = code
                return _failure(out, n*m, point, x, n, DIFF_CALLBACK_FAILURE)
            for i in range(m):
                if not isfinite(fminus[i]):
                    return _failure(out, n*m, point, x, n, DIFF_NONFINITE)
            scale = fmax(fabs(a), fabs(b))
            a_scaled = a / scale
            b_scaled = b / scale
            plus_weight = b_scaled / (b_scaled - a_scaled)
            minus_weight = a_scaled / (a_scaled - b_scaled)
            if not isfinite(plus_weight) or not isfinite(minus_weight):
                return _failure(out, n*m, point, x, n, DIFF_NONFINITE)

        for i in range(m):
            difference = fplus[i] - f0[i]
            if not isfinite(difference):
                return _failure(out, n*m, point, x, n, DIFF_NONFINITE)
            plus_secant = difference / a
            if not isfinite(plus_secant):
                return _failure(out, n*m, point, x, n, DIFF_NONFINITE)
            if scheme == CENTRAL:
                difference = fminus[i] - f0[i]
                if not isfinite(difference):
                    return _failure(out, n*m, point, x, n, DIFF_NONFINITE)
                minus_secant = difference / b
                if not isfinite(minus_secant):
                    return _failure(out, n*m, point, x, n, DIFF_NONFINITE)
                term_plus = plus_weight * plus_secant
                term_minus = minus_weight * minus_secant
                value = term_plus + term_minus
                if (not isfinite(term_plus) or not isfinite(term_minus) or
                        not isfinite(value)):
                    return _failure(out, n*m, point, x, n, DIFF_NONFINITE)
                out[i + j * m] = value
            else:
                out[i + j * m] = plus_secant
    return DIFF_OK


cdef int _steffensen_jacobian(size_t n, const double* x, const double* f0,
        double* out, double* work, ResidualFunction function, void* context,
        size_t* calls, int* callback_code) noexcept nogil:
    # Reuse the complete forward-stencil validation, rounded offsets and
    # whole-output failure handling. Residuals set the steps at every point.
    return _compute_jacobian(n, n, FORWARD, f0, x, f0, out, work,
                             function, NULL, context, calls, callback_code)


cdef class JacobianPlan:
    """Validated, immutable first-derivative configuration for n residuals.

    ``step=None`` selects relative steps at every point. An explicit scalar is
    copied to each axis; a length-n sequence/native double vector supplies
    signed absolute steps. The constructor's workspace_bytes bounds persistent
    step storage. The workspace_bytes property reports the separate 5*n-double
    scratch requirement used by VectorNewton.
    """

    def __init__(self, dimension, scheme='central', step=None,
                 workspace_bytes=64 * 1024 * 1024):
        self._configure(dimension, None, scheme, step, workspace_bytes)

    cdef void _configure(self, object dimension, object output_dimension,
                         object scheme, object step, object workspace_bytes) except *:
        cdef size_t n, i
        cdef int scheme_code
        cdef double value
        cdef OwnedDoubleBuffer* steps = NULL
        cdef DoubleBuffer guard
        cdef VectorView view
        if self._initialized or self._constructing:
            raise RuntimeError('Jacobian plans cannot be reinitialized.')
        self._constructing = True
        try:
            # All coercion may run arbitrary Python. Keep the construction
            # guard set and use checked Python integer products until commit.
            dimension = _nonnegative_integer(dimension, 'dimension')
            byte_limit = _nonnegative_integer(workspace_bytes, 'workspace_bytes')
            if dimension == 0 or dimension > INT_MAX:
                raise ValueError('dimension must be positive and fit the LAPACK C int ABI.')
            output_dimension = dimension if output_dimension is None else _nonnegative_integer(output_dimension, 'output_dimension')
            if output_dimension == 0 or output_dimension > INT_MAX:
                raise ValueError('output_dimension must be positive and fit the LAPACK C int ABI.')
            square = dimension * output_dimension
            if (square > OwnedDoubleBuffer.max_size() or
                    3 * dimension + 2 * output_dimension > OwnedDoubleBuffer.max_size()):
                raise ValueError('Requested Jacobian dimension exceeds native storage limits.')
            if not isinstance(scheme, str):
                raise ValueError('scheme must be forward, backward or central.')
            # Normalize to a plain string so a str subclass cannot survive as
            # mutable configuration or replace native comparisons later.
            normalized = str(scheme)
            if normalized == 'forward':
                scheme_code = FORWARD
            elif normalized == 'backward':
                scheme_code = BACKWARD
            elif normalized == 'central':
                scheme_code = CENTRAL
            else:
                raise ValueError('scheme must be forward, backward or central.')
            n = dimension
            if step is not None:
                if dimension * sizeof(double) > byte_limit:
                    raise ValueError('Requested step storage exceeds workspace_bytes.')
                if isinstance(step, str):
                    raise TypeError('step must be a scalar or a length-dimension numeric vector.')
                try:
                    length = len(step)
                except TypeError:
                    # Scalar conversion precedes allocation; invalid scalar
                    # inputs do not allocate the dimension-sized step vector.
                    value = step
                    if not isfinite(value) or value == 0.:
                        raise ValueError('Explicit steps must be finite and nonzero.')
                    steps = new OwnedDoubleBuffer(n)
                    steps.fill(value)
                else:
                    if length != dimension:
                        raise ValueError('Step vector length must match dimension.')
                    if PyObject_CheckBuffer(step):
                        guard = DoubleBuffer(step)
                        view = guard.vector_view()
                        if view.size() != n:
                            raise ValueError('Step vector length must match dimension.')
                        steps = new OwnedDoubleBuffer(n)
                        for i in range(n):
                            value = view.get(i)
                            if not isfinite(value) or value == 0.:
                                raise ValueError('Explicit steps must be finite and nonzero.')
                            steps.set(i, value)
                    else:
                        steps = new OwnedDoubleBuffer(n)
                        for i in range(n):
                            value = step[i]
                            if not isfinite(value) or value == 0.:
                                raise ValueError('Explicit steps must be finite and nonzero.')
                            steps.set(i, value)
            self._dimension = n
            self._square = square
            self._scheme = scheme_code
            self._step_count = n if steps != NULL else 0
            self._steps = steps
            steps = NULL
            self._initialized = True
        finally:
            if steps != NULL:
                del steps
            self._constructing = False

    def __dealloc__(self):
        if self._steps != NULL:
            del self._steps

    @property
    def dimension(self):
        return self._dimension

    @property
    def scheme(self):
        if self._scheme == FORWARD:
            return 'forward'
        if self._scheme == BACKWARD:
            return 'backward'
        return 'central'

    @property
    def step(self):
        if self._steps == NULL:
            return None
        return tuple([self._steps.get(i) for i in range(self._step_count)])

    @property
    def storage_bytes(self):
        return self._step_count * sizeof(double)

    @property
    def workspace_bytes(self):
        return 5 * self._dimension * sizeof(double)

    cdef int compute(self, const double* x, const double* f0, double* out,
                     double* work, ResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil:
        cdef const double* steps = NULL
        callback_code[0] = 0
        if not self._initialized:return DIFF_BREAKDOWN
        if self._steps != NULL:steps = self._steps.data()
        return _compute_jacobian(self._dimension, self._dimension, self._scheme,
            steps, x, f0, out, work, function, NULL, context, calls, callback_code)


class JacobianEstimate(NamedTuple):
    """Copied rectangular numerical Jacobian and residual-attempt diagnostics."""
    jacobian: tuple
    f_calls: int
    status: str
    callback_code: int


cdef void _pack_point(object source, double* out, size_t count) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector.')
    if PyObject_CheckBuffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != count:raise ValueError('Vector length must match the plan dimension.')
        for i in range(count):out[i] = view.get(i)
    else:
        if len(source) != <Py_ssize_t>count:
            raise ValueError('Vector length must match the plan dimension.')
        for i in range(count):out[i] = source[i]


cdef int _sample(const double* point, size_t n, size_t m,
                 double* out, void* context) noexcept nogil:
    return (<RectangularCallback><object>context).evaluate_raw(point, out)


cdef class RectangularJacobianPlan:
    """Native numerical Jacobian for n inputs and m residuals."""
    def __init__(self, input_dimension, output_dimension, scheme='central', step=None,
                 *, memory_bytes=64*1024*1024):
        cdef JacobianPlan axes
        if self._initialized or self._constructing:
            raise RuntimeError('Jacobian plans cannot be reinitialized.')
        self._constructing = True
        try:
            n = _nonnegative_integer(input_dimension, 'input_dimension')
            m = _nonnegative_integer(output_dimension, 'output_dimension')
            limit = _nonnegative_integer(memory_bytes, 'memory_bytes')
            if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX:
                raise ValueError('Plan dimensions must be positive and fit C int.')
            work = 3*n+2*m
            evaluation = n+m+n*m+work
            step_count = n if step is not None else 0
            total = (evaluation+step_count)*sizeof(double)
            if evaluation > OwnedDoubleBuffer.max_size() or total > limit:
                raise ValueError('Requested rectangular Jacobian exceeds memory_bytes or native limits.')
            axes = JacobianPlan.__new__(JacobianPlan)
            axes._configure(n, m, scheme, step, limit)
            self._axes = axes
            self._m, self._work, self._evaluation_size = m, work, evaluation
            self._memory_bytes = total
            self._initialized = True
        finally:
            self._constructing = False

    cdef int compute(self, const double* x, const double* f0, double* out,
                     double* work, RectangularFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil:
        cdef const double* steps = NULL
        callback_code[0] = 0
        if not self._initialized:return DIFF_BREAKDOWN
        if self._axes._steps != NULL:steps = self._axes._steps.data()
        return _compute_jacobian(self._axes._dimension, self._m, self._axes._scheme,
            steps, x, f0, out, work, NULL, function, context, calls, callback_code)

    def evaluate(self, function, x, f_x=None):
        """Estimate a rectangular Jacobian, adapting an ordinary Python callable."""
        cdef RectangularCallback adapter
        cdef OwnedDoubleBuffer* storage = NULL
        cdef double* point
        cdef double* base
        cdef double* out
        cdef double* work
        cdef size_t n, m, i, j, calls = 0
        cdef int code = 0, status = DIFF_OK
        cdef bint base_failure = False
        if not self._initialized:raise RuntimeError('Jacobian plan is not initialized.')
        n, m = self._axes._dimension, self._m
        if isinstance(function, (JacobianPlan, RectangularJacobianPlan)):
            raise TypeError('Expected a residual callback.')
        adapter = function if isinstance(function, RectangularCallback) else RectangularCallback(function, n, m)
        if adapter._jacobian or adapter._n != n or adapter._m != m:
            raise ValueError('Residual callback dimensions and kind must match the plan.')
        adapter._enter()
        try:
            storage = new OwnedDoubleBuffer(self._evaluation_size)
            point = storage.data()
            base = point+n
            out = base+m
            work = out+n*m
            _fill(out, n*m, NAN)
            _pack_point(x, point, n)
            if f_x is not None:
                _pack_point(f_x, base, m)
            else:
                with nogil:
                    for i in range(n):
                        if not isfinite(point[i]):status = DIFF_NONFINITE
                    if status == DIFF_OK:
                        calls += 1
                        code = adapter.evaluate_raw(point, base)
                        if code != 0:
                            status = DIFF_CALLBACK_FAILURE
                            base_failure = True
            if status == DIFF_OK:
                with nogil:
                    status = self.compute(point, base, out, work, _sample,
                                          <void*>adapter, &calls, &code)
            estimate = JacobianEstimate(
                tuple([tuple([out[i+j*m] for j in range(n)]) for i in range(m)]),
                calls, 'ok' if status == DIFF_OK else 'nonfinite' if status == DIFF_NONFINITE
                else 'breakdown' if status == DIFF_BREAKDOWN else 'callback_failure', code)
            if adapter._error is not None:
                error = adapter._error
                adapter._error = None
                raise error
            if status == DIFF_CALLBACK_FAILURE:
                raise NativeCallbackError(code, estimate, 'function' if base_failure else 'jacobian')
            return estimate
        finally:
            if storage != NULL:del storage
            adapter._leave()

    @property
    def input_dimension(self):return self._axes._dimension if self._initialized else 0
    @property
    def output_dimension(self):return self._m
    @property
    def scheme(self):
        if not self._initialized:raise RuntimeError('Jacobian plan is not initialized.')
        return self._axes.scheme
    @property
    def step(self):
        if not self._initialized:raise RuntimeError('Jacobian plan is not initialized.')
        return self._axes.step
    @property
    def storage_bytes(self):return self._axes.storage_bytes if self._initialized else 0
    @property
    def workspace_bytes(self):return self._work*sizeof(double)
    @property
    def memory_bytes(self):return self._memory_bytes
