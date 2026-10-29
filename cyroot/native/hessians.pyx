# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Native Hessians for square and rectangular vector residuals on actual rounded nodes."""
from typing import NamedTuple
from numbers import Complex, Real
from libc.float cimport DBL_EPSILON
from libc.limits cimport INT_MAX
from libc.math cimport cbrt, fabs, fmax, frexp, isfinite, ldexp, NAN
from libc.string cimport memcpy

from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .derivatives cimport (
    DIFF_OK, DIFF_NONFINITE, DIFF_BREAKDOWN, DIFF_CALLBACK_FAILURE,
)
from .vector cimport VectorCallback
from .rectangular cimport RectangularCallback
from .scalar import NativeCallbackError
from .higher_derivatives import _checked_integer


cdef enum:
    FORWARD = 0
    BACKWARD = 1
    CENTRAL = 2


class HessianEstimate(NamedTuple):
    """Copied immutable tensor and total residual-attempt diagnostics."""
    hessian: tuple
    f_calls: int
    status: str
    callback_code: int


cdef object _status_name(int status):
    if status == DIFF_OK:
        return 'ok'
    if status == DIFF_NONFINITE:
        return 'nonfinite'
    if status == DIFF_BREAKDOWN:
        return 'breakdown'
    return 'callback_failure'


cdef void _fill(double* data, size_t count, double value) noexcept nogil:
    cdef size_t i
    for i in range(count):
        data[i] = value


cdef bint _finite_vector(const double* data, size_t n) noexcept nogil:
    cdef size_t i
    for i in range(n):
        if not isfinite(data[i]):
            return False
    return True


cdef int _failure(double* out, size_t cube, double* point,
                  const double* x, size_t n, int status) noexcept nogil:
    _fill(out, cube, NAN)
    memcpy(point, x, n * sizeof(double))
    return status


cdef double _rescale(double value, double n1, double n2,
                      double d1, double d2, double d3) noexcept nogil:
    """Multiply/divide finite factors without premature range loss.

    Denominators are nonzero. Normalizing each factor keeps mantissa arithmetic
    bounded; only the final result can overflow/underflow binary64.
    """
    cdef int ev, en1, en2, ed1, ed2, ed3
    cdef double mv, mn1, mn2, md1, md2, md3, mantissa
    if value == 0.:
        return value
    mv = frexp(value, &ev)
    mn1 = frexp(n1, &en1)
    mn2 = frexp(n2, &en2)
    md1 = frexp(d1, &ed1)
    md2 = frexp(d2, &ed2)
    md3 = frexp(d3, &ed3)
    mantissa = ((mv * mn1) * mn2) / md1 / md2 / md3
    return ldexp(mantissa, ev + en1 + en2 - ed1 - ed2 - ed3)


cdef bint _valid_config(const HessianConfig* config,
                        size_t output_capacity, size_t capacity) noexcept nogil:
    cdef size_t n = config.dimension, square, cube, work
    cdef size_t maximum = OwnedDoubleBuffer.max_size()
    if n == 0 or n > INT_MAX or n > maximum // n:
        return False
    square = n * n
    if n > maximum // square or square > maximum // 2:
        return False
    cube = square * n
    if n > (maximum - 2 * square) // 10:
        return False
    work = 2 * square + 10 * n
    return (config.square == square and config.cube == cube and
            config.work == work and config.scheme >= FORWARD and
            config.scheme <= CENTRAL and output_capacity >= cube and capacity >= work)


cdef bint _valid_rectangular_config(const RectangularHessianConfig* config,
                                    size_t output_capacity,
                                    size_t capacity) noexcept nogil:
    cdef size_t n = config.input_dimension, m = config.output_dimension
    cdef size_t matrix, tensor, work, evaluation
    cdef size_t maximum = OwnedDoubleBuffer.max_size()
    if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX or n > maximum // m:
        return False
    matrix = n * m
    if n > maximum // matrix or matrix > maximum // 2:
        return False
    tensor = matrix * n
    work = 2 * matrix
    if n > (maximum - work) // 8:
        return False
    work += 8 * n
    if m > (maximum - work) // 2:
        return False
    work += 2 * m
    # All evaluation segments must fit one native allocation as well.
    evaluation = tensor
    if work > maximum - evaluation:
        return False
    evaluation += work
    if n > maximum - evaluation:
        return False
    evaluation += n
    if m > maximum - evaluation:
        return False
    return (config.matrix_size == matrix and config.tensor_size == tensor and
            config.work == work and config.scheme >= FORWARD and
            config.scheme <= CENTRAL and output_capacity >= tensor and capacity >= work)


cdef struct _SquareHessianBridge:
    ResidualFunction function
    void* context


cdef int _square_hessian_residual(const double* x, size_t n, size_t m,
                                  double* out, void* context) noexcept nogil:
    cdef _SquareHessianBridge* bridge = <_SquareHessianBridge*>context
    return bridge.function(x, n, out, bridge.context)


cdef int compute_hessian(const HessianConfig* config,
                         const double* x, const double* f0,
                         double* out, size_t output_capacity,
                         double* workspace, size_t capacity,
                         ResidualFunction function, void* context,
                         size_t* calls, int* callback_code) noexcept nogil:
    cdef RectangularHessianConfig rectangular
    cdef _SquareHessianBridge bridge
    callback_code[0] = 0
    if not _valid_config(config, output_capacity, capacity):
        return DIFF_BREAKDOWN
    rectangular.input_dimension = config.dimension
    rectangular.output_dimension = config.dimension
    rectangular.matrix_size = config.square
    rectangular.tensor_size = config.cube
    rectangular.work = config.work
    rectangular.scheme = config.scheme
    rectangular.steps = config.steps
    bridge.function, bridge.context = function, context
    return compute_rectangular_hessian(&rectangular, x, f0, out, output_capacity,
                                       workspace, capacity, _square_hessian_residual,
                                       &bridge, calls, callback_code)


cdef int compute_rectangular_hessian(const RectangularHessianConfig* config,
                         const double* x, const double* f0,
                         double* out, size_t output_capacity,
                         double* workspace, size_t capacity,
                         RectangularFunction function, void* context,
                         size_t* calls, int* callback_code) noexcept nogil:
    cdef size_t n, m, tensor, i, j, k, side, side2, index, mirror
    cdef double* axes
    cdef double* first
    cdef double* second
    cdef double* an
    cdef double* bn
    cdef double* scales
    cdef double* alpha
    cdef double* beta
    cdef double* point
    cdef double* corner
    cdef double* compensation
    cdef double* sample
    cdef double* sample_j
    cdef double* sample_k
    cdef double h, a, b, separation, value, difference, secant1, secant2
    cdef double wjp, wjm, wkp, wkm, weight_j, weight_k, weight_scale_j, weight_scale_k
    cdef double coordinate_j, coordinate_k, rectangle, coefficient, term, adjusted, updated
    cdef int code
    callback_code[0] = 0
    if not _valid_rectangular_config(config, output_capacity, capacity):
        return DIFF_BREAKDOWN
    n, m = config.input_dimension, config.output_dimension
    tensor = config.tensor_size
    axes = workspace
    first = axes + 2 * config.matrix_size
    second = first + n
    an = second + n
    bn = an + n
    scales = bn + n
    alpha = scales + n
    beta = alpha + n
    point = beta + n
    corner = point + n
    compensation = corner + m
    _fill(out, tensor, NAN)
    memcpy(point, x, n * sizeof(double))
    if not _finite_vector(x, n) or not _finite_vector(f0, m):
        return DIFF_NONFINITE

    # Preflight every axis before any sample, including interpolant weights.
    for j in range(n):
        if config.steps != NULL:
            h = config.steps[j]
        elif config.scheme == CENTRAL:
            h = fmax(1., fabs(x[j])) * 0.0001220703125
        else:
            h = fmax(1., fabs(x[j])) * cbrt(DBL_EPSILON)
        if not isfinite(h):
            return DIFF_NONFINITE
        if config.scheme == CENTRAL:
            first[j], second[j] = x[j] + h, x[j] - h
        elif config.scheme == FORWARD:
            first[j], second[j] = x[j] + h, x[j] + 2. * h
        else:
            first[j], second[j] = x[j] - h, x[j] - 2. * h
        a, b = first[j] - x[j], second[j] - x[j]
        if (not isfinite(first[j]) or not isfinite(second[j]) or
                not isfinite(a) or not isfinite(b)):
            return DIFF_NONFINITE
        if a == 0. or b == 0. or first[j] == second[j]:
            return DIFF_BREAKDOWN
        scales[j] = fmax(fabs(a), fabs(b))
        an[j], bn[j] = a / scales[j], b / scales[j]
        separation = an[j] - bn[j]
        if not isfinite(separation):
            return DIFF_NONFINITE
        if an[j] == 0. or bn[j] == 0. or separation == 0.:
            return DIFF_BREAKDOWN
        alpha[j] = bn[j] / (bn[j] - an[j])
        beta[j] = an[j] / separation
        wjp, wjm = alpha[j] / an[j], beta[j] / bn[j]
        if (not isfinite(alpha[j]) or not isfinite(beta[j]) or
                not isfinite(wjp) or not isfinite(wjm)):
            return DIFF_NONFINITE

    # Axis-major cache: first then second sample for each increasing axis.
    for j in range(n):
        for side in range(2):
            point[j] = first[j] if side == 0 else second[j]
            sample = axes + (2 * j + side) * m
            _fill(sample, m, NAN)
            calls[0] += 1
            code = function(point, n, m, sample, context)
            point[j] = x[j]
            if code != 0:
                callback_code[0] = code
                return _failure(out, tensor, point, x, n, DIFF_CALLBACK_FAILURE)
            if not _finite_vector(sample, m):
                return _failure(out, tensor, point, x, n, DIFF_NONFINITE)

    for j in range(n):
        separation = an[j] - bn[j]
        for i in range(m):
            difference = axes[(2*j)*m+i] - f0[i]
            if not isfinite(difference):
                return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
            secant1 = difference / an[j]
            difference = axes[(2*j+1)*m+i] - f0[i]
            if not isfinite(difference):
                return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
            secant2 = difference / bn[j]
            if not isfinite(secant1) or not isfinite(secant2):
                return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
            difference = secant1 - secant2
            if isfinite(difference):
                value = _rescale(difference, 2., 1., separation, scales[j], scales[j])
            else:
                # An overflowing difference of opposite finite secants can be
                # halved safely, then restored within the final scaled ratio.
                difference = .5 * secant1 - .5 * secant2
                value = _rescale(difference, 4., 1., separation, scales[j], scales[j])
            if not isfinite(value):
                return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
            out[i+m*(j+n*j)] = value

    for j in range(n):
        for k in range(j+1, n):
            if config.scheme != CENTRAL:
                point[j], point[k] = first[j], first[k]
                _fill(corner, m, NAN)
                calls[0] += 1
                code = function(point, n, m, corner, context)
                point[j], point[k] = x[j], x[k]
                if code != 0:
                    callback_code[0] = code
                    return _failure(out, tensor, point, x, n, DIFF_CALLBACK_FAILURE)
                if not _finite_vector(corner, m):
                    return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                for i in range(m):
                    secant1 = corner[i] - axes[(2*j)*m+i]
                    secant2 = axes[(2*k)*m+i] - f0[i]
                    rectangle = secant1 - secant2
                    if (not isfinite(secant1) or not isfinite(secant2) or
                            not isfinite(rectangle)):
                        return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                    value = _rescale(rectangle, 1./an[j], 1./an[k], scales[j], scales[k], 1.)
                    if not isfinite(value):
                        return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                    out[i+m*(j+n*k)] = out[i+m*(k+n*j)] = value
                continue

            # Factor a common coefficient magnitude out of the four-corner
            # sum. Symmetric normalized coefficients become +/-1, preserving
            # subnormal rectangles that multiplying individually by1/4 loses.
            wjp, wjm = alpha[j]/an[j], beta[j]/bn[j]
            wkp, wkm = alpha[k]/an[k], beta[k]/bn[k]
            weight_scale_j, weight_scale_k = fmax(fabs(wjp), fabs(wjm)), fmax(fabs(wkp), fabs(wkm))
            wjp, wjm = wjp/weight_scale_j, wjm/weight_scale_j
            wkp, wkm = wkp/weight_scale_k, wkm/weight_scale_k
            for i in range(m):
                out[i+m*(j+n*k)] = 0.
                compensation[i] = 0.
            for side in range(2):
                coordinate_j = first[j] if side == 0 else second[j]
                weight_j = wjp if side == 0 else wjm
                sample_j = axes + (2*j+side)*m
                for side2 in range(2):
                    coordinate_k = first[k] if side2 == 0 else second[k]
                    weight_k = wkp if side2 == 0 else wkm
                    coefficient = weight_j * weight_k
                    sample_k = axes + (2*k+side2)*m
                    point[j], point[k] = coordinate_j, coordinate_k
                    _fill(corner, m, NAN)
                    calls[0] += 1
                    code = function(point, n, m, corner, context)
                    point[j], point[k] = x[j], x[k]
                    if code != 0:
                        callback_code[0] = code
                        return _failure(out, tensor, point, x, n, DIFF_CALLBACK_FAILURE)
                    if not _finite_vector(corner, m):
                        return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                    for i in range(m):
                        secant1, secant2 = corner[i]-sample_j[i], sample_k[i]-f0[i]
                        rectangle = secant1-secant2
                        term = coefficient * rectangle
                        index = i+m*(j+n*k)
                        adjusted = term-compensation[i]
                        updated = out[index]+adjusted
                        if (not isfinite(secant1) or not isfinite(secant2) or
                                not isfinite(rectangle) or not isfinite(term) or
                                not isfinite(adjusted) or not isfinite(updated)):
                            return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                        compensation[i] = (updated-out[index])-adjusted
                        if not isfinite(compensation[i]):
                            return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                        out[index] = updated
            for i in range(m):
                index, mirror = i+m*(j+n*k), i+m*(k+n*j)
                value = _rescale(out[index], weight_scale_j, weight_scale_k,
                                 scales[j], scales[k], 1.)
                if not isfinite(value):
                    return _failure(out, tensor, point, x, n, DIFF_NONFINITE)
                out[index] = out[mirror] = value
    return DIFF_OK


cdef double _real_step(object value) except *:
    cdef double normalized
    if (isinstance(value, (str, bytes, bytearray, bool)) or
            (isinstance(value, Complex) and not isinstance(value, Real)) or
            getattr(getattr(value, 'dtype', None), 'kind', None) in ('b', 'c')):
        raise TypeError('steps must be finite nonzero real scalars.')
    try:
        len(value)
    except TypeError:
        pass
    else:
        raise TypeError('steps must be finite nonzero real scalars.')
    normalized = value
    if not isfinite(normalized) or normalized == 0.:
        raise ValueError('Explicit steps must be finite and nonzero.')
    return normalized


cdef void _pack_vector(object source, double* out, size_t n) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != n:
            raise ValueError('Vector length must match dimension.')
        for i in range(n):
            out[i] = view.get(i)
    else:
        if len(source) != n:
            raise ValueError('Vector length must match dimension.')
        for i in range(n):
            out[i] = source[i]


cdef object _tensor_tuple(const double* out, size_t n):
    return tuple([tuple([tuple([out[i+n*(j+n*k)] for k in range(n)])
                        for j in range(n)]) for i in range(n)])


cdef int _evaluate_native(const HessianConfig* config, const double* x,
                          double* base, double* out, double* work,
                          bint cached, ResidualFunction function, void* context,
                          size_t* calls, int* callback_code,
                          bint* base_failure) noexcept nogil:
    cdef int code
    callback_code[0] = 0
    base_failure[0] = False
    _fill(out, config.cube, NAN)
    memcpy(work + 2*config.square + 7*config.dimension, x,
           config.dimension*sizeof(double))
    if not _finite_vector(x, config.dimension):
        return DIFF_NONFINITE
    if not cached:
        _fill(base, config.dimension, NAN)
        calls[0] += 1
        code = function(x, config.dimension, base, context)
        if code != 0:
            callback_code[0] = code
            base_failure[0] = True
            return DIFF_CALLBACK_FAILURE
    if not _finite_vector(base, config.dimension):
        return DIFF_NONFINITE
    return compute_hessian(config, x, base, out, config.cube, work, config.work,
                            function, context, calls, callback_code)


cdef int _parse_scheme(object scheme) except -1:
    if not isinstance(scheme, str):
        raise ValueError('scheme must be forward, backward or central.')
    normalized = str(scheme)
    if normalized == 'forward':return FORWARD
    if normalized == 'backward':return BACKWARD
    if normalized == 'central':return CENTRAL
    raise ValueError('scheme must be forward, backward or central.')


cdef OwnedDoubleBuffer* _copy_steps(size_t n, object step) except? NULL:
    cdef OwnedDoubleBuffer* steps = NULL
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    cdef double value
    if step is None:return NULL
    try:
        if (isinstance(step, (str, bytes, bytearray, bool)) or
                (isinstance(step, Complex) and not isinstance(step, Real)) or
                getattr(getattr(step, 'dtype', None), 'kind', None) in ('b', 'c')):
            raise TypeError('step must be a real scalar or length-dimension vector.')
        if is_native_double_buffer(step) and not isinstance(step, Real):
            guard = DoubleBuffer(step)
            view = guard.vector_view()
            if view.size() != n:raise ValueError('Step vector length must match dimension.')
            steps = new OwnedDoubleBuffer(n)
            for i in range(n):
                value = view.get(i)
                if not isfinite(value) or value == 0.:
                    raise ValueError('Explicit steps must be finite and nonzero.')
                steps.set(i, value)
        else:
            try:length = len(step)
            except TypeError:
                value = _real_step(step)
                steps = new OwnedDoubleBuffer(n)
                steps.fill(value)
            else:
                if length != n:raise ValueError('Step vector length must match dimension.')
                steps = new OwnedDoubleBuffer(n)
                for i in range(n):steps.set(i, _real_step(step[i]))
        return steps
    except:
        if steps != NULL:del steps
        raise


cdef class HessianPlan:
    """Immutable actual-node Hessians with a bounded native evaluation footprint.

    memory_bytes bounds step storage plus one evaluation's private input, base,
    tensor and scratch. Python snapshots/owners and allocator overhead are not
    included. Concurrent evaluations each allocate their own evaluation storage.
    """

    def __init__(self, dimension, scheme='central', step=None, *,
                 memory_bytes=64 * 1024 * 1024):
        cdef int scheme_code
        cdef size_t n, i
        cdef double value
        cdef OwnedDoubleBuffer* steps = NULL
        cdef DoubleBuffer guard
        cdef VectorView view
        if self._initialized or self._constructing:
            raise RuntimeError('Hessian plans cannot be reinitialized.')
        self._constructing = True
        try:
            degree = _checked_integer(dimension, 'dimension')
            budget = _checked_integer(memory_bytes, 'memory_bytes')
            if degree == 0 or degree > INT_MAX:
                raise ValueError('dimension must be positive and fit the vector callback ABI.')
            scheme_code = _parse_scheme(scheme)
            square, cube = degree*degree, degree*degree*degree
            work = 2*square+10*degree
            evaluation = cube+work+2*degree
            step_count = degree if step is not None else 0
            total_bytes = (evaluation+step_count)*sizeof(double)
            if (total_bytes > budget or evaluation > OwnedDoubleBuffer.max_size() or
                    step_count > OwnedDoubleBuffer.max_size() or
                    cube > OwnedDoubleBuffer.max_size() or work > OwnedDoubleBuffer.max_size()):
                raise ValueError('Requested Hessian exceeds memory_bytes or native limits.')
            n = degree
            steps = _copy_steps(n, step)
            self._dimension, self._square, self._cube = n, square, cube
            self._work, self._evaluation_size = work, evaluation
            self._step_count, self._memory_bytes = step_count, total_bytes
            self._scheme = scheme_code
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

    cdef void copy_config(self, HessianConfig* config) noexcept nogil:
        config.dimension, config.square = self._dimension, self._square
        config.cube, config.work, config.scheme = self._cube, self._work, self._scheme
        config.steps = NULL
        if self._steps != NULL:
            config.steps = self._steps.data()

    cdef int compute(self, const double* x, const double* f0,
                     double* out, size_t output_capacity,
                     double* workspace, size_t capacity,
                     ResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil:
        cdef HessianConfig config
        callback_code[0] = 0
        if not self._initialized:
            return DIFF_BREAKDOWN
        self.copy_config(&config)
        return compute_hessian(&config, x, f0, out, output_capacity, workspace,
                                capacity, function, context, calls, callback_code)

    def evaluate(self, function, x, f_x=None):
        """Evaluate with a reserved Python/native residual and private storage."""
        cdef VectorCallback adapter
        cdef OwnedDoubleBuffer* storage = NULL
        cdef double* point
        cdef double* base
        cdef double* out
        cdef double* work
        cdef HessianConfig config
        cdef ResidualFunction function_pointer
        cdef void* context
        cdef size_t calls = 0
        cdef int code = 0, status
        cdef bint base_failure = False, cached = f_x is not None
        if not self._initialized:
            raise RuntimeError('Hessian plan is not initialized.')
        adapter = function if isinstance(function, VectorCallback) else VectorCallback(function, self._dimension)
        if adapter._function == NULL:
            raise RuntimeError('Residual callback adapter is not initialized.')
        if adapter._dimension != self._dimension or adapter._jacobian:
            raise ValueError('Expected a residual callback with the plan dimension.')
        if adapter._active:
            raise RuntimeError('This callback adapter is already active.')
        adapter._active = True
        adapter._error = None
        try:
            self.copy_config(&config)
            function_pointer, context = <ResidualFunction>adapter._function, adapter._context
            storage = new OwnedDoubleBuffer(self._evaluation_size)
            point = storage.data()
            base = point+self._dimension
            out = base+self._dimension
            work = out+self._cube
            _pack_vector(x, point, self._dimension)
            if cached:
                _pack_vector(f_x, base, self._dimension)
            with nogil:
                status = _evaluate_native(&config, point, base, out, work, cached,
                    function_pointer, context, &calls, &code, &base_failure)
            estimate = HessianEstimate(_tensor_tuple(out, self._dimension), calls,
                                       _status_name(status), code)
            if adapter._error is not None:
                error = adapter._error
                adapter._error = None
                raise error
            if status == DIFF_CALLBACK_FAILURE:
                raise NativeCallbackError(code, estimate, 'function' if base_failure else 'hessian')
            return estimate
        finally:
            if storage != NULL:
                del storage
            adapter._active = False

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
        return None if self._steps == NULL else tuple([self._steps.get(i) for i in range(self._step_count)])

    @property
    def shape(self):
        return (self._dimension,)*3

    @property
    def sample_count(self):
        if self._scheme == CENTRAL:
            return 2*self._square
        return self._dimension*(self._dimension+3)//2

    @property
    def storage_bytes(self):
        return self._step_count*sizeof(double)

    @property
    def workspace_size(self):
        return self._work

    @property
    def workspace_bytes(self):
        return self._work*sizeof(double)

    @property
    def output_bytes(self):
        return self._cube*sizeof(double)

    @property
    def evaluation_bytes(self):
        return self._evaluation_size*sizeof(double)

    @property
    def memory_bytes(self):
        return self._memory_bytes


cdef object _rectangular_tensor_tuple(const double* out, size_t n, size_t m):
    return tuple([tuple([tuple([out[i+m*(j+n*k)] for k in range(n)])
                        for j in range(n)]) for i in range(m)])


cdef int _evaluate_rectangular_native(const RectangularHessianConfig* config,
        const double* x, double* base, double* out, double* work, bint cached,
        RectangularFunction function, void* context, size_t* calls,
        int* callback_code, bint* base_failure) noexcept nogil:
    cdef int code
    cdef size_t n = config.input_dimension, m = config.output_dimension
    callback_code[0] = 0
    base_failure[0] = False
    _fill(out, config.tensor_size, NAN)
    memcpy(work+2*config.matrix_size+7*n, x, n*sizeof(double))
    if not _finite_vector(x, n):return DIFF_NONFINITE
    if not cached:
        _fill(base, m, NAN)
        calls[0] += 1
        code = function(x, n, m, base, context)
        if code != 0:
            callback_code[0] = code
            base_failure[0] = True
            return DIFF_CALLBACK_FAILURE
    if not _finite_vector(base, m):return DIFF_NONFINITE
    return compute_rectangular_hessian(config, x, base, out, config.tensor_size,
        work, config.work, function, context, calls, callback_code)


cdef class RectangularHessianPlan:
    """Immutable actual-node Hessians of F:R^n -> R^m, shape(m,n,n)."""
    def __init__(self, input_dimension, output_dimension, scheme='central', step=None,
                 *, memory_bytes=64*1024*1024):
        cdef OwnedDoubleBuffer* steps = NULL
        cdef int scheme_code
        if self._initialized or self._constructing:
            raise RuntimeError('Hessian plans cannot be reinitialized.')
        self._constructing = True
        try:
            n = _checked_integer(input_dimension, 'input_dimension')
            m = _checked_integer(output_dimension, 'output_dimension')
            budget = _checked_integer(memory_bytes, 'memory_bytes')
            if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX:
                raise ValueError('Input/output dimensions must be positive and fit C int.')
            scheme_code = _parse_scheme(scheme)
            matrix, tensor = n*m, m*n*n
            work = 2*matrix+8*n+2*m
            evaluation = n+m+tensor+work
            step_count = n if step is not None else 0
            total_bytes = (evaluation+step_count)*sizeof(double)
            if (total_bytes > budget or evaluation > OwnedDoubleBuffer.max_size() or
                    step_count > OwnedDoubleBuffer.max_size() or
                    tensor > OwnedDoubleBuffer.max_size() or work > OwnedDoubleBuffer.max_size()):
                raise ValueError('Requested Hessian exceeds memory_bytes or native limits.')
            steps = _copy_steps(n, step)
            self._n, self._m = n, m
            self._matrix_size, self._tensor_size = matrix, tensor
            self._work, self._evaluation_size = work, evaluation
            self._step_count, self._memory_bytes = step_count, total_bytes
            self._scheme, self._steps = scheme_code, steps
            steps = NULL
            self._initialized = True
        finally:
            if steps != NULL:del steps
            self._constructing = False

    def __dealloc__(self):
        if self._steps != NULL:del self._steps

    cdef void copy_config(self, RectangularHessianConfig* config) noexcept nogil:
        config.input_dimension, config.output_dimension = self._n, self._m
        config.matrix_size, config.tensor_size = self._matrix_size, self._tensor_size
        config.work, config.scheme = self._work, self._scheme
        config.steps = NULL
        if self._steps != NULL:config.steps = self._steps.data()

    cdef int compute(self, const double* x, const double* f0,
                     double* out, size_t output_capacity,
                     double* workspace, size_t capacity,
                     RectangularFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil:
        cdef RectangularHessianConfig config
        callback_code[0] = 0
        if not self._initialized:return DIFF_BREAKDOWN
        self.copy_config(&config)
        return compute_rectangular_hessian(&config, x, f0, out, output_capacity,
            workspace, capacity, function, context, calls, callback_code)

    def evaluate(self, function, x, f_x=None):
        """Evaluate a reserved Python/native residual with private native spans."""
        cdef RectangularCallback rectangular = None
        cdef VectorCallback square = None
        cdef OwnedDoubleBuffer* storage = NULL
        cdef double* point
        cdef double* base
        cdef double* out
        cdef double* work
        cdef RectangularHessianConfig config
        cdef RectangularFunction function_pointer
        cdef void* context
        cdef _SquareHessianBridge bridge
        cdef size_t calls = 0
        cdef int code = 0, status
        cdef bint base_failure = False, cached = f_x is not None
        if self._constructing or not self._initialized:
            raise RuntimeError('Hessian plan is not initialized.')
        if isinstance(function, VectorCallback):
            square = function
            if square._function == NULL or square._constructing:
                raise RuntimeError('Residual callback adapter is not initialized.')
            if self._n != self._m or square._dimension != self._n or square._jacobian:
                raise ValueError('Square residual callback dimensions must match the plan.')
            if square._active:raise RuntimeError('This callback adapter is already active.')
            square._active, square._error = True, None
            bridge.function, bridge.context = <ResidualFunction>square._function, square._context
            function_pointer, context = _square_hessian_residual, &bridge
        else:
            rectangular = function if isinstance(function, RectangularCallback) else RectangularCallback(function, self._n, self._m)
            if (rectangular._function == NULL or rectangular._constructing):
                raise RuntimeError('Residual callback adapter is not initialized.')
            if rectangular._n != self._n or rectangular._m != self._m or rectangular._jacobian:
                raise ValueError('Residual callback dimensions must match the plan.')
            rectangular._enter()
            function_pointer, context = rectangular._function, rectangular._context
        try:
            self.copy_config(&config)
            storage = new OwnedDoubleBuffer(self._evaluation_size)
            point = storage.data()
            base = point+self._n
            out = base+self._m
            work = out+self._tensor_size
            _pack_vector(x, point, self._n)
            if cached:_pack_vector(f_x, base, self._m)
            with nogil:
                status = _evaluate_rectangular_native(&config, point, base, out, work, cached,
                    function_pointer, context, &calls, &code, &base_failure)
            estimate = HessianEstimate(_rectangular_tensor_tuple(out, self._n, self._m),
                                       calls, _status_name(status), code)
            if square is not None and square._error is not None:
                error = square._error
                square._error = None
                raise error
            if rectangular is not None and rectangular._error is not None:
                error = rectangular._error
                rectangular._error = None
                raise error
            if status == DIFF_CALLBACK_FAILURE:
                raise NativeCallbackError(code, estimate, 'function' if base_failure else 'hessian')
            return estimate
        finally:
            if storage != NULL:del storage
            if square is not None:square._active = False
            if rectangular is not None:rectangular._release()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def scheme(self):
        return 'forward' if self._scheme == FORWARD else 'backward' if self._scheme == BACKWARD else 'central'
    @property
    def step(self):
        return None if self._steps == NULL else tuple([self._steps.get(i) for i in range(self._step_count)])
    @property
    def shape(self):return self._m, self._n, self._n
    @property
    def sample_count(self):
        return 2*int(self._n)*int(self._n) if self._scheme == CENTRAL else int(self._n)*(int(self._n)+3)//2
    @property
    def storage_bytes(self):return self._step_count*sizeof(double)
    @property
    def workspace_size(self):return self._work
    @property
    def workspace_bytes(self):return self._work*sizeof(double)
    @property
    def output_bytes(self):return self._tensor_size*sizeof(double)
    @property
    def evaluation_bytes(self):return self._evaluation_size*sizeof(double)
    @property
    def memory_bytes(self):return self._memory_bytes
