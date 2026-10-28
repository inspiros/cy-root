# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Rounded-stencil Fornberg weights and native runtime-order differentiation."""
from numbers import Complex, Real
from operator import index
from typing import NamedTuple
from libc.float cimport DBL_EPSILON
from libc.math cimport fabs, fmax, isfinite, pow, NAN
from libcpp.vector cimport vector

from .scalar cimport ScalarCallback
from .scalar import NativeCallbackError
from .scalar_derivatives cimport (
    SCALAR_DIFF_OK, SCALAR_DIFF_NONFINITE, SCALAR_DIFF_BREAKDOWN,
    SCALAR_DIFF_CALLBACK_FAILURE,
)


cdef enum:
    FORWARD = 0
    BACKWARD = 1
    CENTRAL = 2


class ScalarDerivativeEstimate(NamedTuple):
    """Copied derivative estimate and total residual-attempt diagnostics."""
    derivative: float
    f_calls: int
    status: str
    callback_code: int


def _checked_integer(value, name):
    if isinstance(value, bool) or getattr(getattr(value, 'dtype', None), 'kind', None) == 'b':
        raise ValueError(f'{name} must be an integer, not bool.')
    try:
        result = index(value)
    except TypeError:
        raise ValueError(f'{name} must be an integer.') from None
    if result < 0 or result > <size_t>-1:
        raise ValueError(f'{name} must fit a nonnegative size_t.')
    return result


cdef object _status_name(int status):
    if status == SCALAR_DIFF_OK:
        return 'ok'
    if status == SCALAR_DIFF_NONFINITE:
        return 'nonfinite'
    if status == SCALAR_DIFF_BREAKDOWN:
        return 'breakdown'
    return 'callback_failure'


cdef int compute_scalar_difference(const ScalarDifferenceConfig* config,
                     double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code,
                     double* workspace, size_t capacity) noexcept nogil:
    cdef size_t i, j, k, mn, width, expected_nodes
    cdef double* points = workspace
    cdef double* nodes
    cdef double* coefficients
    cdef double h, offset, scale = 0., displacement
    cdef double c1 = 1., c2, c3, c4 = 0., c5, multiplier
    cdef double term1, term2, difference, value, sample = NAN
    cdef double total = 0., correction = 0., term, adjusted, updated
    cdef int code
    out[0] = NAN
    callback_code[0] = 0
    if (config.order == 0 or config.order > (<size_t>-1) - 3 or
            config.scheme < FORWARD or config.scheme > CENTRAL):
        return SCALAR_DIFF_BREAKDOWN
    expected_nodes = config.order + 1 + (config.order % 2 if config.scheme == CENTRAL else 0)
    if (config.nodes != expected_nodes or
            config.nodes > (<size_t>-1) // (config.order + 3)):
        return SCALAR_DIFF_BREAKDOWN
    if config.work != config.nodes * (config.order + 3) or capacity < config.work:
        return SCALAR_DIFF_BREAKDOWN
    width = config.order + 1
    if not isfinite(x) or not isfinite(f0):
        return SCALAR_DIFF_NONFINITE
    nodes = workspace + config.nodes
    coefficients = nodes + config.nodes
    if config.automatic:
        h = fmax(1., fabs(x)) * pow(DBL_EPSILON, 1. / (config.order + (2. if config.scheme == CENTRAL else 1.)))
    else:
        h = config.step
    if not isfinite(h):
        return SCALAR_DIFF_NONFINITE
    points[0], nodes[0] = x, 0.
    for i in range(1, config.nodes):
        if config.scheme == CENTRAL:
            offset = <double>((i + 1) // 2)
            if i % 2 == 0:
                offset = -offset
        else:
            offset = <double>i
            if config.scheme == BACKWARD:
                offset = -offset
        displacement = offset * h
        points[i] = x + displacement
        nodes[i] = points[i] - x
        if (not isfinite(displacement) or not isfinite(points[i]) or
                not isfinite(nodes[i])):
            return SCALAR_DIFF_NONFINITE
        if nodes[i] == 0.:
            return SCALAR_DIFF_BREAKDOWN
        for j in range(i):
            if points[i] == points[j]:
                return SCALAR_DIFF_BREAKDOWN
        scale = fmax(scale, fabs(nodes[i]))
    for i in range(1, config.nodes):
        nodes[i] /= scale
        for j in range(i):
            if nodes[i] == nodes[j]:
                return SCALAR_DIFF_BREAKDOWN
    for i in range(config.nodes * width):
        coefficients[i] = 0.
    coefficients[0] = 1.

    # Fornberg's recurrence at z=0. Stencil coordinates are normalized
    # rounded displacements, not requested integer multiples of h.
    for i in range(1, config.nodes):
        mn = i if i < config.order else config.order
        c2, c5, c4 = 1., c4, nodes[i]
        for j in range(i):
            c3 = nodes[i] - nodes[j]
            if c3 == 0.:
                return SCALAR_DIFF_BREAKDOWN
            c2 *= c3
            if not isfinite(c2) or c2 == 0.:
                return SCALAR_DIFF_NONFINITE
            if j == i - 1:
                multiplier = c1 / c2
                if not isfinite(multiplier):
                    return SCALAR_DIFF_NONFINITE
                for k in range(mn, 0, -1):
                    term1 = <double>k * coefficients[(i - 1) * width + k - 1]
                    term2 = c5 * coefficients[(i - 1) * width + k]
                    difference = term1 - term2
                    value = multiplier * difference
                    if (not isfinite(term1) or not isfinite(term2) or
                            not isfinite(difference) or not isfinite(value)):
                        return SCALAR_DIFF_NONFINITE
                    coefficients[i * width + k] = value
                value = -multiplier * c5 * coefficients[(i - 1) * width]
                if not isfinite(value):
                    return SCALAR_DIFF_NONFINITE
                coefficients[i * width] = value
            for k in range(mn, 0, -1):
                term1 = c4 * coefficients[j * width + k]
                term2 = <double>k * coefficients[j * width + k - 1]
                difference = term1 - term2
                value = difference / c3
                if (not isfinite(term1) or not isfinite(term2) or
                        not isfinite(difference) or not isfinite(value)):
                    return SCALAR_DIFF_NONFINITE
                coefficients[j * width + k] = value
            value = c4 * coefficients[j * width] / c3
            if not isfinite(value):
                return SCALAR_DIFF_NONFINITE
            coefficients[j * width] = value
        c1 = c2

    for i in range(1, config.nodes):
        sample = NAN
        calls[0] += 1
        code = function(points[i], &sample, context)
        if code != 0:
            callback_code[0] = code
            return SCALAR_DIFF_CALLBACK_FAILURE
        if not isfinite(sample):
            return SCALAR_DIFF_NONFINITE
        # Subtract the cached base to remove the constant contribution
        # before weighted summation. Kahan compensation reduces roundoff.
        difference = sample - f0
        term = coefficients[i * width + config.order] * difference
        adjusted = term - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        if (not isfinite(difference) or not isfinite(term) or
                not isfinite(adjusted) or not isfinite(updated) or
                not isfinite(correction)):
            return SCALAR_DIFF_NONFINITE
        total = updated
    for k in range(config.order):
        total /= scale
        if not isfinite(total):
            return SCALAR_DIFF_NONFINITE
    out[0] = total
    return SCALAR_DIFF_OK



cdef class ScalarFiniteDifferencePlan:
    """Immutable runtime-order plan with configurable node and scratch budgets.

    Derivatives are interpolated on actual rounded points. Central stencils
    include the base and symmetric requested offsets; one-sided stencils use
    order+1 points. Raising max_nodes changes the resource limit, not a formula
    table. Large orders may be inaccurate or unrepresentable in binary64.
    """

    def __init__(self, order, scheme='central', step=None, *, max_nodes=65,
                 workspace_bytes=8 * 1024 * 1024):
        cdef int scheme_code
        cdef double step_value = 0.
        if self._initialized or self._constructing:
            raise RuntimeError('Finite difference plans cannot be reinitialized.')
        self._constructing = True
        try:
            degree = _checked_integer(order, 'order')
            limit = _checked_integer(max_nodes, 'max_nodes')
            byte_limit = _checked_integer(workspace_bytes, 'workspace_bytes')
            if degree == 0 or limit == 0:
                raise ValueError('order and max_nodes must be positive.')
            if not isinstance(scheme, str):
                raise ValueError('scheme must be forward, backward or central.')
            normalized = str(scheme)
            if normalized == 'forward':
                scheme_code = FORWARD
            elif normalized == 'backward':
                scheme_code = BACKWARD
            elif normalized == 'central':
                scheme_code = CENTRAL
            else:
                raise ValueError('scheme must be forward, backward or central.')
            nodes = 2 * ((degree + 1) // 2) + 1 if scheme_code == CENTRAL else degree + 1
            work = nodes * (degree + 3)
            if nodes > limit:
                raise ValueError('Requested stencil exceeds max_nodes.')
            if work > byte_limit // sizeof(double) or work > <size_t>-1:
                raise ValueError('Requested scratch exceeds workspace_bytes or native limits.')
            if step is not None:
                if (isinstance(step, (str, bytes, bytearray, bool)) or
                        (isinstance(step, Complex) and not isinstance(step, Real)) or
                        getattr(getattr(step, 'dtype', None), 'kind', None) in ('b', 'c')):
                    raise TypeError('step must be a finite nonzero real scalar.')
                try:
                    len(step)
                except TypeError:
                    pass
                else:
                    raise TypeError('step must be a finite nonzero real scalar.')
                step_value = step
                if not isfinite(step_value) or step_value == 0.:
                    raise ValueError('Explicit step must be finite and nonzero.')
            self._order, self._nodes, self._work = degree, nodes, work
            self._scheme, self._step = scheme_code, step_value
            self._automatic = step is None
            self._initialized = True
        finally:
            self._constructing = False

    @property
    def order(self):
        return self._order

    @property
    def scheme(self):
        if self._scheme == FORWARD:
            return 'forward'
        if self._scheme == BACKWARD:
            return 'backward'
        return 'central'

    @property
    def step(self):
        return None if self._automatic else self._step

    @property
    def stencil_size(self):
        return self._nodes

    @property
    def sample_count(self):
        return self._nodes - 1 if self._nodes else 0

    @property
    def workspace_size(self):
        return self._work

    @property
    def workspace_bytes(self):
        return self._work * sizeof(double)

    cdef void copy_config(self, ScalarDifferenceConfig* config) noexcept nogil:
        config.order, config.nodes, config.work = self._order, self._nodes, self._work
        config.scheme, config.step, config.automatic = self._scheme, self._step, self._automatic

    cdef int compute(self, double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code,
                     double* workspace, size_t capacity) noexcept nogil:
        cdef ScalarDifferenceConfig config
        if not self._initialized:
            out[0] = NAN
            callback_code[0] = 0
            return SCALAR_DIFF_BREAKDOWN
        self.copy_config(&config)
        return compute_scalar_difference(&config, x, f0, out, function, context,
                                         calls, callback_code, workspace, capacity)

    def evaluate(self, function, x, f_x=None):
        """Evaluate with private scratch and a Python/native residual adapter."""
        cdef ScalarCallback adapter
        cdef vector[double] workspace
        cdef double point, base = NAN, value = NAN
        cdef size_t calls = 0
        cdef int code = 0, status = SCALAR_DIFF_NONFINITE
        stage = 'function'
        if not self._initialized:
            raise RuntimeError('Finite difference plan is not initialized.')
        adapter = function if isinstance(function, ScalarCallback) else ScalarCallback(function)
        if adapter._function == NULL:
            raise RuntimeError('Residual callback adapter is not initialized.')
        if adapter._active:
            raise RuntimeError('This callback adapter is already active.')
        adapter._active = True
        adapter._error = None
        try:
            point = x
            if f_x is not None:
                base = f_x
            if isfinite(point):
                workspace.resize(self._work)
                if f_x is None:
                    with nogil:
                        calls += 1
                        code = adapter._function(point, &base, adapter._context)
                    if code != 0:
                        status = SCALAR_DIFF_CALLBACK_FAILURE
                if code == 0 and isfinite(base):
                    stage = 'derivative'
                    with nogil:
                        status = self.compute(
                            point, base, &value, <ScalarResidualFunction>adapter._function,
                            adapter._context, &calls, &code, workspace.data(), workspace.size())
            estimate = ScalarDerivativeEstimate(value, calls, _status_name(status), code)
            if adapter._error is not None:
                error = adapter._error
                adapter._error = None
                raise error
            if status == SCALAR_DIFF_CALLBACK_FAILURE:
                raise NativeCallbackError(code, estimate, stage)
            return estimate
        finally:
            adapter._active = False
