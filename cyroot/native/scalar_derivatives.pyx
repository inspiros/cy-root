# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Scalar finite differences using rounded displacements and stack scratch."""
from libc.float cimport DBL_EPSILON, DBL_MIN
from libc.math cimport cbrt, fabs, fmax, isfinite, NAN


cdef enum:
    FORWARD = 0
    BACKWARD = 1
    CENTRAL = 2


cdef class ScalarDerivativePlan:
    """Immutable finite-difference configuration for a scalar residual.

    ``step=None`` recomputes a relative step at each point. An explicit signed
    finite nonzero scalar is an absolute step. Evaluation belongs to Newton;
    the plan stores no residual callback, workspace, or sampled values.
    """

    def __init__(self, scheme='central', step=None):
        cdef int scheme_code
        cdef double step_value = 0.
        if self._initialized or self._constructing:
            raise RuntimeError('Scalar derivative plans cannot be reinitialized.')
        self._constructing = True
        try:
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
            if step is not None:
                if isinstance(step, (str, bytes, bytearray)):
                    raise TypeError('step must be a finite nonzero scalar.')
                try:
                    len(step)
                except TypeError:
                    pass
                else:
                    raise TypeError('step must be a finite nonzero scalar.')
                step_value = step
                if not isfinite(step_value) or step_value == 0.:
                    raise ValueError('Explicit step must be finite and nonzero.')
            self._scheme = scheme_code
            self._step = step_value
            self._automatic = step is None
            self._initialized = True
        finally:
            self._constructing = False

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
    def sample_count(self):
        return 2 if self._scheme == CENTRAL else 1

    cdef int compute(self, double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil:
        cdef double h, plus, minus = 0., a, b = 0., scale, a_scaled, b_scaled
        cdef double plus_weight = 0., minus_weight = 0.
        cdef double fplus = NAN, fminus = NAN
        cdef double difference, plus_secant, minus_secant
        cdef double term_plus, term_minus, value
        cdef int code
        out[0] = NAN
        callback_code[0] = 0
        if not self._initialized:
            return SCALAR_DIFF_BREAKDOWN
        if not isfinite(x) or not isfinite(f0):
            return SCALAR_DIFF_NONFINITE
        if self._automatic:
            if self._scheme == CENTRAL:
                h = fmax(1., fabs(x)) * cbrt(DBL_EPSILON)
            else:
                h = fmax(1., fabs(x)) * 1.4901161193847656e-8
        else:
            h = self._step
        if not isfinite(h):
            return SCALAR_DIFF_NONFINITE

        # Validate every required coordinate and actual rounded displacement
        # before any sample. Collapsed points never receive silent repairs.
        if self._scheme == BACKWARD:
            plus = x - h
        else:
            plus = x + h
        a = plus - x
        if not isfinite(plus) or not isfinite(a):
            return SCALAR_DIFF_NONFINITE
        if a == 0.:
            return SCALAR_DIFF_BREAKDOWN
        if self._scheme == CENTRAL:
            minus = x - h
            b = minus - x
            if not isfinite(minus) or not isfinite(b):
                return SCALAR_DIFF_NONFINITE
            if b == 0.:
                return SCALAR_DIFF_BREAKDOWN
            scale = fmax(fabs(a), fabs(b))
            a_scaled = a / scale
            b_scaled = b / scale
            plus_weight = b_scaled / (b_scaled - a_scaled)
            minus_weight = a_scaled / (a_scaled - b_scaled)
            if not isfinite(plus_weight) or not isfinite(minus_weight):
                return SCALAR_DIFF_NONFINITE

        calls[0] += 1
        code = function(plus, &fplus, context)
        if code != 0:
            callback_code[0] = code
            return SCALAR_DIFF_CALLBACK_FAILURE
        if not isfinite(fplus):
            return SCALAR_DIFF_NONFINITE
        if self._scheme == CENTRAL:
            calls[0] += 1
            code = function(minus, &fminus, context)
            if code != 0:
                callback_code[0] = code
                return SCALAR_DIFF_CALLBACK_FAILURE
            if not isfinite(fminus):
                return SCALAR_DIFF_NONFINITE

        difference = fplus - f0
        if not isfinite(difference):
            return SCALAR_DIFF_NONFINITE
        plus_secant = difference / a
        if not isfinite(plus_secant):
            return SCALAR_DIFF_NONFINITE
        if self._scheme == CENTRAL:
            difference = fminus - f0
            if not isfinite(difference):
                return SCALAR_DIFF_NONFINITE
            minus_secant = difference / b
            if not isfinite(minus_secant):
                return SCALAR_DIFF_NONFINITE
            term_plus = plus_weight * plus_secant
            term_minus = minus_weight * minus_secant
            value = term_plus + term_minus
            if (not isfinite(term_plus) or not isfinite(term_minus) or
                    not isfinite(value)):
                return SCALAR_DIFF_NONFINITE
            out[0] = value
        else:
            out[0] = plus_secant
        return SCALAR_DIFF_OK


cdef class ScalarSecondDerivativePlan:
    """Immutable three-point finite differences for the second derivative.

    All schemes sample the residual twice and reuse its value at the current
    point. Central differences are second order on symmetric spacing; forward
    and backward differences are first order. Rounded displacements determine
    the interpolating quadratic, including when the spacing is unequal.
    """

    def __init__(self, scheme='central', step=None):
        cdef int scheme_code
        cdef double step_value = 0.
        if self._initialized or self._constructing:
            raise RuntimeError('Scalar second derivative plans cannot be reinitialized.')
        self._constructing = True
        try:
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
            if step is not None:
                if isinstance(step, (str, bytes, bytearray)):
                    raise TypeError('step must be a finite nonzero scalar.')
                try:
                    len(step)
                except TypeError:
                    pass
                else:
                    raise TypeError('step must be a finite nonzero scalar.')
                step_value = step
                if not isfinite(step_value) or step_value == 0.:
                    raise ValueError('Explicit step must be finite and nonzero.')
            self._scheme = scheme_code
            self._step = step_value
            self._automatic = step is None
            self._initialized = True
        finally:
            self._constructing = False

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
    def sample_count(self):
        return 2

    cdef int compute(self, double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil:
        cdef double h, first, second, a, b, separation
        cdef double first_value = NAN, second_value = NAN
        cdef double difference, first_secant, second_secant, quotient, value
        cdef double first_term, second_term
        cdef int code
        out[0] = NAN
        callback_code[0] = 0
        if not self._initialized:
            return SCALAR_DIFF_BREAKDOWN
        if not isfinite(x) or not isfinite(f0):
            return SCALAR_DIFF_NONFINITE
        if self._automatic:
            if self._scheme == CENTRAL:
                h = fmax(1., fabs(x)) * 0.0001220703125
            else:
                h = fmax(1., fabs(x)) * cbrt(DBL_EPSILON)
        else:
            h = self._step
        if not isfinite(h):
            return SCALAR_DIFF_NONFINITE
        if self._scheme == CENTRAL:
            first, second = x + h, x - h
        elif self._scheme == FORWARD:
            first, second = x + h, x + 2. * h
        else:
            first, second = x - h, x - 2. * h
        a, b = first - x, second - x
        separation = a - b
        # Preflight the entire stencil before the first callback. In particular,
        # two one-sided nodes can round to the same float away from the base.
        if (not isfinite(first) or not isfinite(second) or
                not isfinite(a) or not isfinite(b) or not isfinite(separation)):
            return SCALAR_DIFF_NONFINITE
        if a == 0. or b == 0. or separation == 0.:
            return SCALAR_DIFF_BREAKDOWN

        calls[0] += 1
        code = function(first, &first_value, context)
        if code != 0:
            callback_code[0] = code
            return SCALAR_DIFF_CALLBACK_FAILURE
        if not isfinite(first_value):
            return SCALAR_DIFF_NONFINITE
        calls[0] += 1
        code = function(second, &second_value, context)
        if code != 0:
            callback_code[0] = code
            return SCALAR_DIFF_CALLBACK_FAILURE
        if not isfinite(second_value):
            return SCALAR_DIFF_NONFINITE

        difference = first_value - f0
        if not isfinite(difference):
            return SCALAR_DIFF_NONFINITE
        first_secant = difference / a
        difference = second_value - f0
        if not isfinite(difference):
            return SCALAR_DIFF_NONFINITE
        second_secant = difference / b
        if not isfinite(first_secant) or not isfinite(second_secant):
            return SCALAR_DIFF_NONFINITE
        difference = first_secant - second_secant
        if isfinite(difference):
            # Keep subtraction before division for cancellation accuracy, then
            # divide before doubling for range. For subnormal quotients, double
            # first so that premature underflow does not discard the result.
            quotient = difference / separation
            if fabs(quotient) < DBL_MIN:
                value = (2. * difference) / separation
            else:
                value = 2. * quotient
        else:
            # Opposite large finite secants can overflow on subtraction even
            # when the second derivative is finite. Scale by separation first.
            first_term = first_secant / separation
            second_term = second_secant / separation
            if not isfinite(first_term) or not isfinite(second_term):
                return SCALAR_DIFF_NONFINITE
            quotient = first_term - second_term
            value = 2. * quotient
        if not isfinite(quotient) or not isfinite(value):
            return SCALAR_DIFF_NONFINITE
        out[0] = value
        return SCALAR_DIFF_OK
