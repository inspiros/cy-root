# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Homeier's two-point harmonic-mean Newton method."""
from libc.math cimport fabs, isfinite, NAN
from cyroot._stopping cimport (
    advance_x, residual_met, RUNNING, NONFINITE, BREAKDOWN,
    ZERO_DERIVATIVE, CALLBACK_FAILURE,
)
from .scalar_derivatives cimport (
    ScalarResidualFunction, SCALAR_DIFF_CALLBACK_FAILURE,
    SCALAR_DIFF_NONFINITE, SCALAR_DIFF_BREAKDOWN,
)


cdef class Homeier(Newton):
    """Resumable Homeier with analytic or native numerical derivatives.

    Inherits Newton's constructor and copied state/history interface. Each
    iteration takes a Newton predictor y, then averages the inverse
    derivatives at x and y. A predictor already meeting residual tolerance
    is accepted without an unnecessary derivative or correction evaluation.
    """
    cdef int _derivative_at(self, double x, double f_x, double* out) noexcept nogil:
        cdef int status, code = 0
        if self._plan is None:
            return self._evaluate(True, x, out)
        self._state.df_calls += 1
        out[0] = NAN
        status = self._plan.compute(
            x, f_x, out, <ScalarResidualFunction>self._function._function,
            self._function._context, &self._state.f_calls, &code)
        if status == SCALAR_DIFF_CALLBACK_FAILURE:
            self._state.callback_code = code
            self._state.callback_stage = 2
            self._stop.reason = CALLBACK_FAILURE
        elif status == SCALAR_DIFF_NONFINITE:
            self._stop.reason = NONFINITE
        elif status == SCALAR_DIFF_BREAKDOWN:
            self._stop.reason = BREAKDOWN
        return status

    cdef void _step(self) noexcept nogil:
        cdef double predictor, f_predictor, df_predictor
        cdef double delta, correction, candidate, value, precision, quotient
        if self._stop.reason != RUNNING:
            return
        delta = -self._state.f_root / self._state.df_root
        predictor = self._state.root + delta
        if not isfinite(delta) or not isfinite(predictor):
            self._stop.reason = NONFINITE
            self._record()
            return
        self._state.iters += 1
        if self._evaluate(False, predictor, &f_predictor) != 0:
            self._record()
            return
        if not isfinite(f_predictor):
            self._stop.reason = NONFINITE
            self._record()
            return
        if residual_met(&self._stop, fabs(f_predictor)):
            candidate, value = predictor, f_predictor
        else:
            if self._derivative_at(predictor, f_predictor, &df_predictor) != 0:
                self._record()
                return
            if not isfinite(df_predictor) or df_predictor == 0.:
                self._stop.reason = NONFINITE if not isfinite(df_predictor) else ZERO_DERIVATIVE
                self._record()
                return
            # Ratio form avoids squaring/scaling derivative products.
            quotient = self._state.f_root / df_predictor
            # Preserve subnormal numerators when the quotient is finite. If
            # only the full quotient overflows, its half may remain finite.
            if isfinite(quotient):
                correction = 0.5 * delta - 0.5 * quotient
            else:
                correction = 0.5 * delta - (0.5 * self._state.f_root) / df_predictor
            candidate = self._state.root + correction
            if not isfinite(correction) or not isfinite(candidate):
                self._stop.reason = NONFINITE
                self._record()
                return
            if self._evaluate(False, candidate, &value) != 0:
                self._record()
                return
            if not isfinite(value):
                self._stop.reason = NONFINITE
                self._record()
                return
        precision = fabs(candidate - self._state.root)
        if not isfinite(precision):
            self._stop.reason = NONFINITE
            self._record()
            return
        self._state.root, self._state.f_root = candidate, value
        self._state.df_root = NAN
        self._state.error, self._state.precision = fabs(value), precision
        advance_x(&self._stop, fabs(candidate))
        self._check()
        if self._stop.reason == RUNNING:
            if self._derivative_at(candidate, value, &df_predictor) == 0:
                if not isfinite(df_predictor):
                    self._stop.reason = NONFINITE
                else:
                    self._state.df_root = df_predictor
                    if df_predictor == 0.:
                        self._stop.reason = ZERO_DERIVATIVE
        self._record()


def homeier(function, derivative, x0, **options):
    """Run Homeier; retain a Homeier instance to inspect steps/history."""
    return Homeier(function, derivative, x0, **options).run()
