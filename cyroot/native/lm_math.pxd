"""Allocation-free scaled Levenberg-Marquardt numerical helpers."""
from libc.stddef cimport size_t

cdef extern from "lm_math.hpp" namespace "cyroot::native" nogil:
    # Trusted sized/disjoint spans, column-major J[m*n], A[(m+n)*n].
    # Status0 success,1 nonfinite/range failure,2 invalid scalar configuration.
    # Scratch outputs may be partial on failure. Callers validate dimensions,
    # capacities and lifetimes before entering these allocation-free helpers.
    int lm_prepare(const double* residual, const double* jacobian,
                   size_t n, size_t m, double damping, double* scales,
                   double* augmented, double* rhs, double* residual_scale,
                   double* gradient_norm) noexcept
    int lm_trial(const double* point, const double* normalized_step,
                 const double* scales, size_t n, double residual_scale,
                 double* trial, double* displacement) noexcept
    # Reductions omit the common1/2 objective factor. Model prediction uses
    # the actual rounded coordinate displacement. Nonpositive prediction is
    # a successfully evaluated model, to be rejected by the caller.
    int lm_reductions(const double* residual, const double* jacobian,
                      const double* trial_residual, const double* displacement,
                      size_t n, size_t m, double residual_scale,
                      double* actual, double* predicted) noexcept
