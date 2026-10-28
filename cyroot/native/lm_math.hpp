#ifndef CYROOT_NATIVE_LM_MATH_HPP
#define CYROOT_NATIVE_LM_MATH_HPP

#include "objective.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace cyroot { namespace native {

// LM numerical helpers use trusted, sized, disjoint caller-owned spans.
// All matrices are column-major. No allocation, exceptions, or Python calls.
// Status0 succeeds;1 denotes nonfinite data or arithmetic range failure;
//2 denotes invalid scalar configuration. Scratch outputs may be partial on
// failure; the caller publishes only complete accepted solver states.
// The caller has already validated n,m and every buffer extent.
inline int lm_prepare(const double* residual, const double* jacobian,
                      std::size_t n, std::size_t m, double damping,
                      double* scales, double* augmented, double* rhs,
                      double* residual_scale, double* gradient_norm) noexcept {
    if (!(damping > 0.0) || !std::isfinite(damping)) return 2;
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isfinite(residual[i])) return 1;
        scale = std::max(scale, std::fabs(residual[i]));
    }
    // An exact zero residual is normally handled by the solver before here.
    // A unit scale nevertheless keeps this helper defined at a root.
    if (scale == 0.0) scale = 1.0;
    const std::size_t rows = m + n;
    const double root_damping = std::sqrt(damping);
    double gradient = 0.0;
    for (std::size_t i = 0; i < m; ++i) rhs[i] = -residual[i] / scale;
    for (std::size_t i = 0; i < n; ++i) rhs[m+i] = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double column_scale = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const double value = jacobian[i+m*j];
            if (!std::isfinite(value)) return 1;
            column_scale = std::max(column_scale, std::fabs(value));
        }
        if (column_scale == 0.0) column_scale = 1.0;
        scales[j] = column_scale;
        ObjectiveAccumulator dot;
        for (std::size_t i = 0; i < m; ++i) {
            const double value = jacobian[i+m*j] / column_scale;
            augmented[i+rows*j] = value;
            dot.add_product(value, -rhs[i]);
        }
        for (std::size_t i = 0; i < n; ++i)
            augmented[m+i+rows*j] = i == j ? root_damping : 0.0;
        double component;
        if (dot.value(&component) != 0) return 1;
        gradient = std::max(gradient, std::fabs(component));
    }
    // A=J/column_scale, b=-F/scale. Solve [A;sqrt(lambda)I]z=[b;0].
    // Stationarity uses ||A^T(F/scale)||_infinity, without forming J^T J.
    *residual_scale = scale;
    *gradient_norm = gradient;
    return 0;
}

inline int lm_trial(const double* point, const double* normalized_step,
                    const double* scales, std::size_t n, double residual_scale,
                    double* trial, double* displacement) noexcept {
    if (!(residual_scale > 0.0) || !std::isfinite(residual_scale)) return 2;
    int es = 0;
    const double ms = std::frexp(residual_scale, &es);
    for (std::size_t j = 0; j < n; ++j) {
        if (!std::isfinite(point[j]) || !std::isfinite(normalized_step[j]) ||
            !(scales[j] > 0.0) || !std::isfinite(scales[j])) return 1;
        int ez = 0, ed = 0;
        const double mz = std::frexp(normalized_step[j], &ez);
        const double md = std::frexp(scales[j], &ed);
        // Delta x = scale*z/column_scale; do not overflow scale/column_scale
        // before a small z has a chance to restore the representable range.
        const double step = std::ldexp(ms*mz/md, es+ez-ed);
        if (!std::isfinite(step)) return 1;
        const double next = point[j] + step;
        if (!std::isfinite(next)) return 1;
        const double actual_step = next - point[j];
        if (!std::isfinite(actual_step)) return 1;
        trial[j] = next;
        // Acceptance must evaluate the model at the displacement that survives
        // coordinate rounding, rather than the ideal linear-solver step.
        displacement[j] = actual_step;
    }
    return 0;
}

inline int lm_reductions(const double* residual, const double* jacobian,
                         const double* trial_residual,
                         const double* displacement, std::size_t n,
                         std::size_t m, double residual_scale,
                         double* actual, double* predicted) noexcept {
    if (!(residual_scale > 0.0) || !std::isfinite(residual_scale)) return 2;
    int es = 0;
    const double ms = std::frexp(residual_scale, &es);
    ObjectiveAccumulator actual_sum, predicted_sum;
    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isfinite(residual[i]) || !std::isfinite(trial_residual[i]))
            return 1;
        int ef = 0, et = 0;
        const double mf = std::frexp(residual[i], &ef);
        const double mt = std::frexp(trial_residual[i], &et);
        // frexp(0) has exponent0, which must not force subnormal nonzero
        // operands back into their unscaled range before multiplication.
        const int common = mf == 0.0 ? et : mt == 0.0 ? ef : std::max(ef, et);
        const double a = std::ldexp(mf, ef-common);
        const double b = std::ldexp(mt, et-common);
        // Factored difference of squares avoids cancellation between already
        // rounded squares. Binary scaling avoids raw residual-square overflow.
        actual_sum.add_scaled(((a-b)*(a+b)/ms)/ms, 2*(common-es));
        ObjectiveAccumulator model_change;
        for (std::size_t j = 0; j < n; ++j) {
            const double derivative = jacobian[i+m*j];
            if (!std::isfinite(derivative) || !std::isfinite(displacement[j]))
                return 1;
            int ej = 0, ex = 0;
            const double mj = std::frexp(derivative, &ej);
            const double mx = std::frexp(displacement[j], &ex);
            model_change.add_scaled(mj*mx/ms, ej+ex-es);
        }
        double q;
        if (model_change.value(&q) != 0) return 1;
        // q=(J*actual_delta_x)/scale. The common 1/2 objective factor cancels
        // in the reduction ratio. -q*(2*F/scale+q) retains tiny improvements.
        predicted_sum.add_triple(q, residual[i]/residual_scale, -2.0);
        predicted_sum.add_triple(q, q, -1.0);
    }
    double actual_value, predicted_value;
    if (actual_sum.value(&actual_value) != 0 ||
        predicted_sum.value(&predicted_value) != 0) return 1;
    *actual = actual_value;
    *predicted = predicted_value;
    return 0;
}

}}  // namespace cyroot::native
#endif
