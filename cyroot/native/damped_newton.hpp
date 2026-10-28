#ifndef CYROOT_NATIVE_DAMPED_NEWTON_HPP
#define CYROOT_NATIVE_DAMPED_NEWTON_HPP

#include "bfgs_math.hpp"

namespace cyroot { namespace native {

// Trusted finite, sized spans; J is m-by-n column-major. Normalize by the
// current residual infinity norm, keeping merit squares in a useful range.
// Reuse the existing scaled accumulator rather than forming F^T J first.
inline int newton_merit_slope(const double* f, const double* j, const double* p,
                             std::size_t n, std::size_t m, double scale,
                             double* out) noexcept {
    if (!(scale > 0.0) || !std::isfinite(scale)) return 1;
    int es = 0;
    const double ms = std::frexp(scale, &es);
    ObjectiveAccumulator slope;
    for (std::size_t i = 0; i < m; ++i) {
        ObjectiveAccumulator row;
        for (std::size_t k = 0; k < n; ++k) {
            int ej = 0, ep = 0;
            const double mj = std::frexp(j[i+m*k], &ej);
            const double mp = std::frexp(p[k], &ep);
            row.add_scaled(mj*mp/ms, ej+ep-es);
        }
        double q;
        if (row.value(&q) != 0) return 1;
        slope.add_product(f[i]/scale, q);
    }
    return slope.value(out);
}

}} // namespace cyroot::native
#endif
