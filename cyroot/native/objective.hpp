#ifndef CYROOT_NATIVE_OBJECTIVE_HPP
#define CYROOT_NATIVE_OBJECTIVE_HPP

#include "tensor_ops.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot { namespace native {

// Compensated exponent-scaled accumulation extends intermediate range, not
// precision. Inputs must be finite; underflow is rounded only at final export.
// This small value type performs no allocation and is reusable by transforms.
class ObjectiveAccumulator {
    tensor_detail::ScaledTerm total_{0.0, 0}, correction_{0.0, 0};
    bool finite_ = true;
public:
    void add_scaled(double mantissa, int exponent) noexcept {
        using namespace tensor_detail;
        if (!std::isfinite(mantissa)) { finite_ = false; return; }
        ScaledTerm next, error, combined, discarded;
        two_sum(total_, normalized(mantissa, exponent), next, error);
        two_sum(correction_, error, combined, discarded);
        total_ = next;
        correction_ = combined;
    }
    void add_product(double a, double b) noexcept { add_triple(a, b, 1.0); }
    void add_triple(double a, double b, double c) noexcept {
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) {
            finite_ = false;
            return;
        }
        const auto term = tensor_detail::triple(a, b, c);
        add_scaled(term.mantissa, term.exponent);
    }
    // Returns0 on finite final value,1 on nonfinite input/final overflow,
    //2 on null output. Failure leaves output unchanged.
    int value(double* out) const noexcept {
        if (out == nullptr) return 2;
        if (!finite_) return 1;
        tensor_detail::ScaledTerm rounded, discarded;
        tensor_detail::two_sum(total_, correction_, rounded, discarded);
        const double result = std::ldexp(rounded.mantissa, rounded.exponent);
        if (!std::isfinite(result)) return 1;
        *out = result;
        return 0;
    }
};

inline int objective_value(const double* residual, std::size_t m,
                           double* out) noexcept {
    if (residual == nullptr || out == nullptr || m == 0) return 2;
    ObjectiveAccumulator sum;
    for (std::size_t o = 0; o < m; ++o)
        sum.add_triple(residual[o], residual[o], 0.5);
    return sum.value(out);
}

// Trusted sized/disjoint spans. J[o+m*j], H[o+m*(j+n*k)]. Output is
// gradient[n] or column-major Hessian[n*n]; scratch output may be partial on
// failure. Adapter publishes it only after the entire operation succeeds.
inline int objective_gradient(const double* residual, const double* jacobian,
                               std::size_t n, std::size_t m,
                               double* out) noexcept {
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (residual == nullptr || jacobian == nullptr || out == nullptr ||
        n == 0 || m == 0 || n > maximum/m) return 2;
    for (std::size_t j = 0; j < n; ++j) {
        ObjectiveAccumulator sum;
        for (std::size_t o = 0; o < m; ++o)
            sum.add_product(jacobian[o+m*j], residual[o]);
        const int status = sum.value(out+j);
        if (status != 0) return status;
    }
    return 0;
}

inline int objective_hessian(const double* residual, const double* jacobian,
                              const double* hessian, std::size_t n,
                              std::size_t m, double* out) noexcept {
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (residual == nullptr || jacobian == nullptr || hessian == nullptr ||
        out == nullptr || n == 0 || m == 0 || n > maximum/m ||
        n > maximum/(m*n) || n > maximum/n) return 2;
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < n; ++j) {
            ObjectiveAccumulator sum;
            for (std::size_t o = 0; o < m; ++o) {
                sum.add_product(jacobian[o+m*j], jacobian[o+m*k]);
                sum.add_product(residual[o], hessian[o+m*(j+n*k)]);
            }
            const int status = sum.value(out+j+n*k);
            if (status != 0) return status;
        }
    }
    return 0;
}

}}  // namespace cyroot::native
#endif
