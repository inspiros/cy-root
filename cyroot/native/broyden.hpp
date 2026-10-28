#ifndef CYROOT_NATIVE_BROYDEN_HPP
#define CYROOT_NATIVE_BROYDEN_HPP

#include "tensor_ops.hpp"

namespace cyroot {
namespace native {
namespace broyden_detail {

using tensor_detail::ScaledTerm;
using tensor_detail::normalized;
using tensor_detail::two_sum;

struct Sum {
    ScaledTerm high{0.0, 0}, low{0.0, 0};
    void add(ScaledTerm term) noexcept {
        ScaledTerm next, error, combined, discarded;
        two_sum(high, term, next, error);
        two_sum(low, error, combined, discarded);
        high = next;
        low = combined;
    }
    ScaledTerm value() const noexcept {
        ScaledTerm rounded, discarded;
        two_sum(high, low, rounded, discarded);
        return rounded;
    }
};

inline ScaledTerm product(double a, double b) noexcept {
    return tensor_detail::triple(a, b, 1.0);
}

}  // namespace broyden_detail

// out = matrix + (target-matrix*direction)*direction^T/(direction^T*direction).
// Trusted disjoint valid column-major matrix[n*n], direction[n], target[n],
// out[n*n]. No allocation, Python, exceptions or mutation of input spans.
// 0 success; 1 nonfinite input/final result (all output NaN); 2 invalid checked
// dimension (no writes); 3 zero direction (all output NaN). Scaled compensated
// arithmetic extends intermediate range; it does not increase precision.
inline int broyden_update(const double* matrix, const double* direction,
                          const double* target, std::size_t n,
                          double* out) noexcept {
    using namespace broyden_detail;
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || n > maximum / n) return 2;
    const auto square = n * n;
    tensor_detail::poison(out, square);
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(direction[k]) || !std::isfinite(target[k])) return 1;
    for (std::size_t k = 0; k < square; ++k)
        if (!std::isfinite(matrix[k])) return 1;
    // In one dimension the secant constraint fixes the entire matrix. Use
    // that quotient directly instead of cancelling two large matrix terms.
    if (n == 1) {
        if (direction[0] == 0.0) return 3;
        const double value = target[0] / direction[0];
        if (!std::isfinite(value)) return 1;
        out[0] = value;
        return 0;
    }
    Sum denominator;
    for (std::size_t k = 0; k < n; ++k)
        denominator.add(product(direction[k], direction[k]));
    const auto divisor = denominator.value();
    if (divisor.mantissa == 0.0) return 3;
    for (std::size_t i = 0; i < n; ++i) {
        Sum residual;
        residual.add(normalized(target[i], 0));
        for (std::size_t k = 0; k < n; ++k) {
            auto term = product(matrix[i+n*k], direction[k]);
            term.mantissa = -term.mantissa;
            residual.add(term);
        }
        for (std::size_t j = 0; j < n; ++j) {
            int exponent = 0;
            const double mantissa = std::frexp(direction[j], &exponent);
            Sum entry;
            entry.add(normalized(matrix[i+n*j], 0));
            // Carry both residual components through the final addition:
            // a large -matrix contribution can cancel the original entry,
            // exposing a much smaller target contribution.
            const ScaledTerm components[] = {residual.high, residual.low};
            for (const auto r : components) {
                entry.add(normalized(
                    (r.mantissa * mantissa) / divisor.mantissa,
                    r.exponent + exponent - divisor.exponent));
            }
            const auto result = entry.value();
            out[i+n*j] = std::ldexp(result.mantissa, result.exponent);
            if (!std::isfinite(out[i+n*j])) {
                tensor_detail::poison(out, square);
                return 1;
            }
        }
    }
    return 0;
}

// Row-scaled Klement secant update. For each row, u_j = matrix_ij*direction_j,
// divisor = sum(u_j*u_j), and out_ij = matrix_ij +
// (target_i-sum(u_j))*u_j*matrix_ij/divisor. Same trusted spans, layout,
// statuses and finite-input/output guarantees as broyden_update, except that
// status 3 means ANY row has zero divisor (including a zero scalar matrix).
// Products, divisors and corrections retain binary exponents separately;
// both residual components survive until addition to each original entry.
inline int klement_update(const double* matrix, const double* direction,
                          const double* target, std::size_t n,
                          double* out) noexcept {
    using namespace broyden_detail;
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || n > maximum / n) return 2;
    const auto square = n * n;
    tensor_detail::poison(out, square);
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(direction[k]) || !std::isfinite(target[k])) return 1;
    for (std::size_t k = 0; k < square; ++k)
        if (!std::isfinite(matrix[k])) return 1;
    // A nonzero scalar row fixes the secant quotient exactly. Checking BOTH
    // factors first preserves Klement's breakdown for an initially zero row.
    if (n == 1) {
        if (matrix[0] == 0.0 || direction[0] == 0.0) return 3;
        const double value = target[0] / direction[0];
        if (!std::isfinite(value)) return 1;
        out[0] = value;
        return 0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        Sum denominator, residual;
        residual.add(normalized(target[i], 0));
        for (std::size_t k = 0; k < n; ++k) {
            auto u = product(matrix[i+n*k], direction[k]);
            denominator.add(normalized(u.mantissa * u.mantissa,
                                       2 * u.exponent));
            u.mantissa = -u.mantissa;
            residual.add(u);
        }
        const auto divisor = denominator.value();
        if (divisor.mantissa == 0.0) {
            tensor_detail::poison(out, square);
            return 3;
        }
        for (std::size_t j = 0; j < n; ++j) {
            const double b = matrix[i+n*j];
            const auto u = product(b, direction[j]);
            const auto scaled_b = normalized(b, 0);
            Sum entry;
            entry.add(scaled_b);
            const ScaledTerm components[] = {residual.high, residual.low};
            for (const auto r : components) {
                entry.add(normalized(
                    ((r.mantissa * u.mantissa) * scaled_b.mantissa) /
                        divisor.mantissa,
                    r.exponent + u.exponent + scaled_b.exponent -
                        divisor.exponent));
            }
            const auto result = entry.value();
            out[i+n*j] = std::ldexp(result.mantissa, result.exponent);
            if (!std::isfinite(out[i+n*j])) {
                tensor_detail::poison(out, square);
                return 1;
            }
        }
    }
    return 0;
}

}  // namespace native
}  // namespace cyroot
#endif
