#ifndef CYROOT_NATIVE_HOUSEHOLDER_HPP
#define CYROOT_NATIVE_HOUSEHOLDER_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cyroot {
namespace native {

// A finite coefficient is mantissa * 2**exponent. Nonzero mantissas are
// normalized to [0.5, 1) in magnitude; zero has exponent zero. These entries
// belong to caller-provided scratch, not to a Python object or an allocator.
struct ScaledCoefficient {
    double mantissa;
    std::int64_t exponent;
};

namespace householder_detail {

inline bool add_exponents(std::int64_t left, std::int64_t right,
                          std::int64_t& out) noexcept {
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((right > 0 && left > maximum - right) ||
        (right < 0 && left < minimum - right)) return false;
    out = left + right;
    return true;
}

inline bool subtract_exponents(std::int64_t left, std::int64_t right,
                               std::int64_t& out) noexcept {
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    if ((right > 0 && left < minimum + right) ||
        (right < 0 && left > maximum + right)) return false;
    out = left - right;
    return true;
}

inline ScaledCoefficient from_double(double value) noexcept {
    int exponent = 0;
    const double mantissa = std::frexp(value, &exponent);
    return {mantissa, exponent};
}

inline bool normalize(double mantissa, std::int64_t exponent,
                      ScaledCoefficient& out) noexcept {
    if (mantissa == 0.0) {
        out = {0.0, 0};
        return true;
    }
    int shift = 0;
    out.mantissa = std::frexp(mantissa, &shift);
    return add_exponents(exponent, shift, out.exponent);
}

inline bool multiply(const ScaledCoefficient& left,
                     const ScaledCoefficient& right,
                     ScaledCoefficient& out) noexcept {
    if (left.mantissa == 0.0 || right.mantissa == 0.0) {
        out = {0.0, 0};
        return true;
    }
    std::int64_t exponent;
    return add_exponents(left.exponent, right.exponent, exponent) &&
           normalize(left.mantissa * right.mantissa, exponent, out);
}

// The caller has checked the denominator. Normalized mantissa division stays
// in (0.5, 2) in magnitude, even when the full quotient is outside binary64.
inline bool divide(const ScaledCoefficient& numerator,
                   const ScaledCoefficient& denominator,
                   ScaledCoefficient& out) noexcept {
    if (numerator.mantissa == 0.0) {
        out = {0.0, 0};
        return true;
    }
    std::int64_t exponent;
    return subtract_exponents(numerator.exponent, denominator.exponent,
                              exponent) &&
           normalize(numerator.mantissa / denominator.mantissa, exponent, out);
}

// FastTwoSum after exponent alignment. If exponents are far apart, the smaller
// operand cannot affect the rounded high part: retain it directly as the low
// part, avoiding an artificial underflow during alignment. This compensation
// improves summation but does not make ill-conditioned cancellation exact.
inline bool two_sum(const ScaledCoefficient& left,
                    const ScaledCoefficient& right,
                    ScaledCoefficient& high,
                    ScaledCoefficient& low) noexcept {
    if (left.mantissa == 0.0) {
        high = right;
        low = {0.0, 0};
        return true;
    }
    if (right.mantissa == 0.0) {
        high = left;
        low = {0.0, 0};
        return true;
    }
    const bool left_larger = left.exponent > right.exponent ||
        (left.exponent == right.exponent &&
         std::fabs(left.mantissa) >= std::fabs(right.mantissa));
    const auto& large = left_larger ? left : right;
    const auto& small = left_larger ? right : left;
    // Compare without subtracting exponents, which could overflow int64_t.
    std::int64_t nearby;
    if (!add_exponents(small.exponent, 54, nearby) ||
        large.exponent <= nearby) {
        // Overflow in small.exponent+54 implies the difference is <54.
        const int distance = static_cast<int>(large.exponent - small.exponent);
        const double aligned = std::ldexp(small.mantissa, -distance);
        const double sum = large.mantissa + aligned;
        const double residual = aligned - (sum - large.mantissa);
        return normalize(sum, large.exponent, high) &&
               normalize(residual, large.exponent, low);
    }
    high = large;
    low = small;
    return true;
}

inline bool accumulate(const ScaledCoefficient& term,
                       ScaledCoefficient& sum,
                       ScaledCoefficient& correction) noexcept {
    ScaledCoefficient next, error, combined, discarded;
    if (!two_sum(sum, term, next, error) ||
        !two_sum(correction, error, combined, discarded)) return false;
    sum = next;
    correction = combined;
    return true;
}

inline bool to_double(const ScaledCoefficient& value, double& out) noexcept {
    if (value.mantissa == 0.0) {
        out = 0.0;
        return true;
    }
    if (value.exponent > std::numeric_limits<double>::max_exponent) return false;
    // For binary64 an exponent below -1074 is strictly below half of the
    // smallest subnormal. Underflow of the final step is a valid zero result.
    if (value.exponent < std::numeric_limits<double>::min_exponent -
                         std::numeric_limits<double>::digits) {
        out = std::copysign(0.0, value.mantissa);
        return true;
    }
    out = std::ldexp(value.mantissa, static_cast<int>(value.exponent));
    return std::isfinite(out);
}

}  // namespace householder_detail

// Runtime-order Householder increment: b_(d-1)/b_d, where b is the reciprocal
// Taylor series of f at the current point. derivatives contains raw f..f^(d).
// Arithmetic assumes IEEE binary64 with its usual rounding (no fast-math).
// Trusted disjoint spans: derivatives has order+1 doubles, workspace has at
// least capacity entries, and out has one double. No allocation or exceptions.
// Status: 0 success, 1 nonfinite arithmetic/input, 2 invalid size/breakdown.
inline int householder_step(const double* derivatives, std::size_t order,
                            ScaledCoefficient* workspace, std::size_t capacity,
                            double* out) noexcept {
    using namespace householder_detail;
    *out = std::numeric_limits<double>::quiet_NaN();
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (order == 0 || order > maximum / 2 - 1) return 2;
    const auto length = order + 1;
    const auto required = 2 * length;
    // No real object can supply a span whose byte distance exceeds PTRDIFF_MAX.
    const auto maximum_entries = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(ScaledCoefficient);
    if (required > maximum_entries || capacity < required) return 2;
    if (!std::isfinite(derivatives[0])) return 1;
    if (derivatives[0] == 0.0) {
        *out = 0.0;
        return 0;
    }
    for (std::size_t k = 1; k <= order; ++k) {
        if (!std::isfinite(derivatives[k])) return 1;
    }

    auto* const a = workspace;
    auto* const b = workspace + length;
    const auto one = from_double(1.0);
    auto inverse_factorial = one;
    a[0] = from_double(derivatives[0]);
    for (std::size_t k = 1; k <= order; ++k) {
        ScaledCoefficient next;
        if (!divide(inverse_factorial, from_double(static_cast<double>(k)), next)) return 1;
        inverse_factorial = next;
        if (!multiply(from_double(derivatives[k]), inverse_factorial, a[k])) return 1;
    }
    if (!divide(one, a[0], b[0])) return 1;
    for (std::size_t n = 1; n <= order; ++n) {
        ScaledCoefficient sum{0.0, 0}, correction{0.0, 0};
        for (std::size_t k = 1; k <= n; ++k) {
            ScaledCoefficient term;
            if (!multiply(a[k], b[n-k], term) ||
                !accumulate(term, sum, correction)) return 1;
        }
        ScaledCoefficient total, discarded;
        if (!two_sum(sum, correction, total, discarded)) return 1;
        total.mantissa = -total.mantissa;
        if (!divide(total, a[0], b[n])) return 1;
    }
    if (b[order].mantissa == 0.0) return 2;
    ScaledCoefficient ratio;
    double step;
    if (!divide(b[order-1], b[order], ratio) || !to_double(ratio, step)) return 1;
    *out = step;
    return 0;
}

}  // namespace native
}  // namespace cyroot

#endif  // CYROOT_NATIVE_HOUSEHOLDER_HPP
