#ifndef CYROOT_NATIVE_TENSOR_OPS_HPP
#define CYROOT_NATIVE_TENSOR_OPS_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot {
namespace native {
namespace tensor_detail {

// A contraction term has three binary64 factors, so its exponent is within
// +/- 3222. At most PTRDIFF_MAX terms can be accumulated: even their worst-case
// growth adds fewer than 64 bits. int therefore has ample exponent range.
struct ScaledTerm {
    double mantissa;
    int exponent;
};

inline ScaledTerm normalized(double mantissa, int exponent) noexcept {
    if (mantissa == 0.0) return {0.0, 0};
    int shift = 0;
    const double value = std::frexp(mantissa, &shift);
    return {value, exponent + shift};
}

inline ScaledTerm triple(double a, double b, double c) noexcept {
    if (a == 0.0 || b == 0.0 || c == 0.0) return {0.0, 0};
    int ea = 0, eb = 0, ec = 0;
    const double ma = std::frexp(a, &ea);
    const double mb = std::frexp(b, &eb);
    const double mc = std::frexp(c, &ec);
    return normalized((ma * mb) * mc, ea + eb + ec);
}

inline void two_sum(ScaledTerm a, ScaledTerm b,
                    ScaledTerm& high, ScaledTerm& low) noexcept {
    if (a.mantissa == 0.0) { high = b; low = {0.0, 0}; return; }
    if (b.mantissa == 0.0) { high = a; low = {0.0, 0}; return; }
    const bool a_larger = a.exponent > b.exponent ||
        (a.exponent == b.exponent && std::fabs(a.mantissa) >= std::fabs(b.mantissa));
    const ScaledTerm large = a_larger ? a : b;
    const ScaledTerm small = a_larger ? b : a;
    const int gap = large.exponent - small.exponent;
    if (gap > std::numeric_limits<double>::digits + 1) {
        high = large;
        low = small;
        return;
    }
    const double aligned = std::ldexp(small.mantissa, -gap);
    const double sum = large.mantissa + aligned;
    const double remainder = aligned - (sum - large.mantissa);
    high = normalized(sum, large.exponent);
    low = normalized(remainder, large.exponent);
}

inline void poison(double* out, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        out[i] = std::numeric_limits<double>::quiet_NaN();
}

}  // namespace tensor_detail

// Trusted disjoint valid spans H[m*n*n], direction[n], out[m]. Residual
// Hessian layout H[residual + m*(input_j+n*input_k)]. All ordered input pairs
// are summed; no symmetry assumption is made. No allocation or exceptions.
// Status: 0 success, 1 nonfinite input/final result, 2 invalid dimension.
// Invalid dimension leaves spans untouched; other failures poison all output.
// Compensated exponent-scaled summation extends range, not precision. Callback
// errors and cancellation in the supplied derivatives remain the caller's task.
inline int contract_rectangular_hessian(const double* hessian,
                                         const double* direction, std::size_t n,
                                         std::size_t m, double* out) noexcept {
    using namespace tensor_detail;
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || m == 0 || n > maximum / m) return 2;
    const auto matrix_size = m * n;
    if (n > maximum / matrix_size) return 2;
    poison(out, m);
    for (std::size_t j = 0; j < n; ++j)
        if (!std::isfinite(direction[j])) return 1;
    for (std::size_t i = 0; i < m; ++i) {
        ScaledTerm total{0.0, 0}, correction{0.0, 0};
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                const double h = hessian[i + m * (j + n * k)];
                if (!std::isfinite(h)) { poison(out, m); return 1; }
                const auto term = triple(h, direction[j], direction[k]);
                ScaledTerm next, error, combined, discarded;
                two_sum(total, term, next, error);
                two_sum(correction, error, combined, discarded);
                total = next;
                correction = combined;
            }
        }
        ScaledTerm rounded, discarded;
        two_sum(total, correction, rounded, discarded);
        const double result = std::ldexp(rounded.mantissa, rounded.exponent);
        if (!std::isfinite(result)) { poison(out, m); return 1; }
        out[i] = result;
    }
    return 0;
}

// Square compatibility entry point; the rectangular loop owns the arithmetic.
inline int contract_hessian(const double* hessian, const double* direction,
                             std::size_t n, double* out) noexcept {
    return contract_rectangular_hessian(hessian, direction, n, n, out);
}

// A[o,k] = sum_j H[o,j,k]*direction[j], column-major out[o+m*k].
// Trusted disjoint H[m*n*n],direction[n],out[m*n]. Same statuses and failure
// guarantees as contract_rectangular_hessian, including whole-matrix poisoning.
inline int contract_rectangular_hessian_direction(const double* hessian,
                                                   const double* direction,
                                                   std::size_t n, std::size_t m,
                                                   double* out) noexcept {
    using namespace tensor_detail;
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || m == 0 || n > maximum / m) return 2;
    const auto matrix_size = m * n;
    if (n > maximum / matrix_size) return 2;
    poison(out, matrix_size);
    for (std::size_t j = 0; j < n; ++j)
        if (!std::isfinite(direction[j])) return 1;
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < m; ++i) {
            ScaledTerm total{0.0, 0}, correction{0.0, 0};
            for (std::size_t j = 0; j < n; ++j) {
                const double h = hessian[i + m * (j + n * k)];
                if (!std::isfinite(h)) { poison(out, matrix_size); return 1; }
                const auto term = triple(h, direction[j], 1.0);
                ScaledTerm next, error, combined, discarded;
                two_sum(total, term, next, error);
                two_sum(correction, error, combined, discarded);
                total = next;
                correction = combined;
            }
            ScaledTerm rounded, discarded;
            two_sum(total, correction, rounded, discarded);
            const double value = std::ldexp(rounded.mantissa, rounded.exponent);
            if (!std::isfinite(value)) { poison(out, matrix_size); return 1; }
            out[i + m * k] = value;
        }
    }
    return 0;
}

inline int contract_hessian_direction(const double* hessian,
                                       const double* direction, std::size_t n,
                                       double* out) noexcept {
    return contract_rectangular_hessian_direction(hessian, direction, n, n, out);
}

// Compute -(a+b)/2 without overflowing a+b or prematurely rounding each
// subnormal half. Inputs must be finite; final genuine underflow may round0.
inline double negative_half_sum(double a, double b) noexcept {
    using namespace tensor_detail;
    if (!std::isfinite(a) || !std::isfinite(b))
        return std::numeric_limits<double>::quiet_NaN();
    int ea = 0, eb = 0;
    const ScaledTerm left{std::frexp(a, &ea), ea};
    const ScaledTerm right{std::frexp(b, &eb), eb};
    ScaledTerm rounded, remainder;
    two_sum(left, right, rounded, remainder);
    return -std::ldexp(rounded.mantissa, rounded.exponent - 1);
}

// Column-major matrix-vector multiply, trusted disjoint matrix[m*n],
// direction[n],out[m]. Same dimension/preflight/poison statuses as contraction.
inline int rectangular_matrix_vector(const double* matrix, const double* direction,
                                      std::size_t n, std::size_t m,
                                      double* out) noexcept {
    using namespace tensor_detail;
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || m == 0 || n > maximum / m) return 2;
    poison(out, m);
    for (std::size_t j = 0; j < n; ++j)
        if (!std::isfinite(direction[j])) return 1;
    for (std::size_t i = 0; i < m; ++i) {
        ScaledTerm total{0.0, 0}, correction{0.0, 0};
        for (std::size_t j = 0; j < n; ++j) {
            const double value = matrix[i + m * j];
            if (!std::isfinite(value)) { poison(out, m); return 1; }
            const auto term = triple(value, direction[j], 1.0);
            ScaledTerm next, error, combined, discarded;
            two_sum(total, term, next, error);
            two_sum(correction, error, combined, discarded);
            total = next;
            correction = combined;
        }
        ScaledTerm rounded, discarded;
        two_sum(total, correction, rounded, discarded);
        const double value = std::ldexp(rounded.mantissa, rounded.exponent);
        if (!std::isfinite(value)) { poison(out, m); return 1; }
        out[i] = value;
    }
    return 0;
}

inline int matrix_vector(const double* matrix, const double* direction,
                          std::size_t n, double* out) noexcept {
    return rectangular_matrix_vector(matrix, direction, n, n, out);
}

}  // namespace native
}  // namespace cyroot
#endif
