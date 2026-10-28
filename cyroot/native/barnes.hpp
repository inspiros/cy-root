#ifndef CYROOT_NATIVE_BARNES_HPP
#define CYROOT_NATIVE_BARNES_HPP

#include "broyden.hpp"

namespace cyroot {
namespace native {
namespace barnes_detail {

inline double dot(const double* a, const double* b, std::size_t n) noexcept {
    broyden_detail::Sum sum;
    for (std::size_t k = 0; k < n; ++k)
        sum.add(broyden_detail::product(a[k], b[k]));
    const auto result = sum.value();
    return std::ldexp(result.mantissa, result.exponent);
}

// All vectors here have first been divided by their maximum absolute entry.
// Accepted basis rows have unit norm. Two MGS passes reduce loss of
// orthogonality when the retained steps are nearly linearly dependent.
inline void project(double* row, const double* basis, std::size_t retained,
                    std::size_t n) noexcept {
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t j = 0; j < retained; ++j) {
            const double* q = basis + j*n;
            const double coefficient = dot(row, q, n);
            for (std::size_t k = 0; k < n; ++k)
                row[k] -= coefficient*q[k];
        }
    }
}

}  // namespace barnes_detail

// Trusted disjoint spans: oldest-first history[count*n], step[n], scratch
// basis[n*(n-1)], row[n], and output z[n]. No allocation or input mutation.
// z is a unit vector in the projected step direction, orthogonal to the
// retained numerical basis. Discarded near-dependent history components have
// only approximate protection. Zero/dependent history rows are skipped. No random direction or
// restart is substituted for a collapsed step. Relative dependence threshold
// is 64*epsilon*n, measured against each max-scaled row's original norm.
// 0 success; 1 nonfinite input/result; 2 invalid dimension/count (no writes);
// 3 zero/numerically dependent step. Valid-dimension failures poison all z.
inline int barnes_direction(const double* history, std::size_t count,
                            const double* step, std::size_t n,
                            double* basis, double* row, double* z) noexcept {
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || n > maximum / n || count > n-1) return 2;
    tensor_detail::poison(z, n);
    // Scan every supplied entry before a zero or dependent vector can exit.
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(step[k])) return 1;
    for (std::size_t k = 0; k < count*n; ++k)
        if (!std::isfinite(history[k])) return 1;
    const double threshold = 64.0*std::numeric_limits<double>::epsilon()*n;
    std::size_t retained = 0;
    for (std::size_t j = 0; j <= count; ++j) {
        const bool current = j == count;
        const double* source = current ? step : history+j*n;
        double scale = 0.0;
        for (std::size_t k = 0; k < n; ++k)
            scale = std::fmax(scale, std::fabs(source[k]));
        if (scale == 0.0) {
            if (current) return 3;
            continue;
        }
        for (std::size_t k = 0; k < n; ++k) row[k] = source[k]/scale;
        const double original_norm = std::sqrt(barnes_detail::dot(row, row, n));
        barnes_detail::project(row, basis, retained, n);
        const double norm = std::sqrt(barnes_detail::dot(row, row, n));
        if (!std::isfinite(norm)) return 1;
        if (norm <= threshold*original_norm) {
            if (current) return 3;
            continue;
        }
        double* destination = current ? z : basis+retained*n;
        for (std::size_t k = 0; k < n; ++k) destination[k] = row[k]/norm;
        if (current) return 0;
        ++retained;
    }
    return 3;  // Unreachable: the current step always returns above.
}

// out = matrix + (target-matrix*step)*z^T/(z^T*step).
// Trusted disjoint column-major matrix[n*n], step[n], target[n], z[n],
// out[n*n]. Same allocation/input/status guarantees as barnes_direction.
// Status 3 denotes a zero divisor. Both compensated residual components are
// carried into each update entry to preserve small targets after cancellation.
inline int barnes_update(const double* matrix, const double* step,
                         const double* target, const double* z,
                         std::size_t n, double* out) noexcept {
    using namespace broyden_detail;
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || n > maximum / n) return 2;
    const auto square = n*n;
    tensor_detail::poison(out, square);
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(step[k]) || !std::isfinite(target[k]) ||
            !std::isfinite(z[k])) return 1;
    for (std::size_t k = 0; k < square; ++k)
        if (!std::isfinite(matrix[k])) return 1;
    if (n == 1) {
        if (z[0] == 0.0 || step[0] == 0.0) return 3;
        const double value = target[0]/step[0];
        if (!std::isfinite(value)) return 1;
        out[0] = value;
        return 0;
    }
    Sum denominator;
    for (std::size_t k = 0; k < n; ++k)
        denominator.add(product(z[k], step[k]));
    const auto divisor = denominator.value();
    if (divisor.mantissa == 0.0) return 3;
    for (std::size_t i = 0; i < n; ++i) {
        Sum residual;
        residual.add(normalized(target[i], 0));
        for (std::size_t k = 0; k < n; ++k) {
            auto term = product(matrix[i+n*k], step[k]);
            term.mantissa = -term.mantissa;
            residual.add(term);
        }
        for (std::size_t j = 0; j < n; ++j) {
            int exponent = 0;
            const double mantissa = std::frexp(z[j], &exponent);
            Sum entry;
            entry.add(normalized(matrix[i+n*j], 0));
            const ScaledTerm components[] = {residual.high, residual.low};
            for (const auto r : components)
                entry.add(normalized(
                    (r.mantissa*mantissa)/divisor.mantissa,
                    r.exponent+exponent-divisor.exponent));
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
