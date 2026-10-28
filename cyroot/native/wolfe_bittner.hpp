#ifndef CYROOT_NATIVE_WOLFE_BITTNER_HPP
#define CYROOT_NATIVE_WOLFE_BITTNER_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot {
namespace native {

// Trusted disjoint spans: matrix[rows*k] column-major, weights[k],
// points[k*n] row-major, candidate[n], and affine_error[1]. The target
// affine system is matrix*weights=(1,0,...). No allocation or exceptions.
// 0 success; 1 nonfinite input/arithmetic; 2 invalid dimensions (no writes);
// 3 inconsistent affine system. Valid-dimension failures poison candidate;
// affine_error is the complete finite residual norm if available, else NaN.
inline int wolfe_bittner_candidate(const double* matrix, const double* weights,
                                  const double* points, std::size_t rows,
                                  std::size_t k, std::size_t n,
                                  double* candidate, double* affine_error) noexcept {
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (rows == 0 || k == 0 || n == 0 || k > maximum ||
        rows > maximum/k || n > maximum/k) return 2;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    *affine_error = nan;
    const auto fail = [=](int code) noexcept {
        for (std::size_t i = 0; i < n; ++i) candidate[i] = nan;
        return code;
    };
    // Preflight all inputs, including points not used by zero weights.
    for (std::size_t i = 0; i < rows*k; ++i)
        if (!std::isfinite(matrix[i])) return fail(1);
    for (std::size_t i = 0; i < k*n; ++i)
        if (!std::isfinite(points[i])) return fail(1);
    double weight_norm = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
        if (!std::isfinite(weights[j])) return fail(1);
        const double magnitude = std::fabs(weights[j]);
        if (magnitude > weight_norm) weight_norm = magnitude;
    }
    double matrix_norm = 0.0, error = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        double value = 0.0, row_norm = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            const double entry = matrix[i+rows*j];
            const double product = entry*weights[j];
            if (!std::isfinite(product)) return fail(1);
            value += product;
            row_norm += std::fabs(entry);
            if (!std::isfinite(value) || !std::isfinite(row_norm))
                return fail(1);
        }
        const double residual = std::fabs(value-(i == 0 ? 1.0 : 0.0));
        if (!std::isfinite(residual)) return fail(1);
        if (residual > error) error = residual;
        if (row_norm > matrix_norm) matrix_norm = row_norm;
    }
    *affine_error = error;
    const double scale = matrix_norm*weight_norm+1.0;
    const double bound = 32.0*std::numeric_limits<double>::epsilon()*
                         static_cast<double>(rows > k ? rows : k)*scale;
    if (!std::isfinite(scale) || !std::isfinite(bound)) return fail(1);
    if (error > bound) return fail(3);
    for (std::size_t i = 0; i < n; ++i) {
        double value = 0.0;
        for (std::size_t j = 0; j < k; ++j) {
            const double product = weights[j]*points[j*n+i];
            if (!std::isfinite(product)) return fail(1);
            value += product;
            if (!std::isfinite(value)) return fail(1);
        }
        candidate[i] = value;
    }
    return 0;
}

}  // namespace native
}  // namespace cyroot
#endif
