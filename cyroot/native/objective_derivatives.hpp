#ifndef CYROOT_NATIVE_OBJECTIVE_DERIVATIVES_HPP
#define CYROOT_NATIVE_OBJECTIVE_DERIVATIVES_HPP

#include "objective.hpp"
#include <cstring>

namespace cyroot { namespace native {

// Exact Leibniz transform for phi=.5*<F,F>, derivative order>=1. D[k] has
// residual-first layout D[k][o+m*(j0+n*(j1+...))], length m*n^k. Output has
// n^order entries with its first input index fastest. Subsets containing slot0
// select exactly one of each complementary pair, cancelling the factor .5.
// Caller supplies valid disjoint input, output and scratch spans. No allocation,
// Python, exceptions, derivative symmetrization or numerical differentiation.
// Status0 success;1 nonfinite input/final result;2 invalid dimension/capacity/
// work bound. Out is unchanged on failure; scratch is unspecified.
inline int objective_derivative(const double* const* derivatives,
        const std::size_t* capacities, std::size_t n, std::size_t m,
        std::size_t order, double* out, std::size_t output_capacity,
        double* scratch, std::size_t scratch_capacity,
        std::size_t max_terms) noexcept {
    constexpr auto bits = std::numeric_limits<std::size_t>::digits;
    constexpr auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (!derivatives || !capacities || !out || !scratch || !n || !m ||
        !order || order >= bits) return 2;
    std::size_t powers[bits];
    powers[0] = 1;
    for (std::size_t k = 0; k <= order; ++k) {
        if (k) {
            if (powers[k-1] > maximum/n) return 2;
            powers[k] = powers[k-1]*n;
        }
        if (m > maximum/powers[k] || !derivatives[k] ||
            capacities[k] < m*powers[k]) return 2;
    }
    const auto count = powers[order];
    const auto partitions = std::size_t(1) << (order-1);
    if (output_capacity < count || scratch_capacity < count ||
        count > max_terms/m || partitions > max_terms/(m*count)) return 2;
    // Validate every provided derivative, including factors whose partner is0.
    for (std::size_t k = 0; k <= order; ++k)
        for (std::size_t j = 0; j < m*powers[k]; ++j)
            if (!std::isfinite(derivatives[k][j])) return 1;
    std::size_t axes[bits];
    for (std::size_t flat = 0; flat < count; ++flat) {
        auto index = flat;
        for (std::size_t j = 0; j < order; ++j) {
            axes[j] = index % n;
            index /= n;
        }
        ObjectiveAccumulator sum;
        for (std::size_t subset = 0; subset < partitions; ++subset) {
            const auto mask = (subset << 1) | std::size_t(1);
            std::size_t left = 0, right = 0, a = 0, b = 0;
            for (std::size_t j = 0; j < order; ++j) {
                if ((mask >> j) & 1) left += axes[j]*powers[a++];
                else right += axes[j]*powers[b++];
            }
            for (std::size_t o = 0; o < m; ++o)
                sum.add_product(derivatives[a][o+m*left], derivatives[b][o+m*right]);
        }
        if (sum.value(scratch+flat) != 0) return 1;
    }
    std::memcpy(out, scratch, count*sizeof(double));
    return 0;
}

}}  // namespace cyroot::native
#endif
