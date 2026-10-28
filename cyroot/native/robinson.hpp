#ifndef CYROOT_NATIVE_ROBINSON_HPP
#define CYROOT_NATIVE_ROBINSON_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot {
namespace native {

// Trusted disjoint spans previous[n], current[n], nodes[n*n],
// transpose[n*n], and scratch[n]. Each nodes column is a complete probe.
// transpose[j+n*i] stores its actual rounded displacement in coordinate i.
// The efficient basis is a deterministic Householder reflection mapping e0
// to -copysign(1,u0)*u, where u is the normalized separation direction.
// 0 success; 1 nonfinite input/result; 2 invalid dimension (no writes);
// 3 zero separation or a wholly collapsed probe. Valid-dimension failures
// poison both complete outputs. No allocation, input mutation, or exceptions.
inline int robinson_nodes(const double* previous, const double* current,
                          std::size_t n, bool identity, double* nodes,
                          double* transpose, double* scratch) noexcept {
    const auto maximum = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (n == 0 || n > maximum/n) return 2;
    const auto square = n*n;
    const auto fail = [=](int code) noexcept {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t k = 0; k < square; ++k)
            nodes[k] = transpose[k] = nan;
        return code;
    };
    // Scan all supplied coordinates before zero-separation early exits.
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(previous[i]) || !std::isfinite(current[i]))
            return fail(1);
    double radius = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        scratch[i] = current[i]-previous[i];
        if (!std::isfinite(scratch[i])) return fail(1);
        radius = std::hypot(radius, scratch[i]);
    }
    if (!std::isfinite(radius)) return fail(1);
    if (radius == 0.0) return fail(3);
    double coefficient = 0.0;
    if (!identity) {
        for (std::size_t i = 0; i < n; ++i) scratch[i] /= radius;
        // Choosing the sign of u0 avoids cancellation; zero uses +1.
        scratch[0] += scratch[0] < 0.0 ? -1.0 : 1.0;
        double squared_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            squared_norm += scratch[i]*scratch[i];
        coefficient = 2.0/squared_norm;
    }
    for (std::size_t j = 0; j < n; ++j) {
        bool moved = false;
        for (std::size_t i = 0; i < n; ++i) {
            double q = i == j ? 1.0 : 0.0;
            if (!identity) q -= coefficient*scratch[i]*scratch[j];
            const double node = current[i]+radius*q;
            const double displacement = node-current[i];
            if (!std::isfinite(node) || !std::isfinite(displacement))
                return fail(1);
            nodes[i+n*j] = node;
            transpose[j+n*i] = displacement;
            moved = moved || displacement != 0.0;
        }
        if (!moved) return fail(3);
    }
    return 0;
}

}  // namespace native
}  // namespace cyroot
#endif
