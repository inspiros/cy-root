#ifndef CYROOT_NATIVE_HIGHER_MIXED_HPP
#define CYROOT_NATIVE_HIGHER_MIXED_HPP

#include <cmath>
#include <cstddef>
#include <limits>

namespace cyroot {
namespace native {

// The immutable graph is prepared by a plan, outside numerical evaluation.
// Its contents and allocated spans are trusted; the kernel checks its declared
// dimensions, products and caller-owned capacities. All spans are disjoint.
struct TensorDifferenceConfig {
    std::size_t n, m, order, components, samples, terms, tensor_size;
    std::size_t input_size, nodes, work;
    int scheme;  // 0 forward, 1 backward, 2 central
    const double* steps;  // NULL selects automatic steps; otherwise n entries
    const std::size_t* indices;
    const std::size_t* starts;
    const std::size_t* term_samples;
    const int* offsets;
    const std::size_t* scatter;
};

using TensorDifferenceFunction = int (*)(const double*, std::size_t,
                                        std::size_t, double*, void*);

namespace higher_mixed_detail {

inline bool product(std::size_t a, std::size_t b,
                    std::size_t& result) noexcept {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        return false;
    result = a * b;
    return true;
}

inline bool add(std::size_t a, std::size_t b,
                std::size_t& result) noexcept {
    if (a > std::numeric_limits<std::size_t>::max() - b) return false;
    result = a + b;
    return true;
}

inline std::size_t gcd(std::size_t a, std::size_t b) noexcept {
    while (b != 0) {
        const std::size_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

inline bool byte_span(std::size_t count, std::size_t width) noexcept {
    // Pointer arithmetic must also fit ptrdiff_t, independently of size_t.
    return count <= static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / width;
}

inline bool valid_config(const TensorDifferenceConfig& c,
                         std::size_t output_capacity,
                         std::size_t capacity) noexcept {
    if (c.n == 0 || c.m == 0 || c.order < 3 || c.scheme < 0 || c.scheme > 2 ||
        c.order > capacity || c.nodes > capacity || c.n > capacity || c.m > capacity ||
        c.indices == nullptr || c.starts == nullptr ||
        c.term_samples == nullptr || c.offsets == nullptr || c.scatter == nullptr)
        return false;
    std::size_t nodes;
    if (!add(c.order, 1, nodes) ||
        (c.scheme == 2 && !add(nodes, c.order % 2, nodes)) || c.nodes != nodes)
        return false;
    // Requested integer offsets are represented by the graph's int entries.
    const std::size_t offset_limit = c.scheme == 2 ? (nodes - 1) / 2 : nodes - 1;
    if (offset_limit > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
    std::size_t input_size = 1;
    for (std::size_t j = 0; j < c.order; ++j)
        if (!product(input_size, c.n, input_size)) return false;
    std::size_t tensor_size;
    if (input_size != c.input_size || !product(input_size, c.m, tensor_size) ||
        tensor_size != c.tensor_size || output_capacity < tensor_size ||
        !byte_span(tensor_size, sizeof(double))) return false;

    // comb(n+order-1, order), with cancellation before multiplication.
    std::size_t components = 1;
    for (std::size_t j = 1; j <= c.order; ++j) {
        std::size_t numerator;
        if (!add(c.n - 1, j, numerator)) return false;
        std::size_t denominator = j;
        const std::size_t common = gcd(numerator, denominator);
        numerator /= common;
        denominator /= common;
        components /= denominator;  // exact for the binomial recurrence
        if (!product(components, numerator, components)) return false;
    }
    if (components != c.components || c.samples == 0 ||
        c.terms < components || c.samples - 1 > c.terms) return false;
    std::size_t count;
    if (!product(components, c.order, count) ||
        !byte_span(count, sizeof(std::size_t)) ||
        !add(components, 1, count) || !byte_span(count, sizeof(std::size_t)) ||
        !byte_span(c.terms, sizeof(std::size_t)) ||
        !product(c.samples, c.n, count) || !byte_span(count, sizeof(int)) ||
        !byte_span(input_size, sizeof(std::size_t))) return false;
    // Each component has the product of its repeated-axis stencil widths.
    // Check the declared term count without allocating a composition grid.
    std::size_t terms = 0;
    for (std::size_t component = 0; component < components; ++component) {
        if (c.starts[component] != terms) return false;
        const std::size_t* indices = c.indices + component * c.order;
        std::size_t component_terms = 1;
        for (std::size_t first = 0; first < c.order;) {
            std::size_t end = first + 1;
            while (end < c.order && indices[end] == indices[first]) ++end;
            const std::size_t degree = end - first;
            const std::size_t width = degree + 1 + (c.scheme == 2 ? degree % 2 : 0);
            if (!product(component_terms, width, component_terms)) return false;
            first = end;
        }
        if (!add(terms, component_terms, terms)) return false;
    }
    if (c.starts[components] != terms || c.terms != terms) return false;

    std::size_t work = 0;
    if (!product(c.samples, c.m, count) || !add(work, count, work) ||
        !product(2, c.n, count) || !add(work, count, work) ||
        !product(components, c.m, count) || !add(work, count, work) ||
        !product(c.order, nodes, count) || !add(work, count, work) ||
        !add(work, c.order, work) ||
        !product(nodes, c.order + 1, count) || !add(work, count, work) ||
        !product(2, nodes, count) || !add(work, count, work)) return false;
    return c.work == work && capacity >= work &&
        byte_span(work, sizeof(double)) && byte_span(c.n, sizeof(double)) &&
        byte_span(c.m, sizeof(double));
}

inline double offset(std::size_t slot, int scheme) noexcept {
    if (scheme == 2) {
        const double value = static_cast<double>((slot + 1) / 2);
        return slot % 2 == 0 ? -value : value;
    }
    const double value = static_cast<double>(slot);
    return scheme == 1 ? -value : value;
}

// Build rounded coordinates and normalized displacements, including base0.
// Status1: finite arithmetic overflow; status2: collapsed rounded nodes.
inline int coordinates(double x, double step, std::size_t width, int scheme,
                        double* points, double* nodes, double& scale) noexcept {
    scale = 0.0;
    points[0] = x;
    nodes[0] = 0.0;
    for (std::size_t i = 1; i < width; ++i) {
        const double displacement = offset(i, scheme) * step;
        points[i] = x + displacement;
        nodes[i] = points[i] - x;
        if (!std::isfinite(displacement) || !std::isfinite(points[i]) ||
            !std::isfinite(nodes[i])) return 1;
        for (std::size_t j = 0; j < i; ++j)
            if (points[i] == points[j]) return 2;
        if (std::fabs(nodes[i]) > scale) scale = std::fabs(nodes[i]);
    }
    for (std::size_t i = 1; i < width; ++i) {
        nodes[i] /= scale;
        for (std::size_t j = 0; j < i; ++j)
            if (nodes[i] == nodes[j]) return 2;
    }
    return 0;
}

// Fornberg interpolation weights at zero, using the actual rounded nodes.
inline int weights(const double* nodes, std::size_t count, std::size_t order,
                   double* coefficients, double* out) noexcept {
    const std::size_t width = order + 1;
    for (std::size_t i = 0; i < count * width; ++i) coefficients[i] = 0.0;
    coefficients[0] = 1.0;
    double c1 = 1.0, c4 = 0.0;
    for (std::size_t i = 1; i < count; ++i) {
        const std::size_t mn = i < order ? i : order;
        double c2 = 1.0;
        const double c5 = c4;
        c4 = nodes[i];
        for (std::size_t j = 0; j < i; ++j) {
            const double c3 = nodes[i] - nodes[j];
            if (c3 == 0.0) return 2;
            c2 *= c3;
            if (!std::isfinite(c2) || c2 == 0.0) return 1;
            if (j == i - 1) {
                const double multiplier = c1 / c2;
                if (!std::isfinite(multiplier)) return 1;
                for (std::size_t k = mn; k != 0; --k) {
                    const double t1 = static_cast<double>(k) *
                        coefficients[(i - 1) * width + k - 1];
                    const double t2 = c5 * coefficients[(i - 1) * width + k];
                    const double difference = t1 - t2;
                    const double value = multiplier * difference;
                    if (!std::isfinite(t1) || !std::isfinite(t2) ||
                        !std::isfinite(difference) || !std::isfinite(value)) return 1;
                    coefficients[i * width + k] = value;
                }
                const double value = -multiplier * c5 * coefficients[(i - 1) * width];
                if (!std::isfinite(value)) return 1;
                coefficients[i * width] = value;
            }
            for (std::size_t k = mn; k != 0; --k) {
                const double t1 = c4 * coefficients[j * width + k];
                const double t2 = static_cast<double>(k) * coefficients[j * width + k - 1];
                const double difference = t1 - t2;
                const double value = difference / c3;
                if (!std::isfinite(t1) || !std::isfinite(t2) ||
                    !std::isfinite(difference) || !std::isfinite(value)) return 1;
                coefficients[j * width + k] = value;
            }
            const double value = c4 * coefficients[j * width] / c3;
            if (!std::isfinite(value)) return 1;
            coefficients[j * width] = value;
        }
        c1 = c2;
    }
    for (std::size_t i = 0; i < count; ++i) out[i] = coefficients[i * width + order];
    return 0;
}

// Divide by the product of each group's scale^multiplicity without forming
// that product or prematurely overflowing/underflowing intermediate quotients.
inline int rescale(double value, const std::size_t* indices, std::size_t order,
                   const double* scales, double& out) noexcept {
    if (value == 0.0) {
        out = value;
        return 0;
    }
    int shift = 0;
    double mantissa = std::frexp(value, &shift);
    long long exponent = shift;
    std::size_t group = 0;
    for (std::size_t first = 0; first < order; ++group) {
        std::size_t end = first + 1;
        while (end < order && indices[end] == indices[first]) ++end;
        int scale_exponent = 0;
        const double scale_mantissa = std::frexp(scales[group], &scale_exponent);
        for (std::size_t i = first; i < end; ++i) {
            mantissa = std::frexp(mantissa / scale_mantissa, &shift);
            const long long change = static_cast<long long>(shift) - scale_exponent;
            if ((change > 0 && exponent > std::numeric_limits<long long>::max() - change) ||
                (change < 0 && exponent < std::numeric_limits<long long>::min() - change))
                return 1;
            exponent += change;
        }
        first = end;
    }
    if (exponent > std::numeric_limits<int>::max()) return 1;
    if (exponent < std::numeric_limits<int>::min()) {
        out = std::copysign(0.0, mantissa);
        return 0;
    }
    out = std::ldexp(mantissa, static_cast<int>(exponent));
    return std::isfinite(out) ? 0 : 1;
}

}  // namespace higher_mixed_detail

// Status:0 ok,1 nonfinite,2 breakdown,3 callback failure. No heap/Python calls.
// Invalid declared shapes/counts/capacities return2 before scratch/output writes.
// On valid configuration all failures poison the complete tensor and restore
// trial x. Calls counts stencil attempts only (the caller supplies cached f0).
inline int compute_tensor_difference(const TensorDifferenceConfig* config,
    const double* x, const double* f0, double* out, std::size_t output_capacity,
    double* work, std::size_t capacity, TensorDifferenceFunction function,
    void* context, std::size_t* calls, int* callback_code) noexcept {
    using namespace higher_mixed_detail;
    if (callback_code == nullptr) return 2;
    *callback_code = 0;
    if (config == nullptr || x == nullptr || f0 == nullptr || out == nullptr ||
        work == nullptr || function == nullptr || calls == nullptr ||
        !valid_config(*config, output_capacity, capacity)) return 2;
    const TensorDifferenceConfig& c = *config;
    if (*calls > std::numeric_limits<std::size_t>::max() - (c.samples - 1)) return 2;
    double* cache = work;
    double* trial = cache + c.samples * c.m;
    double* steps = trial + c.n;
    double* values = steps + c.n;
    double* group_weights = values + c.components * c.m;
    double* scales = group_weights + c.order * c.nodes;
    double* coefficients = scales + c.order;
    double* points = coefficients + c.nodes * (c.order + 1);
    double* nodes = points + c.nodes;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t i = 0; i < c.tensor_size; ++i) out[i] = nan;
    for (std::size_t i = 0; i < c.n; ++i) trial[i] = x[i];
    // One exit helper covers callbacks that leave trialpoint contents visible.
    const auto finish = [&](int status) noexcept {
        for (std::size_t j = 0; j < c.n; ++j) trial[j] = x[j];
        return status;
    };
    for (std::size_t j = 0; j < c.n; ++j)
        if (!std::isfinite(x[j])) return finish(1);
    for (std::size_t o = 0; o < c.m; ++o)
        if (!std::isfinite(f0[o])) return finish(1);
    const double automatic_factor = std::pow(std::numeric_limits<double>::epsilon(),
        1.0 / (static_cast<double>(c.order) + (c.scheme == 2 ? 2.0 : 1.0)));
    // Validate every possible requested axis offset before any stencil calls.
    for (std::size_t j = 0; j < c.n; ++j) {
        steps[j] = c.steps == nullptr ?
            (std::fabs(x[j]) > 1.0 ? std::fabs(x[j]) : 1.0) * automatic_factor :
            c.steps[j];
        if (!std::isfinite(steps[j])) return finish(1);
        if (steps[j] == 0.0) return finish(2);
        double scale;
        const int status = coordinates(x[j], steps[j], c.nodes, c.scheme,
                                        points, nodes, scale);
        if (status != 0) return finish(status);
    }
    for (std::size_t i = 0; i < c.samples * c.m; ++i) cache[i] = nan;
    for (std::size_t o = 0; o < c.m; ++o) cache[o] = f0[o];
    for (std::size_t sample = 1; sample < c.samples; ++sample) {
        for (std::size_t j = 0; j < c.n; ++j)
            trial[j] = x[j] + static_cast<double>(c.offsets[sample * c.n + j]) * steps[j];
        ++*calls;
        const int code = function(trial, c.n, c.m, cache + sample * c.m, context);
        if (code != 0) {
            *callback_code = code;
            return finish(3);
        }
        for (std::size_t o = 0; o < c.m; ++o)
            if (!std::isfinite(cache[sample * c.m + o])) return finish(1);
    }
    for (std::size_t component = 0; component < c.components; ++component) {
        const std::size_t* indices = c.indices + component * c.order;
        std::size_t groups = 0;
        for (std::size_t first = 0; first < c.order; ++groups) {
            std::size_t end = first + 1;
            while (end < c.order && indices[end] == indices[first]) ++end;
            const std::size_t degree = end - first;
            const std::size_t width = degree + 1 + (c.scheme == 2 ? degree % 2 : 0);
            int status = coordinates(x[indices[first]], steps[indices[first]], width,
                c.scheme, points, nodes, scales[groups]);
            if (status != 0) return finish(status);
            status = weights(nodes, width, degree, coefficients,
                             group_weights + groups * c.nodes);
            if (status != 0) return finish(status);
            first = end;
        }
        for (std::size_t o = 0; o < c.m; ++o) {
            double total = 0.0, correction = 0.0;
            for (std::size_t term = c.starts[component]; term < c.starts[component + 1]; ++term) {
                std::size_t remaining = term - c.starts[component];
                std::size_t group = 0;
                double weight = 1.0;
                for (std::size_t first = 0; first < c.order; ++group) {
                    std::size_t end = first + 1;
                    while (end < c.order && indices[end] == indices[first]) ++end;
                    const std::size_t degree = end - first;
                    const std::size_t width = degree + 1 + (c.scheme == 2 ? degree % 2 : 0);
                    weight *= group_weights[group * c.nodes + remaining % width];
                    if (!std::isfinite(weight)) return finish(1);
                    remaining /= width;
                    first = end;
                }
                const double difference = cache[c.term_samples[term] * c.m + o] - f0[o];
                const double contribution = weight * difference;
                const double adjusted = contribution - correction;
                const double updated = total + adjusted;
                correction = (updated - total) - adjusted;
                if (!std::isfinite(difference) || !std::isfinite(contribution) ||
                    !std::isfinite(adjusted) || !std::isfinite(updated) ||
                    !std::isfinite(correction)) return finish(1);
                total = updated;
            }
            const int status = rescale(total, indices, c.order, scales,
                                      values[component * c.m + o]);
            if (status != 0) return finish(status);
        }
    }
    for (std::size_t flat = 0; flat < c.input_size; ++flat)
        for (std::size_t o = 0; o < c.m; ++o)
            out[o + c.m * flat] = values[c.scatter[flat] * c.m + o];
    return finish(0);
}

}  // namespace native
}  // namespace cyroot

#endif
