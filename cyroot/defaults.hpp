#ifndef CYROOT_DEFAULTS_HPP
#define CYROOT_DEFAULTS_HPP

#include <cstddef>
#include <limits>

// The single definition of the shared numerical defaults. Every solver family
// and root-acceptance wrapper takes its defaults from here (through
// cyroot/defaults.pxd); method-specific controls keep their own defaults.
namespace cyroot {
    // stop condition args
    static constexpr double ETOL = 1e-10;
    static constexpr double ERTOL = 4 * std::numeric_limits<double>::epsilon();
    static constexpr double PTOL = 1e-12;
    static constexpr double PRTOL = 4 * std::numeric_limits<double>::epsilon();
    static constexpr std::size_t MAX_ITER = 200;

    // derivative approximation args
    static constexpr double FINITE_DIFF_STEP = 1e-3;
}

#endif  // CYROOT_DEFAULTS_HPP
