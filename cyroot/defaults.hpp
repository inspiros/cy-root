#include <cstdlib>
#include <limits>

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
