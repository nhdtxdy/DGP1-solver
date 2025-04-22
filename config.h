#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>

#define USE_INTEGER_WEIGHTS

#ifdef USE_INTEGER_WEIGHTS
    using WeightType = long long;
#else
    using WeightType = double;
#endif

// SSP parameters
namespace ssp {
    constexpr int REAL_M_LIMIT = 5000; // Modify this to change the value range to search.

    // -------------------------------------------------------------------
    // DO NOT MODIFY UNLESS ABSOLUTELY SURE
    constexpr int M_LIMIT = REAL_M_LIMIT * 2;
    constexpr int E_LIMIT = M_LIMIT * 2;
    constexpr int OFFSET = E_LIMIT; // 0 to 2 * M_LIMIT
    constexpr int WINDOW = E_LIMIT * 2 + 1;
    // -------------------------------------------------------------------
    
    const int MIN_LOOKAHEAD_DEPTH = std::max(7, (int)log2(WINDOW) - 3); // Specifies the minimum distance for the SSP to be enabled.
    const int MAX_CYCLE_RULES_SIZE = 5; // Limit how many SSP rules will be checked for every node (try 1-10).
}

namespace randomize {
    constexpr int RANDOMIZE_ARRAY_SIZE = 5e6;
}

// Epsilon value for floating-point precision
constexpr WeightType eps = (WeightType)1e-6;

#endif // CONFIG_H

