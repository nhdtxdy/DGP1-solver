#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>

#define USE_INTEGER_WEIGHTS

#ifdef USE_INTEGER_WEIGHTS
    using WeightType = long long;
#else
    using WeightType = double;
#endif

// Knapsack parameters
namespace ssp {
    constexpr int M_LIMIT = 5000; // the actual range is -M_LIMIT/2 to M_LIMIT/2 (usually doubled, unless known for sure)
    constexpr int E_LIMIT = M_LIMIT * 2;
    constexpr int OFFSET = E_LIMIT; // 0 to 2 * M_LIMIT
    constexpr int WINDOW = E_LIMIT * 2 + 1;
    const int MIN_LOOKAHEAD_DEPTH = std::max(7, (int)log2(WINDOW) - 3);
    const int MAX_CYCLE_RULES_SIZE = 5;
}

// Epsilon value for floating-point precision
constexpr WeightType eps = (WeightType)1e-6;

#endif // CONFIG_H

