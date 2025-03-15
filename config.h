#include <math.h>

#define USE_INTEGER_WEIGHTS

#ifdef USE_INTEGER_WEIGHTS
    using WeightType = int;
#else
    using WeightType = double;
#endif

// Knapsack parameters
constexpr int M_LIMIT = 5000; // -M_LIMIT/2 to M_LIMIT/2 (usually doubled, unless known for sure)
constexpr int E_LIMIT = M_LIMIT * 2;
constexpr int OFFSET = E_LIMIT; // 0 to 2 * M_LIMIT
constexpr int WINDOW = E_LIMIT * 2 + 1;
const int MIN_LOOKAHEAD_DEPTH = std::max(7, (int)log2(WINDOW) - 3);

// Epsilon value for floating-point precision
constexpr WeightType eps = 1e-6;

