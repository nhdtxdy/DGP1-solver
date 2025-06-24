#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#define USE_INTEGER_WEIGHTS

#ifdef USE_INTEGER_WEIGHTS
    using WeightType = int;
#else
    using WeightType = double;
#endif

// SSP parameters - these are compile-time constants
namespace ssp {
    constexpr int REAL_M_LIMIT = 7; // Modify this to change the value range to search. [-REAL_M_LIMIT, REAL_M_LIMIT]

    // -------------------------------------------------------------------
    // DO NOT MODIFY UNLESS ABSOLUTELY SURE
    constexpr int M_LIMIT = REAL_M_LIMIT * 2;
    constexpr int E_LIMIT = M_LIMIT;
    constexpr int OFFSET = E_LIMIT; // 0 to 2 * M_LIMIT
    constexpr int WINDOW = E_LIMIT * 2 + 1;
    // -------------------------------------------------------------------
    constexpr int MIN_LOOKAHEAD_DEPTH = 0;
    // const int MIN_LOOKAHEAD_DEPTH = std::max(7, (int)log2(WINDOW) - 3); // Specifies the minimum distance for the SSP to be enabled.
    const int MAX_CYCLE_RULES_SIZE = 100000; // Limit how many SSP rules will be checked for every node (try 1-10).
}

// Runtime-configurable parameters
class Config {
public:
    struct Randomize {
        int RANDOMIZE_ARRAY_SIZE = 5000000; // 5e6
    } randomize;

    struct NodeScoring {
        double DEGREE_PROD = 1;
        double CYCLE_COUNT_PROD = 1;
        double WEIGHT_SUM_PROD = 0.;
        double CYCLE_PARTICIPATION_PROD = 0.25;

        double DEGREE_POW = 1;
        double CYCLE_COUNT_POW = 1;
        double WEIGHT_SUM_POW = 1;
        double CYCLE_PARTICIPATION_POW = 1;
    } node_scoring;

    WeightType eps = (WeightType)1e-6;

    // Singleton access
    static Config& getInstance();

    // Load configuration from file
    bool loadFromFile(const std::string& filename);
    
    // Print current configuration
    void printConfig() const;

private:
    Config() = default;
    static Config* instance;

    // Helper to trim strings
    static std::string trim(const std::string &s);
};

// Convenience macro
#define config Config::getInstance()

#endif // CONFIG_H

