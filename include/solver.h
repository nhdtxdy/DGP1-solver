#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <iostream>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include "dsu.h"
#include <optional>
#include <random>
#include <map>
#include "config.h"
#include "bitset2/bitset2.hpp"

using Rule = std::tuple<int, int, WeightType>;

// template<size_t N>
// using BS2 = Bitset2::bitset2<N>;

enum OptimizationSetting { OPT_DEFAULT, OPT_HIGHEST_ORDER, OPT_HIGHEST_CYCLE, OPT_HIGHEST_SCORE };

struct Adj {
    int v;
    WeightType weight;
};

struct Edge {
    int u, v;
    WeightType weight;
};

struct NodeMetadata {
    double degree;
    double cycle_count;
    double weight_sum;
    double cycle_participation;

    double evaluate() const;
};

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
}

template <typename WeightType>
struct TripleHash {
    std::size_t operator()(const Rule& t) const {
        uint64_t x = static_cast<uint32_t>(std::get<0>(t));
        uint64_t y = static_cast<uint32_t>(std::get<1>(t));
        uint64_t z;

        if constexpr (std::is_integral_v<WeightType>) {
            z = static_cast<uint64_t>(std::get<2>(t));
        } else if constexpr (std::is_floating_point_v<WeightType>) {
            // Bit-cast float to uint64_t for consistent hashing
            double d = std::get<2>(t);
            z = *reinterpret_cast<uint64_t*>(&d);
        } else {
            static_assert(!sizeof(WeightType), "Unsupported WeightType");
        }

        uint64_t combined = x;
        combined = combined * 31 + y;
        combined = combined * 31 + z;

        return static_cast<std::size_t>(splitmix64(combined));
    }
};

class Solver {
public:
    Solver(int n,
           const std::vector<Edge>& edges,
           OptimizationSetting rootSelection,
           OptimizationSetting neighborSelection,
           bool bridgesOpt,
           bool listAllSolutions,
           bool randomize,
           bool knapsack,
           bool triangleInequality,
           bool preprocessKnapsack,
           bool bounds);
    int solve(std::ostream &out);

private:
    void bfs(int v);
    void dfs(int v, int par, bool can_binlift);
    void getsz(int v, int par);
    std::unordered_set<int> buildDfsTree(const std::vector<int> &idx, bool first_time);
    bool tryAssign(int v, WeightType val);
    std::optional<std::vector<std::unordered_map<int, WeightType>>> tryAssignAll(int v, WeightType val, int par = -1, int dep = 1);
    int buildAdjFromEdges();
    void dfs_bridges(int v, int par = -1);
    void find_bridges();
    void outputCombinedResult(std::ostream &out, const std::vector<std::vector<std::unordered_map<int, WeightType>>> &all_res, int num_solutions = -1);
    void saveDFSTree();
    void get_knapsack(int v, WeightType w_par, int par = -1);
    bool can_knapsack(int u, int v, WeightType w);
    bool verify_solution(const std::map<int, WeightType> &sol);
    void calculate_sum_pathw(int v, int par = -1, WeightType w = 0);
    void do_dijkstra();

    std::mt19937 rng;
    std::uniform_real_distribution<double> ran_gen;
    int n, logn, timer = 0, dfs_counter = 0;
    bool bridgesOpt, m_listAllSolutions, m_randomize, m_knapsack, m_triangleInequality, m_preprocessKnapsack, m_bounds;
    DSU bridged_dsu;
    std::vector<std::vector<Adj>> adj, dfs_tree_adj, back_adj, forward_adj;
    std::vector<int> vis;
    std::set<std::pair<int, int>> bridges;
    std::vector<Rule> bridge_rules;
    std::vector<std::vector<Rule>> cycle_rules;
    std::vector<Edge> edges;
    std::vector<int> tin, low, sz, parent, dep, in_cycles;
    std::vector<WeightType> value;
    OptimizationSetting m_rootSelection, m_neighborSelection;
    std::vector<std::vector<std::bitset<ssp::WINDOW>>> knapsack;
    std::vector<std::vector<int>> binlift;
    std::vector<std::vector<WeightType>> path_sum, max_w;
    std::pair<WeightType, WeightType> get_maxw_sum_path(int u, int v);
    std::vector<bool> randomize_array;
    std::vector<std::vector<WeightType>> dist;
    std::vector<std::vector<std::pair<WeightType, WeightType>>> bounds; // persistent
    std::vector<double> node_scores;
    std::vector<NodeMetadata> node_metadata;
    std::vector<WeightType> parent_w;
    // std::unordered_map<Rule, bool, TripleHash<WeightType>> knapsack_memoization;
    std::vector<std::vector<std::unordered_map<WeightType, bool>>> knapsack_memoization;
};

#endif // SOLVER_H