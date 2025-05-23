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
#include <bitset>
#include <map>
#include "config.h"

using Rule = std::tuple<int, int, WeightType>;

enum OptimizationSetting { OPT_DEFAULT, OPT_HIGHEST_ORDER, OPT_HIGHEST_CYCLE };

struct Adj {
    int v;
    WeightType weight;
};

struct Edge {
    int u, v;
    WeightType weight;
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
           bool triangleInequality);
    int solve(std::ostream &out);

private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> ran_gen;

    int n, logn;

    DSU bridged_dsu;

    std::vector<std::vector<Adj>> adj, dfs_tree_adj, back_adj, forward_adj;
    std::vector<int> vis;
    std::set<std::pair<int, int>> bridges;
    std::vector<Rule> bridge_rules;
    std::vector<std::vector<Rule>> cycle_rules;
    std::vector<Edge> edges;
    std::vector<int> tin, low, dist, sz, parent, dep, in_cycles;

    std::vector<WeightType> value;

    int timer = 0;

    OptimizationSetting m_rootSelection, m_neighborSelection;
    
    bool bridgesOpt, m_listAllSolutions, m_randomize, m_knapsack, m_triangleInequality;

    std::vector<std::vector<std::bitset<ssp::WINDOW>>> knapsack;
    std::vector<std::vector<int>> binlift;
    std::vector<std::vector<WeightType>> path_sum, max_w;

    void bfs(int v);
    void dfs(int v, int par, bool can_binlift);
    void getsz(int v, int par);
    std::unordered_set<int> buildDfsTree(const std::vector<int> &idx, bool first_time);
    bool tryAssign(int v, WeightType val);
    std::optional<std::vector<std::unordered_map<int, WeightType>>> tryAssignAll(int v, WeightType val, int par = -1);
    int buildAdjFromEdges();
    void dfs_bridges(int v, int par = -1);
    void find_bridges();
    void outputCombinedResult(std::ostream &out, const std::vector<std::vector<std::unordered_map<int, WeightType>>> &all_res, int num_solutions = -1);
    void saveDFSTree();
    void get_knapsack(int v, WeightType w_par, int par = -1);
    bool can_knapsack(int u, int v, WeightType w);
    bool verify_solution(const std::map<int, WeightType> &sol);
    void calculate_sum_pathw(int v, int par = -1, WeightType w = 0);

    std::pair<WeightType, WeightType> get_maxw_sum_path(int u, int v);
};

#endif // SOLVER_H