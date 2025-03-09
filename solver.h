#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <iostream>
#include <set>
#include <tuple>
#include <unordered_map>
#include "dsu.h"
#include <optional>

using Rule = std::tuple<int, int, double>;

enum OptimizationSetting { OPT_DEFAULT, OPT_HIGHEST_ORDER, OPT_LOWEST_ORDER };

struct Adj {
    int v;
    double weight;
};

struct Edge {
    int u, v;
    double weight;
};

class Solver {
public:
    Solver(int n, const std::vector<Edge>& edges);
    int solve(std::ostream &out, OptimizationSetting opt, bool bridgesOpt, bool listAllSolutions);

private:
    DSU bridged_dsu;

    static constexpr double eps = 1e-6;

    std::vector<std::vector<Adj>> adj, dfs_tree_adj, back_adj;
    std::vector<int> vis;
    std::set<std::pair<int, int>> bridges;
    std::vector<Rule> rules;
    std::vector<Edge> edges;
    std::vector<int> tin, low, dist, sz;

    std::unordered_map<int, double> value;


    int timer = 0;
    int n;
    
    bool bridge_opt_set, m_listAllSolutions;

    void bfs(int v);
    void dfs(int v, int par);
    void getsz(int v, int par);
    std::vector<int> buildDfsTree(const std::vector<int> &idx);
    bool tryAssign(int v, double val);
    std::optional<std::vector<std::unordered_map<int, double>>> tryAssignAll(int v, double val, int par = -1);
    std::vector<int> getOrder(OptimizationSetting opt);
    int buildAdjFromEdges();
    void dfs_bridges(int v, int par = -1);
    void find_bridges();
    void outputCombinedResult(std::ostream &out, const std::vector<std::vector<std::unordered_map<int, double>>> &all_res, int num_solutions = -1);
};

#endif // SOLVER_H