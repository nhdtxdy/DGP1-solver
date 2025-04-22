#include "solver.h"
#include <numeric>
#include <queue>
#include <math.h>
#include "dsu.h"
#include <map>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

// merge assignments from two branches
void merge(std::vector<std::unordered_map<int, WeightType>> &base, std::vector<std::unordered_map<int, WeightType>> to_merge) {
    std::vector<std::unordered_map<int, WeightType>> merged;
    for (const std::unordered_map<int, WeightType>& b : base) {
        for (const std::unordered_map<int, WeightType> &tm : to_merge) {
            std::unordered_map<int, WeightType> cpy = b;
            cpy.insert(tm.begin(), tm.end());
            merged.push_back(cpy);
        }
    }
    swap(base, merged);
}

// merge two knapsack bitsets (for knapsack optimization)
std::bitset<ssp::WINDOW> bitset_merge(const std::bitset<ssp::WINDOW> &A, const std::bitset<ssp::WINDOW> &B) {
    std::bitset<ssp::WINDOW> const *ptrA = &A;
    std::bitset<ssp::WINDOW> const *ptrB = &B;
    if (A.count() < B.count()) {
        std::swap(ptrA, ptrB);
    }
    // A > B
    std::bitset<ssp::WINDOW> res;
    for (int idx = ptrB->_Find_first(); idx < (int)ptrB->size(); idx = ptrB->_Find_next(idx)) {
        int shift = idx - ssp::OFFSET;
        if (shift >= 0 && shift < ssp::WINDOW) res |= ((*ptrA) << shift);
        else if (shift < 0 && (-shift) < ssp::WINDOW) res |= ((*ptrA) >> (-shift));
    }

    return res;
}

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

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        uint64_t a = static_cast<uint32_t>(p.first);
        uint64_t b = static_cast<uint32_t>(p.second);
        uint64_t combined = a * 31 + b;
        return static_cast<std::size_t>(splitmix64(combined));
    }
};

// checks if there exists an assignment with sum "value" using edges on the path from v to u
bool Solver::can_knapsack(int u, int v, WeightType value) {
    static std::unordered_map<Rule, bool, TripleHash<WeightType>> memoization;

    if (memoization.count(Rule{u, v, value})) {
        // std::cerr << "already memorized: " << u << ' ' << v << ' ' << value << '\n';
        return memoization[Rule{u, v, value}];
    }

    WeightType sum_path_uv = get_maxw_sum_path(u, v).first; // sum from path u -> v
    if ((std::abs(value) - sum_path_uv) > eps) {
        return (memoization[Rule{u, v, value}] = false);
    }

    int cpy_u = u, cpy_v = v;
    WeightType cpy_value = value;

    // std::cerr << "studying rule: " << u << ' ' << v << ' ' << value << '\n';

    std::bitset<ssp::WINDOW> bset;
    bool first = true;
    for (int i = logn; i >= 0; --i) {
        int mid = binlift[u][i];
        if (mid != -1 && dep[mid] >= dep[v]) {
            if (first) {
                bset = knapsack[u][i];
            }
            else {
                bset = bitset_merge(bset, knapsack[u][i]);
            }
            first = false;
            u = mid;
        }
    }

    return (memoization[Rule{cpy_u, cpy_v, cpy_value}] = bset[value + ssp::OFFSET]);
}

std::optional<std::vector<std::unordered_map<int, WeightType>>> Solver::tryAssignAll(int v, WeightType val, int par) {
    static int counter = 0;
    // static int lowest_infeasible_cycle = -1;
    // static std::unordered_set<std::pair<int, int>, PairHash> tested_cycles;

    // std::cerr << v << ' ' << val << ' ' << par << ' ' << lowest_infeasible_cycle << '\n';

    // if (dep[v] < lowest_infeasible_cycle) {
        // std::cerr << "PROBLEM: " << v << ' ' << lowest_infeasible_cycle << '\n';
        // return std::nullopt;
    // }

    if (m_knapsack && (val > ssp::M_LIMIT || val < -ssp::M_LIMIT)) return std::nullopt;
    value[v] = val;

    // detect invalid cycle through back edges
    for (auto &p : back_adj[v]) {
        int u = p.v;
        WeightType w = p.weight;
        if (std::abs(std::abs(value[v] - value[u]) - w) > eps) {
            // if (!tested_cycles.count({v, u})) {
            //     std::cerr << "OOPS: " << u << ' ' << v << ' ' << dep[u] << '\n';
            //     lowest_infeasible_cycle = dep[u];
            // }
            return std::nullopt;
        }
        // else {
        //     tested_cycles.insert({v, u});
        // }
    }

    if (m_knapsack) {
        for (const Rule &rule : cycle_rules[v]) {
            int x, y;
            WeightType w;
            std::tie(x, y, w) = rule;

            // value[x] is known
            WeightType potential_y1 = value[x] + w;
            WeightType potential_y2 = value[x] - w;

            if (!(-ssp::M_LIMIT <= potential_y1 && potential_y1 <= ssp::M_LIMIT && can_knapsack(y, v, potential_y1 - val)) && !(-ssp::M_LIMIT <= potential_y2 && potential_y2 <= ssp::M_LIMIT && can_knapsack(y, v, potential_y2 - val))) {
                return std::nullopt;
            }
        }
    }

    std::vector<std::unordered_map<int, WeightType>> merged;
    std::unordered_map<int, WeightType> single_map;
    single_map[v] = val;
    merged.push_back(single_map);

    for (auto &p : dfs_tree_adj[v]) {
        int u = p.v;
        WeightType w = p.weight;
        if (u == par) continue;

        ++counter;
        if (counter >= randomize::RANDOMIZE_ARRAY_SIZE) {
            counter -= randomize::RANDOMIZE_ARRAY_SIZE;
        }

        // 50% chance to flip the sign of w
        if (m_randomize && randomize_array[counter]) {
            w = -w;
        }


        // TO STUDY: TRAVERSAL ORDER HEURISTICS (BIG CHILD IN SMALL-TO-LARGE TECHNIQUE?)

        std::vector<std::unordered_map<int, WeightType>> to_merge;

        auto d1 = tryAssignAll(u, val + w, v);
        if (d1.has_value()) {
            // swap is O(1) while insert is O(n)
            std::swap(to_merge, d1.value());
        }

        if (this->m_listAllSolutions || to_merge.empty()) {
            const auto &d2 = tryAssignAll(u, val - w, v);
            if (d2.has_value()) {
                const std::vector<std::unordered_map<int, WeightType>> &d2_value = d2.value();
                to_merge.insert(to_merge.end(), d2_value.begin(), d2_value.end());
            }
        }

        if (to_merge.empty()) {
            return std::nullopt;
        }

        merge(merged, to_merge);
    }

    // lowest_infeasible_cycle = -1;

    return merged;
}

// DFS helper for computing translations (in case of bridges optimization)
std::vector<std::unordered_map<int, WeightType>> dfsTranslations(int v,
                    WeightType val,
                    int par,
                    const std::vector<std::vector<std::tuple<int, WeightType, WeightType>>>& compAdj,
                    bool listAllSolutions)
{
    std::vector<std::unordered_map<int, WeightType>> merged;
    std::unordered_map<int, WeightType> single_map;
    single_map[v] = val;
    merged.push_back(single_map);

    for (auto &p : compAdj[v]) {
        int u;
        WeightType w;
        WeightType cur_dist;
        std::tie(u, w, cur_dist) = p;
        if (u == par) continue;

        std::vector<std::unordered_map<int, WeightType>> to_merge;

        // v -> u: cur_dist, want w
        auto d1_value = dfsTranslations(u, val + (w - cur_dist), v, compAdj, listAllSolutions);
        std::swap(to_merge, d1_value);

        if (listAllSolutions || to_merge.empty()) {
            const auto &d2_value = dfsTranslations(u, val + (-w - cur_dist), v, compAdj, listAllSolutions);
            to_merge.insert(to_merge.end(), d2_value.begin(), d2_value.end());
        }

        merge(merged, to_merge);
    }

    return merged;
}


// This iterator iterates over the Cartesian product of candidate solutions (one per component)
// (from all_res) and, for each merged candidate solution, enumerates all scaled solutions based on a set of rules.
// Each rule is a tuple {u, v, w} meaning that the final values must satisfy:
//    | (merged[u] + T[compMap[u]']) - (merged[v] + T[compMap[v]']) | = w,
// where compMap[u]' is the re-indexed component (a contiguous index 0..k-1).
//
// Since each rule gives two choices (±), the DFS in the scaling routine enumerates all translation vectors.
class CombinedSolutionIterator {
public:
    // Constructor:
    //   all_res: for each (re-indexed) component, a vector of candidate solutions.
    //            Each candidate solution is an unordered_map<int,WeightType> (vertex -> base value).
    //   rules: a vector of rules; each rule is a tuple {u, v, w}.
    //   compMap: mapping from vertex to its DSU representative (1-indexed, not necessarily contiguous).
    //            (Note: all_res is assumed to be ordered by these re-indexed components.)
    CombinedSolutionIterator(
        const std::vector<std::vector<std::unordered_map<int, WeightType>>>& all_res,
        const std::vector<Rule>& rules,
        const std::vector<int>& compMap,
        bool listAllSolutions)
        : all_res(all_res), rules(rules), compMap(compMap), has_next(true), m_listAllSolutions(listAllSolutions)
    {
        int numComponents = all_res.size();
        indices.resize(numComponents, 0);
        for (int i = 0; i < numComponents; i++) {
            if (all_res[i].empty()) {
            has_next = false;
            break;
            }
        }
        int n = compMap.size() - 1;
        for (int i = 1; i <= n; ++i) {
        int rep = compMap[i];
        compSet.insert(rep);
        }
        // Map each DSU rep to a contiguous index.
        int idx = 0;
        for (int rep : compSet) {
            repToIndex[rep] = idx++;
        }
        advanceBuffer(); // Preload buffer from the first merged candidate.
    }
    
    // Returns true if there is another combined (scaled) solution.
    bool hasNext() const {
        return (!buffer.empty() || has_next);
    }
    
    // Returns the next combined (scaled) solution.
    std::unordered_map<int, WeightType> next() {
        if (buffer.empty()) {
            advanceBuffer();
        }
        std::unordered_map<int, WeightType> sol = buffer.front();
        buffer.erase(buffer.begin());
        return sol;
    }
    
private:
    const std::vector<std::vector<std::unordered_map<int, WeightType>>>& all_res;
    const std::vector<std::tuple<int, int, WeightType>>& rules;
    const std::vector<int>& compMap;  // DSU representative for each vertex (1-indexed).
    std::vector<int> indices;         // current candidate index for each component.
    bool has_next, m_listAllSolutions;
    std::vector<std::unordered_map<int, WeightType>> buffer;  // buffer holding scaled solutions for current merged candidate.
    
    std::set<int> compSet;
    std::unordered_map<int, int> repToIndex;

    // Merge candidate solutions from each component according to current indices.
    std::unordered_map<int, WeightType> mergeCurrentCandidate() {
        std::unordered_map<int, WeightType> merged;
        for (size_t i = 0; i < all_res.size(); i++) {
            const auto &sol = all_res[i][indices[i]];
            for (const auto &p : sol) {
                // Candidate solutions from different components are guaranteed to use disjoint vertex sets.
                merged[p.first] = p.second;
            }
        }
        return merged;
    }
    
    // Advance the multi-digit indices.
    void advanceIndices() {
        for (int i = all_res.size() - 1; i >= 0; i--) {
            indices[i]++;
            if (indices[i] < (int)all_res[i].size()) {
                break; // no carry needed.
            } else {
                indices[i] = 0;
                if (i == 0) {
                    has_next = false;
                }
            }
        }
    }
    
    // Given a merged solution, we want to find all translation vectors T (one per component)
    // so that for each rule {u, v, w} (with u and v in different components)
    // we have:
    //    T[comp(u)'] - T[comp(v)'] = ± (w - (merged[u] - merged[v])),
    // where comp(u)' is the re-indexed component.
    // Since compMap is not necessarily 0-indexed, we first re-index.
    std::vector<std::unordered_map<int, WeightType>> scaleAllTranslations(const std::unordered_map<int, WeightType>& merged) {
        int k = repToIndex.size();

        // Build the component adjacency list.
        std::vector<std::vector<std::tuple<int, WeightType, WeightType>>> compAdj(k);

        // For each rule, if both vertices appear in merged, add an edge.
        for (const auto &r : rules) {
            int u, v;
            WeightType w;
            std::tie(u, v, w) = r;
            int rep_u = compMap[u];
            int rep_v = compMap[v];
            // Get re-indexed component indices.
            int cu = repToIndex[rep_u];
            int cv = repToIndex[rep_v];
            int cur_dist = merged.at(v) - merged.at(u);

            // u -> v
            compAdj[cu].push_back({cv, w, cur_dist});

            // v -> u
            compAdj[cv].push_back({cu, w, -cur_dist});
        }

        int base = repToIndex[*compSet.begin()];
        return dfsTranslations(base, 0, -1, compAdj, m_listAllSolutions);
    }
    
    // For the current merged candidate solution, generate the scaled solutions and fill the buffer.
    // Each scaled solution is produced by adding T[comp(u)'] to merged[u] for each vertex u.
    void advanceBuffer() {
        if (!has_next) return;

        std::unordered_map<int, WeightType> merged = mergeCurrentCandidate();
        std::vector<std::unordered_map<int, WeightType>> translations = scaleAllTranslations(merged);

        // For each translation vector, compute the scaled solution.
        for (const auto &T : translations) {
            std::unordered_map<int, WeightType> scaled;
            for (const auto &p : merged) {
                int u = p.first;
                // Look up DSU rep for u.
                int rep = compMap[u];
                int compIdx = repToIndex[rep];
                scaled[u] = p.second + T.at(compIdx);
            }
          buffer.push_back(scaled);
        }
        advanceIndices();
    }
    
};

int Solver::buildAdjFromEdges() {
    timer = 0;
    DSU dsu(n);
    adj.assign(n + 1, std::vector<Adj>());

    for (const Edge &edge: edges) {
        int u = edge.u, v = edge.v;
        WeightType w = edge.weight;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        dsu.unite(u, v);
    }

    return dsu.num_ccs();
}

Solver::Solver(int n, const std::vector<Edge>& edges, OptimizationSetting rootSelection, OptimizationSetting neighborSelection, bool bridgesOpt, bool listAllSolutions, bool randomize, bool knapsack, bool triangleInequality) :
            rng(std::random_device{}()), 
            ran_gen(0.0, 1.0),
            n(n),
            edges(edges),
            m_rootSelection{rootSelection},
            m_neighborSelection{neighborSelection},
            bridgesOpt(bridgesOpt),
            m_listAllSolutions(listAllSolutions),
            m_randomize(randomize),
            m_knapsack(knapsack),
            m_triangleInequality(triangleInequality)
{
    std::cerr << "DGP1 Solver started for n = " << n << ", m = " << edges.size() << '\n';
    std::cerr << "DGP1 Solver initialized with optimizations: ";
    std::cerr << (knapsack ? "ssp, " : "");
    std::cerr << (randomize ? "randomize, " : "");
    std::cerr << (bridgesOpt ? "bridges, " : "");
    std::cerr << (triangleInequality ? "triangle inequality, " : "");
    std::cerr << "\n---------------------------\n";
    std::cerr << "List all solutions is " << ((listAllSolutions) ? "enabled" : "disabled") << ".\n";
    std::cerr << "---------------------------\n";
    adj.resize(n + 1);
    dfs_tree_adj.resize(n + 1);
    vis.resize(n + 1, 0);
    value.resize(n + 1);
    dist.resize(n + 1, 0);
    tin.resize(n + 1);
    low.resize(n + 1);
    sz.resize(n + 1, 0);
    back_adj.resize(n + 1);
    forward_adj.resize(n + 1);
    parent.resize(n + 1);
    cycle_rules.resize(n + 1);
    dep.resize(n + 1, 0);
    in_cycles.resize(n + 1, 0);

    this->knapsack.resize(n + 1);
    this->binlift.resize(n + 1);
    this->path_sum.resize(n + 1);
    this->max_w.resize(n + 1);

    this->logn = 31;
    for (int i = 30; i >= 0; --i) {
        if ((1 << i) >= n) this->logn = i;
    }

    if (randomize) {
        randomize_array.resize(randomize::RANDOMIZE_ARRAY_SIZE);
        for (int i = 0; i < randomize::RANDOMIZE_ARRAY_SIZE; ++i) {
            double chance = ran_gen(rng);
            if (chance > 0.5) {
                randomize_array[i] = true;
            }
            else {
                randomize_array[i] = false;
            }
        }
    }

    // cerr << "logn: " << logn << '\n';

    for (int i = 1; i <= n; ++i) {
        this->knapsack[i].resize(logn + 1);
        this->binlift[i].resize(logn + 1, -1);
        this->path_sum[i].resize(logn + 1, 0);
        this->max_w[i].resize(logn + 1, 0);
    }

    if (buildAdjFromEdges() != 1) {
        std::cerr << "Error initializing solver: Graph is not connected\n";
        exit(1);
    }

    std::cerr << "Solver initialization done!\n";
}

void Solver::find_bridges() {
    bridges.clear();
    bridge_rules.clear();
    vis.assign(n + 1, false);
    for (int i = 1; i <= n; ++i) {
        if (!vis[i]) {
            dfs_bridges(i);
        }
    }
    vis.assign(n + 1, false);
}

void Solver::dfs_bridges(int v, int par) {
    vis[v] = 1;
    tin[v] = ++timer;
    low[v] = tin[v];
    for (auto &p : adj[v]) {
        int u = p.v;
        WeightType w = p.weight;
        if (u == par) continue;
        if (vis[u] == 1) {
            low[v] = std::min(low[v], tin[u]);
        }
        else if (vis[u] == 0) {
            dfs_bridges(u, v);
            low[v] = std::min(low[v], low[u]);
        }
        if (low[u] > tin[v]) {
            bridges.insert({v, u});
            bridge_rules.push_back({v, u, w});
        }
    }
    vis[v] = 2;
}

void Solver::dfs(int v, int par, bool can_binlift) {
    if (can_binlift) {
        parent[v] = par;
        binlift[v][0] = par;
        for (int i = 1; i <= logn; ++i) {
            if (binlift[v][i - 1] != -1) {
                int mid = binlift[v][i - 1];
                binlift[v][i] = binlift[mid][i - 1];
            }
        }
    }

    vis[v] = 1;
    
    for (auto &p : adj[v]) {
        int u = p.v;
        WeightType w = p.weight;
        if (u == par) continue;
        if (vis[u] == 1) {
            back_adj[v].push_back({u, w});

            if (m_knapsack && can_binlift) {
                int curr = parent[v];
                int dist = 1;
                while (true) {
                    if (dist > ssp::MIN_LOOKAHEAD_DEPTH) {
                        cycle_rules[curr].push_back({u, v, w});
                    }
                    if (curr == u) break;
                    curr = parent[curr];
                    ++dist;
                }
            }

            forward_adj[u].push_back({v, w});
        }
        else if (vis[u] == 0) {
            dfs_tree_adj[v].push_back({u, w});
            // dfs_tree_adj[u].push_back({v, w});
            dfs(u, v, can_binlift);
        }
    }
    vis[v] = 2;
}

void Solver::get_knapsack(int v, WeightType w_par, int par) {
    if (par != -1) {
        knapsack[v][0].set(w_par + ssp::OFFSET);
        knapsack[v][0].set(-w_par + ssp::OFFSET);

        for (int i = 1; i <= logn; ++i) {
            if (binlift[v][i - 1] != -1) {
                int mid = binlift[v][i - 1];
                // knapsack[v][i] <- knapsack[v][i - 1], knapsack[mid][i - 1]
                int v_cpy = v;
                if (knapsack[v_cpy][i - 1].count() < knapsack[mid][i - 1].count()) {
                    std::swap(v_cpy, mid);
                }
                const auto &bset1 = knapsack[mid][i - 1];
                for (int idx = bset1._Find_first(); idx < (int)bset1.size(); idx = bset1._Find_next(idx)) {
                    int shift = idx - ssp::OFFSET;
                    if (shift >= 0 && shift < ssp::WINDOW) knapsack[v][i] |= (knapsack[v_cpy][i - 1] << shift);
                    else if (shift < 0 && (-shift) < ssp::WINDOW) knapsack[v][i] |= (knapsack[v_cpy][i - 1] >> (-shift));
                }
            }
        }
    }

    for (const auto &p : dfs_tree_adj[v]) {
        int u = p.v;
        int w = (int)p.weight;
        if (u == par) continue;
        get_knapsack(u, w, v);
    }
}

void Solver::getsz(int v, int par = -1) {
    if (par != -1) {
        dep[v] = dep[par] + 1;
    }

    for (auto &p : dfs_tree_adj[v]) {
        int u = p.v;
        if (u == par) continue;
        getsz(u, v);
        sz[v] += sz[u];
    }
    sz[v]++;
}

void Solver::calculate_sum_pathw(int v, int par, WeightType w) {
    if (par != -1) {
        path_sum[v][0] = w;
        max_w[v][0] = w;
    }
    for (int i = 1; i <= logn; ++i) {
        int mid = binlift[v][i - 1];
        if (mid != -1) {
            path_sum[v][i] = path_sum[v][i - 1] + path_sum[mid][i - 1];
            max_w[v][i] = std::max(max_w[v][i - 1], max_w[mid][i - 1]);
        }
    }
    for (const auto &p : dfs_tree_adj[v]) {
        int u = p.v;
        WeightType w = p.weight;
        if (u == par) continue;
        calculate_sum_pathw(u, v, w);
    }
}

std::unordered_set<int> Solver::buildDfsTree(const std::vector<int> &idx, bool first_time) {
    std::unordered_set<int> res;
    vis.assign(n + 1, false);
    for (int i = 1; i <= n; ++i) {
        back_adj[i].clear();
        dfs_tree_adj[i].clear();
        sz[i] = 0;
        dep[i] = 0;
        cycle_rules[i].clear();
        parent[i] = -1;
        for (int j = 0; j <= logn; ++j) {
            binlift[i][j] = -1;
        }
    }
    bool can_binlift = !first_time;
    for (int i = 1; i <= n; ++i) {
        if (!vis[idx[i]]) {
            res.insert(idx[i]);
            dfs(idx[i], -1, can_binlift);
            getsz(idx[i]);
            if (can_binlift && m_knapsack) {
                std::cerr << "Starting get knapsack...\n";
                auto start_bs_count = std::chrono::high_resolution_clock::now();
                get_knapsack(idx[i], -1, -1);
                auto end_bs_count = std::chrono::high_resolution_clock::now();
                auto time_bs_count = std::chrono::duration_cast<std::chrono::milliseconds>(end_bs_count - start_bs_count).count();
                std::cerr << "Get knapsack done in " << time_bs_count << " milliseconds.\n";
            }
            if (can_binlift && m_triangleInequality) {
                calculate_sum_pathw(idx[i]);
            }
        }
    }

    this->bridged_dsu = DSU(n);
    for (auto &edge : edges) {
        int u = edge.u, v = edge.v;
        if (bridges.count({u, v}) || bridges.count({v, u})) continue;
        bridged_dsu.unite(u, v);
    }

    vis.assign(n + 1, false);
    return res;
}

std::pair<WeightType, WeightType> Solver::get_maxw_sum_path(int u, int v) {
    static std::unordered_map<std::pair<int, int>, std::pair<WeightType, WeightType>, PairHash> memoization;
    
    if (memoization.count({u, v})) {
        return memoization[{u, v}];
    }

    std::pair<WeightType, WeightType> res = {0, 0};
    if (dep[u] > dep[v]) std::swap(u, v);
    // u is ancestor of v
    for (int i = logn; i >= 0; --i) {
        if (binlift[v][i] != -1 && dep[binlift[v][i]] >= dep[u]) {
            res.first += path_sum[v][i];
            res.second = std::max(res.second, max_w[v][i]);
            v = binlift[v][i];
        }
    }
    return (memoization[{u, v}] = res);
}

int Solver::solve(std::ostream &out) {
    auto start_bs_count = std::chrono::high_resolution_clock::now();

    if (bridgesOpt) {
        find_bridges();
        std::cerr << "[Bridges] There are " << bridges.size() << " bridges in the graph.\n";
        // purge all bridge edges
        std::vector<Edge> t_edges;
        for (auto edge : edges) {
            int u = edge.u, v = edge.v;
            if (bridges.count({u, v}) || bridges.count({v, u})) continue;
            t_edges.push_back(edge);
        }
        edges = t_edges;

        buildAdjFromEdges();
    }

    std::vector<int> idx(n + 1);
    std::iota(idx.begin(), idx.end(), 0);

    std::unordered_set<int> dfs_roots = buildDfsTree(idx, true);

    // ------------------------------------------
    // calculate cycle count for each node

    std::vector<std::pair<int, int>> additional_edges;
    std::vector<int> psum(n + 1, 0);
    std::vector<int> cycle_count(n + 1, 0);
    std::set<std::pair<int, int>> leaves;

    for (int i = 1; i <= n; ++i) {
        if (dfs_tree_adj[i].size() == 1 && !dfs_roots.count(i)) {
            leaves.insert({dep[i], i});
        }
        for (const auto &p : back_adj[i]) {
            int j = p.v;
            // j is ancestor of i
            additional_edges.push_back({i, j});
        }
    }

    for (const auto &p : additional_edges) {
        int i = p.first , j = p.second;
        psum[i]++;
        if (parent[j] != -1) psum[parent[j]]--;
    }

    while (!leaves.empty()) {
        auto p = *leaves.rbegin();
        leaves.erase(p);
        int u = p.second;
        cycle_count[u] += psum[u];
        if (parent[u] != -1) {
            cycle_count[parent[u]] += cycle_count[u];
            leaves.insert({dep[parent[u]], parent[u]});
        }
    }

    auto highest_cycle_sort = [&] (int i, int j) {
        if (cycle_count[i] == cycle_count[j]) {
            return adj[i].size() > adj[j].size();
        }
        return cycle_count[i] > cycle_count[j];
    };

    auto highest_order_sort = [&] (int i, int j) {
        if (adj[i].size() == adj[j].size()) {
            return cycle_count[i] > cycle_count[j];
        }
        return adj[i].size() > adj[j].size();
    };

    if (m_rootSelection == OPT_HIGHEST_ORDER) {
        std::sort(idx.begin() + 1, idx.begin() + n + 1, highest_order_sort);
    }
    else if (m_rootSelection == OPT_HIGHEST_CYCLE) {
        std::sort(idx.begin() + 1, idx.begin() + n + 1, highest_cycle_sort);
    }

    for (int i = 1; i <= n; ++i) {
        if (m_neighborSelection == OPT_HIGHEST_ORDER) {
            std::sort(adj[i].begin(), adj[i].end(), [&] (const Adj &A, const Adj &B) {
                if (adj[A.v].size() == adj[B.v].size()) {
                    return cycle_count[A.v] > cycle_count[B.v];
                }
                return adj[A.v].size() > adj[B.v].size();
            });
        }
        else if (m_neighborSelection == OPT_HIGHEST_CYCLE) {
            std::sort(adj[i].begin(), adj[i].end(), [&] (const Adj &A, const Adj &B) {
                if (cycle_count[A.v] == cycle_count[B.v]) {
                    return adj[A.v].size() > adj[B.v].size();
                }
                return cycle_count[A.v] > cycle_count[B.v];
            });
        }
    }
    // -------------------------------------------

    std::unordered_set<int> components = buildDfsTree(idx, false);

    std::cerr << "Finished constructing DFS tree.\n";

    saveDFSTree();

    for (int i = 1; i <= n; ++i) {
        std::sort(back_adj[i].begin(), back_adj[i].end(), [&] (const Adj &A, const Adj &B) {
            return dep[A.v] > dep[B.v];
        });
    }

    if (m_triangleInequality) {
        for (int i = 1; i <= n; ++i) {
            for (const auto &p : back_adj[i]) {
                int j = p.v;
                WeightType w = p.weight;
                const std::pair<WeightType, WeightType> &pp = get_maxw_sum_path(j, i);

                WeightType mw = pp.second;
                WeightType sum = pp.first;

                sum += w;
                mw = std::max(mw, w);

                if ((2 * mw - sum) > eps) {
                    std::cerr << "[Triangle inequality] No solution found!\n";
                    return 1;
                }
            }
        }
    }

    if (m_knapsack) {
        for (int i = 1; i <= n; ++i) {
            for (const auto &p : back_adj[i]) {
                int j = p.v;
                WeightType w = p.weight;
                if (!can_knapsack(i, j, w)) {
                    std::cerr << "[SSP][Pre-DFS] Infeasible cycle detected. No solution found.\n";
                    return 1;
                }
            }
        }
        int shrunk = 0;
        for (int i = 1; i <= n; ++i) {
            
            // Strat 1: Random shuffle
            // std::shuffle(cycle_rules[i].begin(), cycle_rules[i].end(), rng);
            
            // Strat 2: sort by depth
            std::sort(cycle_rules[i].begin(), cycle_rules[i].end(), [&] (const Rule &A, const Rule &B) {
                int u_A, v_A, u_B, v_B;
                WeightType w_A, w_B;
                std::tie(u_A, v_A, w_A) = A;
                std::tie(u_B, v_B, w_B) = B;
                return dep[v_A] < dep[v_B];
            });
            if (cycle_rules[i].size() > ssp::MAX_CYCLE_RULES_SIZE) {
                ++shrunk;
                cycle_rules[i].resize(ssp::MAX_CYCLE_RULES_SIZE);
            }
        }
        std::cerr << "[SSP] [MAX_CYCLE_RULES_SIZE] Shrunk " << shrunk << " cycle_rules\n";
    }

    std::vector<std::vector<std::unordered_map<int, WeightType>>> all_res;

    for (int component : components) {
        std::optional<std::vector<std::unordered_map<int, WeightType>>> res = tryAssignAll(component, 0);
        if (!res.has_value()) {
            std::cerr << "No solution found!\n";
            return 1;
        }
        all_res.push_back(res.value());
    }

    outputCombinedResult(out, all_res);

    auto end_bs_count = std::chrono::high_resolution_clock::now();
    auto time_bs_count = std::chrono::duration_cast<std::chrono::seconds>(end_bs_count - start_bs_count).count();

    std::cerr << "Solver finished in " << time_bs_count << " seconds.\n";

    return 0;
}

void Solver::saveDFSTree() {
    std::ofstream logFile("log.txt");
    if (!logFile) {
        std::cerr << "Error: Could not open log.txt for writing!\n";
        return;
    }

    logFile << "DFS_TREE_ADJ\n";
    for (int i = 1; i <= n; ++i) {
        logFile << i << ":";
        for (auto &p : dfs_tree_adj[i]) {
            logFile << " " << p.v;
        }
        logFile << "\n";
    }

    logFile << "\nBACK_EDGES\n";
    for (int i = 1; i <= n; ++i) {
        logFile << i << ":";
        for (auto &p : back_adj[i]) {
            logFile << " " << p.v;
        }
        logFile << "\n";
    }

    logFile.close();
    std::cerr << "DFS Tree and Back Edges saved to log.txt\n";
}

bool Solver::verify_solution(const std::map<int, WeightType> &sol) {
    for (const auto &edge : edges) {
        int u = edge.u, v = edge.v;
        WeightType w = edge.weight;
        if (std::abs(std::abs(sol.at(u) - sol.at(v)) - w) > eps) return false;
    }
    return true;
}

void Solver::outputCombinedResult(std::ostream &out, const std::vector<std::vector<std::unordered_map<int, WeightType>>> &all_res, int num_solutions) {
    // algorithm: pick one from each component, merge and scale

    std::vector<int> compMap(n + 1);
    for (int i = 1; i <= n; ++i) compMap[i] = bridged_dsu.find(i);
    CombinedSolutionIterator iter(all_res, bridge_rules, compMap, m_listAllSolutions);
    int solCount = 0;
    while (iter.hasNext()) {
        std::unordered_map<int, WeightType> sol = iter.next();
        std::map<int, WeightType> sorted_sol(sol.begin(), sol.end());

        std::cerr << "Verifying solution...\n";
        if (!verify_solution(sorted_sol)) {
            std::cerr << "SOLUTION ERROR.\n";
            return;
        }
        std::cerr << "Solution verified!\n";

        out << "-------------------------\n";
        out << "Solution " << ++solCount << ":\n";
        for (const auto &p : sorted_sol) {
            out << "Vertex " << p.first << " -> " << p.second << "\n";
        }
        out << "-------------------------\n";
        std::cerr << "Saved solution " << solCount << " to file.\n";
        if (solCount == num_solutions) return;
    }
}
