#include "solver.h"
#include <numeric>
#include <queue>
#include <math.h>
#include "dsu.h"
#include <map>
#include <fstream>
#include <chrono>
using namespace std;

// merge assignments from two branches
void merge(vector<unordered_map<int, double>> &base, vector<unordered_map<int, double>> to_merge) {
    vector<unordered_map<int, double>> merged;
    for (const unordered_map<int, double>& b : base) {
        for (const unordered_map<int, double> &tm : to_merge) {
            unordered_map<int, double> cpy = b;
            cpy.insert(tm.begin(), tm.end());
            merged.push_back(cpy);
        }
    }
    swap(base, merged);
}

// merge two knapsack bitsets (for knapsack optimization)
bitset<WINDOW> bitset_merge(const bitset<WINDOW> &A, const bitset<WINDOW> &B) {
    bitset<WINDOW> const *ptrA = &A;
    bitset<WINDOW> const *ptrB = &B;
    if (A.count() < B.count()) {
        swap(ptrA, ptrB);
    }
    // A > B
    bitset<WINDOW> res;
    for (int idx = ptrB->_Find_first(); idx < (int)ptrB->size(); idx = ptrB->_Find_next(idx)) {
        int shift = idx - OFFSET;
        if (shift >= 0 && shift < WINDOW) res |= ((*ptrA) << shift);
        else if (shift < 0 && (-shift) < WINDOW) res |= ((*ptrA) >> (-shift));
    }

    return res;
}

// checks if there exists an assignment with sum "value" using edges on the path from v to u
bool Solver::can_knapsack(int u, int v, int value) {
    bitset<WINDOW> bset;
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

    return bset[value + OFFSET];
}

optional<vector<unordered_map<int, double>>> Solver::tryAssignAll(int v, double val, int par) {
    if (m_knapsack && (val > M_LIMIT || val < -M_LIMIT)) return nullopt;
    value[v] = val;

    // detect invalid cycle through back edges
    for (auto &p : back_adj[v]) {
        int u = p.v;
        double w = p.weight;
        if (fabs(fabs(value[v] - value[u]) - w) > eps) {
            return nullopt;
        }
    }

    if (m_knapsack) {
        for (const Rule &rule : cycle_rules[v]) {
            int x, y;
            double w_;
            tie(x, y, w_) = rule;
            int w = (int)w_;

            // value[x] is known
            int potential_y1 = (int)value[x] + w;
            int potential_y2 = (int)value[x] - w;

            if (!(-M_LIMIT <= potential_y1 && potential_y1 <= M_LIMIT && can_knapsack(y, v, potential_y1 - val)) && !(-M_LIMIT <= potential_y2 && potential_y2 <= M_LIMIT && can_knapsack(y, v, potential_y2 - val))) {
                return nullopt;
            }
        }
    }

    vector<unordered_map<int, double>> merged;
    unordered_map<int, double> single_map;
    single_map[v] = val;
    merged.push_back(single_map);

    for (auto &p : dfs_tree_adj[v]) {
        int u = p.v;
        double w = p.weight;
        if (u == par) continue;

        // 50% chance to flip the sign of w
        if (m_randomize) {
            double chance = ran_gen(rng);
            if (chance > 0.5) w = -w;
        }


        // TO STUDY: TRAVERSAL ORDER HEURISTICS (BIG CHILD IN SMALL-TO-LARGE TECHNIQUE?)

        vector<unordered_map<int, double>> to_merge;

        auto d1 = tryAssignAll(u, val + w, v);
        if (d1.has_value()) {
            // swap is O(1) while insert is O(n)
            swap(to_merge, d1.value());
        }

        if (this->m_listAllSolutions || to_merge.empty()) {
            const auto &d2 = tryAssignAll(u, val - w, v);
            if (d2.has_value()) {
                const vector<unordered_map<int, double>> &d2_value = d2.value();
                to_merge.insert(to_merge.end(), d2_value.begin(), d2_value.end());
            }
        }

        if (to_merge.empty()) {
            return nullopt;
        }

        merge(merged, to_merge);
    }

    return merged;
}

// DFS helper for computing translations (in case of bridges optimization)
vector<unordered_map<int, double>> dfsTranslations(int v,
                    double val,
                    int par,
                    const vector<vector<tuple<int, double, double>>>& compAdj,
                    bool listAllSolutions)
{
    vector<unordered_map<int, double>> merged;
    unordered_map<int, double> single_map;
    single_map[v] = val;
    merged.push_back(single_map);

    for (auto &p : compAdj[v]) {
        int u;
        double w;
        double cur_dist;
        tie(u, w, cur_dist) = p;
        if (u == par) continue;

        vector<unordered_map<int, double>> to_merge;

        // v -> u: cur_dist, want w
        auto d1_value = dfsTranslations(u, val + (w - cur_dist), v, compAdj, listAllSolutions);
        swap(to_merge, d1_value);

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
    //            Each candidate solution is an unordered_map<int,double> (vertex -> base value).
    //   rules: a vector of rules; each rule is a tuple {u, v, w}.
    //   compMap: mapping from vertex to its DSU representative (1-indexed, not necessarily contiguous).
    //            (Note: all_res is assumed to be ordered by these re-indexed components.)
    CombinedSolutionIterator(
         const vector<vector<unordered_map<int, double>>>& all_res,
         const vector<Rule>& rules,
         const vector<int>& compMap,
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
    unordered_map<int, double> next() {
         if (buffer.empty()) {
             advanceBuffer();
         }
         unordered_map<int, double> sol = buffer.front();
         buffer.erase(buffer.begin());
         return sol;
    }
    
private:
    const vector<vector<unordered_map<int, double>>>& all_res;
    const vector<tuple<int, int, double>>& rules;
    const vector<int>& compMap;  // DSU representative for each vertex (1-indexed).
    vector<int> indices;         // current candidate index for each component.
    bool has_next, m_listAllSolutions;
    vector<unordered_map<int, double>> buffer;  // buffer holding scaled solutions for current merged candidate.
    
    set<int> compSet;
    unordered_map<int, int> repToIndex;

    // Merge candidate solutions from each component according to current indices.
    unordered_map<int, double> mergeCurrentCandidate() {
         unordered_map<int, double> merged;
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
    vector<unordered_map<int, double>> scaleAllTranslations(const unordered_map<int, double>& merged) {
        int k = repToIndex.size();

        // Build the component adjacency list.
        vector<vector<tuple<int, double, double>>> compAdj(k);

        // For each rule, if both vertices appear in merged, add an edge.
        for (const auto &r : rules) {
          int u, v;
          double w;
          tie(u, v, w) = r;
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

        unordered_map<int, double> merged = mergeCurrentCandidate();
        vector<unordered_map<int, double>> translations = scaleAllTranslations(merged);

        // For each translation vector, compute the scaled solution.
        for (const auto &T : translations) {
          unordered_map<int, double> scaled;
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
    adj.assign(n + 1, vector<Adj>());

    for (const Edge &edge: edges) {
        int u = edge.u, v = edge.v;
        double w = edge.weight;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        dsu.unite(u, v);
    }

    return dsu.num_ccs();
}

Solver::Solver(int n, const vector<Edge>& edges, OptimizationSetting opt, bool bridgesOpt, bool listAllSolutions, bool randomize, bool knapsack) :
            rng(random_device{}()), 
            ran_gen(0.0, 1.0),
            n(n),
            edges(edges),
            opt(opt),
            bridgesOpt(bridgesOpt),
            m_listAllSolutions(listAllSolutions),
            m_randomize(randomize),
            m_knapsack(knapsack)
{
    cerr << "DGP1 Solver initialized with optimizations: ";
    cerr << (knapsack ? "knapsack, " : "");
    cerr << (randomize ? "randomize, " : "");
    cerr << (bridgesOpt ? "bridges, " : "");
    cerr << ((opt == OPT_HIGHEST_ORDER) ? "prioritize-high-degree-vertices, " : ((opt == OPT_LOWEST_ORDER) ? "prioritize-low-degree-vertices, " : ""));
    cerr << "\n---------------------------\n";
    cerr << "List all solutions is " << ((listAllSolutions) ? "enabled" : "disabled") << ".\n";
    cerr << "---------------------------\n";
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

    this->knapsack.resize(n + 1);
    this->binlift.resize(n + 1);

    this->logn = 31;
    for (int i = 30; i >= 0; --i) {
        if ((1 << i) >= n) this->logn = i;
    }

    // cerr << "logn: " << logn << '\n';

    for (int i = 1; i <= n; ++i) {
        this->knapsack[i].resize(logn + 1);
        this->binlift[i].resize(logn + 1, -1);
    }

    if (buildAdjFromEdges() != 1) {
        cerr << "Error initializing solver: Graph is not connected\n";
        exit(1);
    }

    cerr << "Solver initialization done!\n";
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
        double w = p.weight;
        if (u == par) continue;
        if (vis[u] == 1) {
            low[v] = min(low[v], tin[u]);
        }
        else if (vis[u] == 0) {
            dfs_bridges(u, v);
            low[v] = min(low[v], low[u]);
        }
        if (low[u] > tin[v]) {
            bridges.insert({v, u});
            bridge_rules.push_back({v, u, w});
        }
    }
    vis[v] = 2;
}

void Solver::dfs(int v, int par = -1) {
    parent[v] = par;
    binlift[v][0] = par;
    for (int i = 1; i <= logn; ++i) {
        if (binlift[v][i - 1] != -1) {
            int mid = binlift[v][i - 1];
            binlift[v][i] = binlift[mid][i - 1];
        }
    }

    vis[v] = 1;
    
    for (auto &p : adj[v]) {
        int u = p.v;
        double w = p.weight;
        if (u == par) continue;
        if (vis[u] == 1) {
            back_adj[v].push_back({u, w});

            if (m_knapsack) {
                int curr = parent[v];
                while (true) {
                    cycle_rules[curr].push_back({u, v, w});
                    if (curr == u) break;
                    curr = parent[curr];
                }
            }

            forward_adj[u].push_back({v, w});
        }
        else if (vis[u] == 0) {
            dfs_tree_adj[v].push_back({u, w});
            // dfs_tree_adj[u].push_back({v, w});
            dfs(u, v);
        }
    }
    vis[v] = 2;
}

void Solver::get_knapsack(int v, int w_par, int par) {
    if (par != -1) {
        knapsack[v][0].set(w_par + OFFSET);
        knapsack[v][0].set(-w_par + OFFSET);

        for (int i = 1; i <= logn; ++i) {
            if (binlift[v][i - 1] != -1) {
                int mid = binlift[v][i - 1];
                // knapsack[v][i] <- knapsack[v][i - 1], knapsack[mid][i - 1]
                int v_cpy = v;
                if (knapsack[v_cpy][i - 1].count() < knapsack[mid][i - 1].count()) {
                    swap(v_cpy, mid);
                }
                const auto &bset1 = knapsack[mid][i - 1];
                for (int idx = bset1._Find_first(); idx < (int)bset1.size(); idx = bset1._Find_next(idx)) {
                    int shift = idx - OFFSET;
                    if (shift >= 0 && shift < WINDOW) knapsack[v][i] |= (knapsack[v_cpy][i - 1] << shift);
                    else if (shift < 0 && (-shift) < WINDOW) knapsack[v][i] |= (knapsack[v_cpy][i - 1] >> (-shift));
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

vector<int> Solver::buildDfsTree(const vector<int> &idx) {
    vector<int> res;
    vis.assign(n + 1, false);
    for (int i = 1; i <= n; ++i) {
        if (!vis[idx[i]]) {
            res.push_back(idx[i]);
            dfs(idx[i]);
            getsz(idx[i]);
            if (m_knapsack) {
                cerr << "Starting get knapsack...\n";
                auto start_bs_count = std::chrono::high_resolution_clock::now();
                get_knapsack(idx[i], -1, -1);
                auto end_bs_count = std::chrono::high_resolution_clock::now();
                auto time_bs_count = std::chrono::duration_cast<std::chrono::milliseconds>(end_bs_count - start_bs_count).count();
                cerr << "Get knapsack done in " << time_bs_count << " milliseconds.\n";
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

vector<int> Solver::getOrder(OptimizationSetting opt) {
    vector<int> order(n + 1);
    iota(order.begin(), order.end(), 0);
    vector<int> deg(n + 1);
    for (int i = 1; i <= n; ++i) {
        deg[i] = adj[i].size();
        if (opt == OPT_HIGHEST_ORDER) sort(adj[i].begin(), adj[i].end(), [&] (const Adj &a, const Adj &b) {
            return deg[a.v] > deg[b.v];
        });
        else if (opt == OPT_LOWEST_ORDER) sort(adj[i].begin(), adj[i].end(), [&] (const Adj &a, const Adj &b) {
            return deg[a.v] < deg[b.v];
        });
    }
    if (opt == OPT_HIGHEST_ORDER) sort(order.begin() + 1, order.begin() + n + 1, [&] (int a, int b) {
        return deg[a] > deg[b];
    });
    else if (opt == OPT_LOWEST_ORDER) sort(order.begin() + 1, order.begin() + n + 1, [&] (int a, int b) {
        return deg[a] < deg[b];
    });

    return order;
}

int Solver::solve(ostream &out) {
    auto start_bs_count = chrono::high_resolution_clock::now();

    if (bridgesOpt) {
        find_bridges();
        cerr << "There are " << bridges.size() << " bridges in the graph.\n";
        // purge all bridge edges
        vector<Edge> t_edges;
        for (auto edge : edges) {
            int u = edge.u, v = edge.v;
            if (bridges.count({u, v}) || bridges.count({v, u})) continue;
            t_edges.push_back(edge);
        }
        edges = t_edges;

        buildAdjFromEdges();
    }

    vector<int> idx = getOrder(opt);
    vector<int> components = buildDfsTree(idx);

    saveDFSTree();

    vector<vector<unordered_map<int, double>>> all_res;

    for (int component : components) {
        optional<vector<unordered_map<int, double>>> res = tryAssignAll(component, 0);
        if (!res.has_value()) {
            cerr << "No solution found!\n";
            return 1;
        }
        all_res.push_back(res.value());
    }

    outputCombinedResult(out, all_res);

    auto end_bs_count = chrono::high_resolution_clock::now();
    auto time_bs_count = chrono::duration_cast<chrono::seconds>(end_bs_count - start_bs_count).count();

    cerr << "Solver finished in " << time_bs_count << " seconds.\n";

    return 0;
}

void Solver::saveDFSTree() {
    ofstream logFile("log.txt");
    if (!logFile) {
        cerr << "Error: Could not open log.txt for writing!\n";
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
    cerr << "DFS Tree and Back Edges saved to log.txt\n";
}

bool Solver::verify_solution(const map<int, double> &sol) {
    for (const auto &edge : edges) {
        int u = edge.u, v = edge.v;
        double w = edge.weight;
        if (fabs(fabs(sol.at(u) - sol.at(v)) - w) > eps) return false;
    }
    return true;
}

void Solver::outputCombinedResult(ostream &out, const vector<vector<unordered_map<int, double>>> &all_res, int num_solutions) {
    // algorithm: pick one from each component, merge and scale

    vector<int> compMap(n + 1);
    for (int i = 1; i <= n; ++i) compMap[i] = bridged_dsu.find(i);
    CombinedSolutionIterator iter(all_res, bridge_rules, compMap, m_listAllSolutions);
    int solCount = 0;
    while (iter.hasNext()) {
        unordered_map<int, double> sol = iter.next();
        map<int, double> sorted_sol(sol.begin(), sol.end());

        cerr << "Verifying solution...\n";
        if (!verify_solution(sorted_sol)) {
            cerr << "SOLUTION ERROR.\n";
            return;
        }
        cerr << "Solution verified!\n";

        out << "-------------------------\n";
        out << "Solution " << ++solCount << ":\n";
        for (const auto &p : sorted_sol) {
            out << "Vertex " << p.first << " -> " << p.second << "\n";
        }
        out << "-------------------------\n";
        cerr << "Saved solution " << solCount << " to file.\n";
        if (solCount == num_solutions) return;
    }
}

// bool Solver::tryAssign(int v, double val = 0) {
//     // cerr << "Visit " << v << ' ' << val << '\n';
//     vis[v] = true;
//     value[v] = val;
//     bool ok = true;
//     for (auto &p : adj[v]) {
//         int u = p.v;
//         double w = p.weight;
//         if (vis[u] && fabs(fabs(value[v] - value[u]) - w) > eps) {
//             ok = false;
//             break;
//         }
//         if (!vis[u]) {
//             if (!tryAssign(u, val + w) && !tryAssign(u, val - w)) {
//                 ok = false;
//                 break;
//             }
//         }
//     }
//     if (ok) return true;
//     vis[v] = false;
//     return ok;
// }