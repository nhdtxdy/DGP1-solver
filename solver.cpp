#include "solver.h"
#include <numeric>
#include <queue>
#include <math.h>
#include "dsu.h"
using namespace std;

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

optional<vector<unordered_map<int, double>>> Solver::tryAssignAll(int v, double val, int par) {
    // detect back edge
    value[v] = val;
    // cerr << "v - value: " << v << ' ' << val << '\n';

    for (auto &p : back_adj[v]) {
        int u = p.v;
        double w = p.weight;
        if (fabs(fabs(value[v] - value[u]) - w) > eps) {
            // cerr << u << ' ' << v << ' ' << value[u] << ' ' << value[v] << ' ' << w << '\n';
            return nullopt;
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

        // should guarantee no bridge
        // if (bridge_opt_set && bridges.count({v, u})) continue;

        // TO STUDY: TRAVERSAL ORDER HEURISTICS (BIG CHILD?)

        vector<unordered_map<int, double>> to_merge;

        auto d1 = tryAssignAll(u, val + w, v);
        if (d1.has_value()) {
            swap(to_merge, d1.value());
            // const vector<unordered_map<int, double>> &d1_value = d1.value();
            // to_merge.insert(to_merge.end(), d1_value.begin(), d1_value.end());
        }

        if (this->m_listAllSolutions || to_merge.empty()) {
            const auto &d2 = tryAssignAll(u, val - w, v);
            if (d2.has_value()) {
                const vector<unordered_map<int, double>> &d2_value = d2.value();
                to_merge.insert(to_merge.end(), d2_value.begin(), d2_value.end());
            }
        }

        if (to_merge.empty()) {
            // cerr << v << ' ' << val << " WTF\n";
            return nullopt;
        }

        merge(merged, to_merge);
    }

    // cerr << "v - merged: " << v << '\n';
    // for (int i = 0; i < merged.size(); ++i) {
    //     for (auto &p : merged[i]) {
    //         cerr << p.first << ' ' << p.second << '\n';
    //     }
    //     cerr << '\n';
    // }
    // cerr << '\n';

    return merged;
}

// -------------------------
// DFS helper for computing translations
// -------------------------
// Here we use an unordered_map version if desired, but here we assume T is a vector
// with contiguous indices (0...k-1). (k is the number of distinct DSU representatives
// that appear in the merged solution.)
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


// -------------------------
// CombinedSolutionIterator
// -------------------------
//
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
                // cerr << "NGU\n";
                has_next = false;
                break;
             }
         }
         int n = compMap.size() - 1;
         for (int i = 1; i <= n; ++i) {
            int rep = compMap[i];
            // cerr << i << ' ' << rep << '\n';
            compSet.insert(rep);
         }
         // Map each DSU rep to a contiguous index.
         int idx = 0;
         for (int rep : compSet) {
             repToIndex[rep] = idx++;
             // cerr << "rep - idx: " << rep << ' ' << idx << '\n';
         }
         // cerr << "RTI size: " << repToIndex.size() << '\n';
         advanceBuffer(); // Preload buffer from the first merged candidate.
         // cerr << "here\n";
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
                  // We assume that candidate solutions from different components use disjoint vertex sets.
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
    
    // --------------------------
    // Scaling functions:
    // --------------------------
    //
    // Given a merged solution, we want to find all translation vectors T (one per component)
    // so that for each rule {u, v, w} (with u and v in different components)
    // we have:
    //    T[comp(u)'] - T[comp(v)'] = ± (w - (merged[u] - merged[v])),
    // where comp(u)' is the re-indexed component.
    // Since compMap is not necessarily 0-indexed, we first re-index.
    vector<unordered_map<int, double>> scaleAllTranslations(const unordered_map<int, double>& merged) {
         // Determine the set of DSU reps that occur in merged.
         int k = repToIndex.size();
         // cerr << "k = " << k << '\n';
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
        
        // cerr << "here\n";

         // Prepare translation vector T for k components, initially set to NAN.
         // Choose a base: here we choose the component corresponding to the smallest rep in compSet.
         int base = repToIndex[*compSet.begin()];
         return dfsTranslations(base, 0, -1, compAdj, m_listAllSolutions);
    }
    
    // For the current merged candidate solution, generate the scaled solutions and fill the buffer.
    // Each scaled solution is produced by adding T[comp(u)'] to merged[u] for each vertex u.
    void advanceBuffer() {
         if (!has_next) return;
         unordered_map<int, double> merged = mergeCurrentCandidate();
         vector<unordered_map<int, double>> translations = scaleAllTranslations(merged);
         // cerr << "Translation size: " << translations.size() << '\n';
         // cerr << "ADVANCE BUFFER:\n";
         // cerr << "MERGED:\n";
         // for (auto &p : merged) {
            // cerr << p.first << ' ' << p.second << '\n';
         // }
         // cerr << "compMap: " << '\n';
         // for (int i = 0; i < compMap.size(); ++i) {
            // cerr << i << ' ' << compMap[i] << '\n';
         // }
         // For each translation vector, compute the scaled solution.
         for (const auto &T : translations) {
              unordered_map<int, double> scaled;
              // For each vertex in merged, determine its re-indexed component.
              for (const auto &p : merged) {
                  int u = p.first;
                  // Look up DSU rep for u.
                  int rep = compMap[u];
                  // We need to map rep to its contiguous index.
                  // Build repToIndex mapping for merged.
                  // (Rebuild it here; alternatively, cache it.)
                  int compIdx = repToIndex[rep];
                  scaled[u] = p.second + T.at(compIdx);
              }
              buffer.push_back(scaled);
         }
         // After processing the current merged candidate, advance indices.
         advanceIndices();
    }
    
};

int Solver::buildAdjFromEdges() {
    timer = 0;
    DSU dsu(n);
    adj.assign(n + 1, vector<Adj>());
    for (const Edge &edge: edges) {
        int u = edge.u, v = edge.v;
        // cerr << "u-v: " << u << ' ' << v << '\n';
        double w = edge.weight;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        dsu.unite(u, v);
    }
    return dsu.num_ccs();
}

Solver::Solver(int n, const vector<Edge>& edges) {
    this->n = n;
    adj.resize(n + 1);
    // shortest_path_tree_adj.resize(n + 1);
    dfs_tree_adj.resize(n + 1);
    vis.resize(n + 1, 0);
    // value.resize(n + 1);
    dist.resize(n + 1, 0);
    tin.resize(n + 1);
    low.resize(n + 1);
    sz.resize(n + 1, 0);
    back_adj.resize(n + 1);
    this->edges = edges;
    if (buildAdjFromEdges() != 1) {
        cerr << "Error: Graph is not connected\n";
        exit(1);
    }
}

// void Solver::bfs(int v) {
//     vis[v] = true;
//     dist[v] = 0;
//     queue<int> que;
//     que.push(v);
//     while (!que.empty()) {
//         int t = que.front();
//         que.pop();
//         for (auto &p : adj[t]) {
//             int u = p.v;
//             double w = p.weight;
//             if (vis[u]) continue;
//             vis[u] = true;
//             dist[u] = dist[t] + 1;
//             que.push(u);
//             // t - u
//             shortest_path_tree_adj[t].push_back(p);
//             shortest_path_tree_adj[u].push_back({t, w});
//         }
//     }
// }

void Solver::find_bridges() {
    bridges.clear();
    rules.clear();
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
            // dfs_tree_adj[u].push_back({v, w});
            dfs_bridges(u, v);
            low[v] = min(low[v], low[u]);
        }
        if (low[u] > tin[v]) {
            bridges.insert({v, u});
            rules.push_back({v, u, w});
        }
    }
    vis[v] = 2;
}

void Solver::dfs(int v, int par = -1) {
    vis[v] = 1;
    
    for (auto &p : adj[v]) {
        int u = p.v;
        double w = p.weight;
        if (u == par) continue;
        if (vis[u] == 1) {
            back_adj[v].push_back({u, w});
        }
        else if (vis[u] == 0) {
            dfs_tree_adj[v].push_back({u, w});
            // dfs_tree_adj[u].push_back({v, w});
            dfs(u, v);
        }
    }
    vis[v] = 2;
}

void Solver::getsz(int v, int par = -1) {
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

int Solver::solve(ostream &out, OptimizationSetting opt, bool bridgesOpt, bool listAllSolutions) {
    // if (!listAllSolutions) {
    //     for (int i = 1; i <= n; ++i) {
    //         if (!vis[idx[i]]) {
    //             // cerr << "not visited: " << idx[i] << '\n';
    //             if (!tryAssign(idx[i], 0)) {
    //                 cerr << "Cannot find a suitable assignment\n";
    //                 return 1;
    //             }
    //         }
    //     }
    //     out << "Solution found:\n";
    //     for (int i = 1; i <= n; ++i) {
    //         out << "Vertex " << i << " - value " << value[i] << '\n';
    //     }
    //     return 0;
    // }

    this->m_listAllSolutions = listAllSolutions;

    if (bridgesOpt) {
        find_bridges();
        // purge all bridge edges
        vector<Edge> t_edges;
        for (auto edge : edges) {
            int u = edge.u, v = edge.v;
            if (bridges.count({u, v}) || bridges.count({v, u})) continue;
            t_edges.push_back(edge);
        }
        edges = t_edges;
        // cerr << "new edges:\n";
        // for (auto edge : edges) {
        //     int u = edge.u, v = edge.v;
        //     cerr << u << ' ' << v << '\n';
        // }
        // cerr << "end\n";
        buildAdjFromEdges();
    }

    this->bridge_opt_set = bridgesOpt;
    vector<int> idx = getOrder(opt);
    vector<int> components = buildDfsTree(idx);

    vector<vector<unordered_map<int, double>>> all_res;

    for (int component : components) {
        // cerr << "try: " << component << '\n';
        optional<vector<unordered_map<int, double>>> res = tryAssignAll(component, 0);
        if (!res.has_value()) {
            cerr << "No solution found!\n";
            exit(1);
        }
        all_res.push_back(res.value());
    }

    outputCombinedResult(out, all_res);

    // out << "Solutions found:\n\n";
    // for (int i = 0; i < (int)res.size(); ++i) {
    //     out << "Solution #" << i + 1 << '\n';
    //     for (int j = 1; j <= n; ++j) {
    //         out << "Vertex " << j << " - value " << res[i][j] << '\n';
    //     }
    //     out << '\n';
    // }

    return 0;
}

void Solver::outputCombinedResult(ostream &out, const vector<vector<unordered_map<int, double>>> &all_res, int num_solutions) {
    // algorithm: pick one from each component, merge and scale

    // cerr << all_res.size() << '\n';
    // cerr << all_res[0].size() << '\n';

    // for (int i = 0; i < all_res[0].size(); ++i) {
    //     for (auto &p : all_res[0][i]) {
    //         cerr << p.first << ' ' << p.second << '\n';
    //     }
    //     cerr << '\n';
    // }

    vector<int> compMap(n + 1);
    for (int i = 1; i <= n; ++i) compMap[i] = bridged_dsu.find(i);
    CombinedSolutionIterator iter(all_res, rules, compMap, m_listAllSolutions);
    int solCount = 0;
    while (iter.hasNext()) {
        unordered_map<int, double> sol = iter.next();
        out << "-------------------------\n";
        out << "Solution " << ++solCount << ":\n";
        for (const auto &p : sol) {
            out << "Vertex " << p.first << " -> " << p.second << "\n";
        }
        out << "-------------------------\n";
        cerr << "Saved solution " << solCount << " to file.\n";
        if (solCount == num_solutions) return;
    }
}
