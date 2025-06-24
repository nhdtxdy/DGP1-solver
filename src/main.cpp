#include "solver.h"
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include "config.h"
#include <algorithm>
#include <cstdlib>

void writeMiniZincModel(const std::string &filename, int n, const std::vector<Edge> &edges) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open MiniZinc output file " << filename << "\n";
        return;
    }

    WeightType B_value = 0;
    for (const auto &e : edges) {
        B_value = std::max<WeightType>(B_value, static_cast<WeightType>(std::abs(e.weight)));
    }

    if (B_value == 0) B_value = 1; // Ensure the domain is non-empty.

    out << "int: B = " << B_value << ";\n";
    out << "int: n = " << n << ";\n";
    out << "array[1..n] of var 0..B: P;\n\n";

    for (const auto &e : edges) {
        out << "constraint abs(P[" << e.u << "] - P[" << e.v << "]) = " << e.weight << ";\n";
    }

    out << "\nsolve satisfy;\n";

    out.close();
}

void writeGurobiModel(const std::string &filename, int n, const std::vector<Edge> &edges) {
    const int DOMAIN_MAX = 14;  // Values are in [0,14]

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open Gurobi output file " << filename << "\n";
        return;
    }

    // Dummy objective – we only care about feasibility.
    out << "Minimize\n  0\n\n";

    out << "Subject To\n";

    // 1. Each vertex gets exactly one value.
    for (int i = 1; i <= n; ++i) {
        out << "  assign_" << i << ": ";
        for (int v = 0; v <= DOMAIN_MAX; ++v) {
            if (v) out << " + ";
            out << "p" << i << "_" << v;
        }
        out << " = 1\n";
    }

    // 2. Edge constraints – forbid value pairs whose difference is not equal to the required weight.
    size_t cid = 0;
    for (const auto &e : edges) {
        int w = static_cast<int>(std::abs(e.weight));
        for (int vu = 0; vu <= DOMAIN_MAX; ++vu) {
            for (int vv = 0; vv <= DOMAIN_MAX; ++vv) {
                if (std::abs(vu - vv) == w) continue; // allowed pairs – skip
                out << "  c" << cid++ << ": p" << e.u << "_" << vu << " + p" << e.v << "_" << vv << " <= 1\n";
            }
        }
    }

    // 3. OPTIONAL: fix P[1] = 0 to remove translation symmetry.
    if (n > 0) {
        out << "  fix_root: p1_0 = 1\n";
    }

    out << "\nBinary\n";
    for (int i = 1; i <= n; ++i) {
        for (int v = 0; v <= DOMAIN_MAX; ++v) {
            out << "  p" << i << "_" << v << "\n";
        }
    }

    out << "\nEnd\n";
    out.close();
}

void trimComment(std::string &line) {
    size_t pos = line.find('#');
    if (pos != std::string::npos) {
        line = line.substr(0, pos);
    }
}

std::string trim(const std::string &s) {
    size_t first = s.find_first_not_of(" \t");
    if (first == std::string::npos)
        return "";
    size_t last = s.find_last_not_of(" \t");
    return s.substr(first, (last - first + 1));
}

int main(int argc, char *argv[]) {
    // Intended to make I/O faster as we do not use C-style I/O here.
    // But disabling sync_with_stdio is not thread-safe, so this needs
    // to change if multithreading optimizations are applied in the future.
    std::ios::sync_with_stdio(false);
    std::cin.tie(0); std::cout.tie(0);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename> [output_filename] [optimizations] [--list-all-solutions] [--export-mzn=<filename>] [--export-gurobi=<filename>]\n";
        return 1;
    }

    #ifdef USE_INTEGER_WEIGHTS
        std::cerr << "[WARNING] DGP1 solver is compiled with ONLY integer weights support.\n";
    #endif

    std::string inputFilename = argv[1];
    bool outputFileSet = false;
    std::string outputFilename;
    std::string configFilename;
    
    OptimizationSetting rootSelection = OPT_DEFAULT, neighborSelection = OPT_DEFAULT;
    bool bridgesOpt = false;
    bool randomize = false;
    bool listAllSolutions = false;
    bool knapsack = false;
    bool triangleInequality = false;
    bool preprocessKnapsack = false;
    bool bounds = false;
    bool exportMzn = false;
    std::string mznFilename;
    bool exportGurobi = false;
    std::string gurobiFilename;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.starts_with("--config=")) {
            configFilename = arg.substr(9);
        } else if (arg.starts_with("--optimizations=")) {
            std::string settings = arg.substr(16); // Extract the part after "--optimizations="
            std::stringstream ss(settings);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (token == "bridges") {
                    bridgesOpt = true;
                } else if (token == "randomize") {
                    randomize = true;
                } else if (token == "ssp") {
                    knapsack = true;
                } else if (token == "ssp-preprocess") {
                    preprocessKnapsack = true;
                } else if (token == "triangle") {
                    triangleInequality = true;
                } else if (token == "bounds") {
                    bounds = true;
                } else {
                    std::cerr << "Unknown optimization option in --optimizations: " << token << "\n";
                    return 1;
                }
            }
        } else if (arg.starts_with("--root=")) {
            std::string token = arg.substr(7);
            if (token == "default") {
                rootSelection = OPT_DEFAULT;
            }
            else if (token == "highest-order") {
                rootSelection = OPT_HIGHEST_ORDER;
            }
            else if (token == "highest-cycle") {
                rootSelection = OPT_HIGHEST_CYCLE;
            }
            else if (token == "highest-score") {
                rootSelection = OPT_HIGHEST_SCORE;
            }
            else {
                std::cerr << "Unknown root selection strategy: " << token << '\n';
                return 1;
            }
        } else if (arg.starts_with("--neighbors=")) {
            std::string token = arg.substr(12);
            if (token == "default") {
                neighborSelection = OPT_DEFAULT;
            }
            else if (token == "highest-order") {
                neighborSelection = OPT_HIGHEST_ORDER;
            }
            else if (token == "highest-cycle") {
                neighborSelection = OPT_HIGHEST_CYCLE;
            }
            else if (token == "highest-score") {
                neighborSelection = OPT_HIGHEST_SCORE;
            }
            else {
                std::cerr << "Unknown neighbor selection strategy: " << token << '\n';
                return 1;
            }
        } else if (arg == "--list-all-solutions") {
            listAllSolutions = true;
        } else if (arg.starts_with("--export-mzn=")) {
            exportMzn = true;
            mznFilename = arg.substr(13); // Extract filename after the '='
        } else if (arg.starts_with("--export-gurobi=")) {
            exportGurobi = true;
            gurobiFilename = arg.substr(16);
        } else {
            if (!outputFileSet) {
                outputFilename = arg;
                outputFileSet = true;
            } else {
                std::cerr << "Unrecognized argument: " << arg << "\n";
                return 1;
            }
        }
    }

    if (!configFilename.empty()) {
        config.loadFromFile(configFilename);
    }

    // Print the final configuration that will be used by the solver
    config.printConfig();

    std::ifstream file(inputFilename);
    if (!file) {
        std::cerr << "Error: Could not open file " << inputFilename << "\n";
        return 1;
    }

    std::string line;
    int n = 0;

    auto checkValidVertex = [&] (int v) {
        return 1 <= v && v <= n;
    };

    int lineno = 0;
    std::vector<Edge> edges;

    while (std::getline(file, line)) {
        ++lineno;
        trimComment(line); 
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "param") {
            iss >> token;
            if (token == "n") {
                std::string eq;
                iss >> eq;
                iss >> n;
            } else if (token == ":") {
                iss >> token;
                if (token == "E") {
                    while (getline(file, line)) {
                        ++lineno;
                        trimComment(line);
                        line = trim(line);
                        if (line.empty()) continue;
                        if (line == ";") break;

                        std::istringstream edgeStream(line);
                        int u, v, I;
                        double c;
                        if (edgeStream >> u >> v >> c >> I) {
                            if (!checkValidVertex(u) || !checkValidVertex(v)) {
                                std::cerr << "Error on line " << lineno << ": invalid vertex!\n";
                                return 0;
                            }
                            #ifdef USE_INTEGER_WEIGHTS
                                if (std::abs(floor(c) - c) > config.eps) {
                                    std::cerr << "[ERROR] Floating-point value detected in input graph!\n";
                                    return 1;
                                }
                            #endif
                            edges.push_back({u, v, (WeightType)c});
                        }
                    }
                }
            }
        }
    }

    file.close();

    if (exportMzn) {
        if (mznFilename.empty()) {
            mznFilename = "graph.mzn"; // default filename if none provided
        }
        writeMiniZincModel(mznFilename, n, edges);
        // return 0;
    }

    if (exportGurobi) {
        if (gurobiFilename.empty()) {
            gurobiFilename = "graph.lp";
        }
        writeGurobiModel(gurobiFilename, n, edges);
        // return 0;
    }

    if (exportMzn || exportGurobi) {
        return 0;
    }

    if (knapsack) {
        assert(std::is_integral<WeightType>::value && std::is_signed<WeightType>::value &&
            "Error: SSP optimization requires the program to be compiled with USE_INTEGER_WEIGHTS.");
    }

    Solver solver(n, edges, rootSelection, neighborSelection, bridgesOpt, listAllSolutions, randomize, knapsack, triangleInequality, preprocessKnapsack, bounds);

    int ret = 0;

    if (outputFileSet) {
        std::ofstream outfile(outputFilename);
        if (!outfile) {
            std::cerr << "Error: Could not open output file " << outputFilename << "\n";
            return 1;
        }
        
        ret = solver.solve(outfile);
        outfile.close();
    }
    else {
        ret = solver.solve(std::cout);
    }
   
    return ret;
}
