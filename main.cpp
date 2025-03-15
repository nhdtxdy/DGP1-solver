#include "solver.h"
#include <bits/stdc++.h>
using namespace std;

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
    ios::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_filename> [output_filename] [optimizations] [--list-all-solutions]\n";
        return 1;
    }

    #ifdef USE_INTEGER_WEIGHTS
        cerr << "[WARNING] DGP1 solver is compiled with ONLY integer weights support.\n";
    #endif

    string inputFilename = argv[1];
    bool outputFileSet = false;
    string outputFilename;
    
    OptimizationSetting rootSelection = OPT_DEFAULT, neighborSelection = OPT_DEFAULT;
    bool bridgesOpt = false;
    bool randomize = false;
    bool listAllSolutions = false;
    bool knapsack = false;
    bool triangleInequality = false;
    
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg.starts_with("--optimizations=")) {
            string settings = arg.substr(16); // Extract the part after "--optimizations="
            stringstream ss(settings);
            string token;
            while (getline(ss, token, ',')) {
                token = trim(token);
                if (token == "bridges") {
                    bridgesOpt = true;
                } else if (token == "randomize") {
                    randomize = true;
                } else if (token == "knapsack") {
                    knapsack = true;
                } else if (token == "triangle") {
                    triangleInequality = true;
                } else {
                    cerr << "Unknown optimization option in --optimizations: " << token << "\n";
                    return 1;
                }
            }
        } else if (arg.starts_with("--root=")) {
            string token = arg.substr(7);
            if (token == "default") {
                rootSelection = OPT_DEFAULT;
            }
            else if (token == "highest-order") {
                rootSelection = OPT_HIGHEST_ORDER;
            }
            else if (token == "highest-cycle") {
                rootSelection = OPT_HIGHEST_CYCLE;
            }
            else {
                cerr << "Unknown root selection strategy: " << token << '\n';
                return 1;
            }
        } else if (arg.starts_with("--neighbors=")) {
            string token = arg.substr(12);
            if (token == "default") {
                neighborSelection = OPT_DEFAULT;
            }
            else if (token == "highest-order") {
                neighborSelection = OPT_HIGHEST_ORDER;
            }
            else if (token == "highest-cycle") {
                neighborSelection = OPT_HIGHEST_CYCLE;
            }
            else {
                cerr << "Unknown neighbor selection strategy: " << token << '\n';
                return 1;
            }
        } else if (arg == "--list-all-solutions") {
            listAllSolutions = true;
        } else {
            if (!outputFileSet) {
                outputFilename = arg;
                outputFileSet = true;
            } else {
                cerr << "Unrecognized argument: " << arg << "\n";
                return 1;
            }
        }
    }

    ifstream file(inputFilename);
    if (!file) {
        cerr << "Error: Could not open file " << inputFilename << "\n";
        return 1;
    }

    string line;
    int n = 0;

    auto checkValidVertex = [&] (int v) {
        return 1 <= v && v <= n;
    };

    int lineno = 0;
    vector<Edge> edges;

    while (getline(file, line)) {
        ++lineno;
        trimComment(line); 
        line = trim(line);
        if (line.empty()) continue;

        istringstream iss(line);
        string token;
        iss >> token;

        if (token == "param") {
            iss >> token;
            if (token == "n") {
                string eq;
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

                        istringstream edgeStream(line);
                        int u, v, I;
                        WeightType c;
                        if (edgeStream >> u >> v >> c >> I) {
                            if (!checkValidVertex(u) || !checkValidVertex(v)) {
                                cerr << "Error on line " << lineno << ": invalid vertex!\n";
                                return 0;
                            }
                            edges.push_back({u, v, c});
                        }
                    }
                }
            }
        }
    }

    file.close();

    Solver solver(n, edges, rootSelection, neighborSelection, bridgesOpt, listAllSolutions, randomize, knapsack, triangleInequality);

    if (outputFileSet) {
        ofstream outfile(outputFilename);
        if (!outfile) {
            cerr << "Error: Could not open output file " << outputFilename << "\n";
            return 1;
        }
        
        solver.solve(outfile);
        outfile.close();
    }
    else {
        solver.solve(cout);
    }
   
    return 0;
}
