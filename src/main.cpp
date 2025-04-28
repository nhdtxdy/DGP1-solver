#include "solver.h"
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>

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
        std::cerr << "Usage: " << argv[0] << " <input_filename> [output_filename] [optimizations] [--list-all-solutions]\n";
        return 1;
    }

    #ifdef USE_INTEGER_WEIGHTS
        std::cerr << "[WARNING] DGP1 solver is compiled with ONLY integer weights support.\n";
    #endif

    std::string inputFilename = argv[1];
    bool outputFileSet = false;
    std::string outputFilename;
    
    OptimizationSetting rootSelection = OPT_DEFAULT, neighborSelection = OPT_DEFAULT;
    bool bridgesOpt = false;
    bool randomize = false;
    bool listAllSolutions = false;
    bool knapsack = false;
    bool triangleInequality = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.starts_with("--optimizations=")) {
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
                } else if (token == "triangle") {
                    triangleInequality = true;
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
            else {
                std::cerr << "Unknown neighbor selection strategy: " << token << '\n';
                return 1;
            }
        } else if (arg == "--list-all-solutions") {
            listAllSolutions = true;
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
                                if (std::abs(floor(c) - c) > eps) {
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

    if (knapsack) {
        assert(std::is_integral<WeightType>::value && std::is_signed<WeightType>::value &&
            "Error: SSP optimization requires the program to be compiled with USE_INTEGER_WEIGHTS.");
    }

    Solver solver(n, edges, rootSelection, neighborSelection, bridgesOpt, listAllSolutions, randomize, knapsack, triangleInequality);

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
