#include <iostream>
#include <fstream>
#include <vector>
#include "dsu.h"
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <iomanip>

struct Edge {
    int u, v;
    double weight;
};

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

bool readGraph(const std::string &inputFilename, int &n, std::vector<Edge> &edges) {
    std::ifstream file(inputFilename);
    if (!file) {
        std::cerr << "Error: Could not open file " << inputFilename << "\n";
        return false;
    }

    std::string line;
    int lineno = 0;
    n = 0;
    auto checkValidVertex = [&] (int v) {
        return 1 <= v && v <= n;
    };

    while (getline(file, line)) {
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
                iss >> eq; // should be ":="
                iss >> n;
            } else if (token == ":") {
                iss >> token;
                if (token == "E") {
                    // Read the edges until we hit a line with ";".
                    while (getline(file, line)) {
                        ++lineno;
                        trimComment(line);
                        line = trim(line);
                        if (line.empty()) continue;
                        if (line == ";")
                            break;
                        std::istringstream edgeStream(line);
                        int u, v, I;
                        double c;
                        if (edgeStream >> u >> v >> c >> I) {
                            if (!checkValidVertex(u) || !checkValidVertex(v)) {
                                std::cerr << "Error on line " << lineno << ": invalid vertex!\n";
                                return false;
                            }
                            edges.push_back({u, v, c});
                        }
                    }
                }
            }
        }
    }
    file.close();
    return true;
}

// Splits the graph into connected components, renumbers vertices, and writes each component to a file.
void splitutil(const std::string &inputFilename) {
    int n;
    std::vector<Edge> edges;
    if (!readGraph(inputFilename, n, edges))
        return;

    // Use DSU to determine connected components.
    DSU dsu(n);
    for (const auto &edge : edges) {
        dsu.unite(edge.u, edge.v);
    }

    // Group vertices by their DSU root.
    std::map<int, std::vector<int>> components;
    for (int i = 1; i <= n; ++i) {
        int root = dsu.find(i);
        components[root].push_back(i);
    }

    int compNum = 0;
    for (const auto &comp : components) {
        ++compNum;
        std::vector<int> oldVertices = comp.second;
        std::sort(oldVertices.begin(), oldVertices.end());

        // Create a mapping from old vertex IDs to new IDs (starting from 1).
        std::map<int, int> newId;
        int newCounter = 1;
        for (int v : oldVertices)
            newId[v] = newCounter++;

        // Filter edges that belong entirely to this component.
        std::vector<Edge> compEdges;
        for (const auto &edge : edges) {
            if (newId.count(edge.u) && newId.count(edge.v)) {
                compEdges.push_back({newId[edge.u], newId[edge.v], edge.weight});
            }
        }

        // Create an output filename (e.g., "graph_comp1.dat").
        std::ostringstream oss;
        oss << "graph_comp" << compNum << ".dat";
        std::string outputFile = oss.str();
        std::ofstream fout(outputFile);
        if (!fout) {
            std::cerr << "Error: Could not open output file " << outputFile << "\n";
            continue;
        }

        // Write the graph in the desired AMPL .dat format.
        fout << "# DGP1 instance in AMPL .dat format\n";
        fout << "# " << outputFile << "\n";
        fout << "param n := " << oldVertices.size() << " ;\n";
        fout << "param : E : c I :=\n";
        for (const auto &edge : compEdges) {
            fout << "  " << edge.u << " " << edge.v << " " 
                 << std::fixed << std::setprecision(3) << edge.weight << " 1\n";
        }
        fout << ";\n";
        fout.close();

        std::cout << "Written component " << compNum << " to file " << outputFile << "\n";
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>\n";
        return 1;
    }
    std::string inputFilename = argv[1];
    splitutil(inputFilename);
    return 0;
}