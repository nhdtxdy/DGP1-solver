#include "config.h"
#include <stdexcept>

// Initialize static member
Config* Config::instance = nullptr;

Config& Config::getInstance() {
    if (!instance) {
        instance = new Config();
    }
    return *instance;
}

std::string Config::trim(const std::string& s) {
    size_t first = s.find_first_not_of(" \t\n\r");
    if (std::string::npos == first) {
        return "";
    }
    size_t last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, (last - first + 1));
}

bool Config::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << ". Using default values." << std::endl;
        return false;
    }
    
    std::cout << "Loading configuration from " << filename << "..." << std::endl;

    std::string line;
    while (std::getline(file, line)) {
        // Remove comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim(line);
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            if (std::getline(iss, value)) {
                key = trim(key);
                value = trim(value);

                try {
                    // Parse values
                    if (key == "RANDOMIZE_ARRAY_SIZE") {
                        randomize.RANDOMIZE_ARRAY_SIZE = std::stoi(value);
                    } else if (key == "DEGREE_PROD") {
                        node_scoring.DEGREE_PROD = std::stod(value);
                    } else if (key == "CYCLE_COUNT_PROD") {
                        node_scoring.CYCLE_COUNT_PROD = std::stod(value);
                    } else if (key == "WEIGHT_SUM_PROD") {
                        node_scoring.WEIGHT_SUM_PROD = std::stod(value);
                    } else if (key == "CYCLE_PARTICIPATION_PROD") {
                        node_scoring.CYCLE_PARTICIPATION_PROD = std::stod(value);
                    } else if (key == "DEGREE_POW") {
                        node_scoring.DEGREE_POW = std::stod(value);
                    } else if (key == "CYCLE_COUNT_POW") {
                        node_scoring.CYCLE_COUNT_POW = std::stod(value);
                    } else if (key == "WEIGHT_SUM_POW") {
                        node_scoring.WEIGHT_SUM_POW = std::stod(value);
                    } else if (key == "CYCLE_PARTICIPATION_POW") {
                        node_scoring.CYCLE_PARTICIPATION_POW = std::stod(value);
                    } else if (key == "eps") {
                        eps = (WeightType)std::stod(value);
                    } else {
                        std::cerr << "Warning: Unknown config key: " << key << std::endl;
                    }
                } catch (const std::invalid_argument& ia) {
                    std::cerr << "Warning: Invalid value for key '" << key << "': " << value << std::endl;
                } catch (const std::out_of_range& oor) {
                    std::cerr << "Warning: Value out of range for key '" << key << "': " << value << std::endl;
                }
            }
        }
    }
    
    file.close();
    std::cout << "Configuration loaded." << std::endl;
    return true;
}

void Config::printConfig() const {
    std::cout << "--- Runtime Configuration ---" << std::endl;
    std::cout << "Randomize:" << std::endl;
    std::cout << "  RANDOMIZE_ARRAY_SIZE: " << randomize.RANDOMIZE_ARRAY_SIZE << std::endl;
    std::cout << "Node Scoring:" << std::endl;
    std::cout << "  DEGREE_PROD: " << node_scoring.DEGREE_PROD << std::endl;
    std::cout << "  CYCLE_COUNT_PROD: " << node_scoring.CYCLE_COUNT_PROD << std::endl;
    std::cout << "  WEIGHT_SUM_PROD: " << node_scoring.WEIGHT_SUM_PROD << std::endl;
    std::cout << "  CYCLE_PARTICIPATION_PROD: " << node_scoring.CYCLE_PARTICIPATION_PROD << std::endl;
    std::cout << "  DEGREE_POW: " << node_scoring.DEGREE_POW << std::endl;
    std::cout << "  CYCLE_COUNT_POW: " << node_scoring.CYCLE_COUNT_POW << std::endl;
    std::cout << "  WEIGHT_SUM_POW: " << node_scoring.WEIGHT_SUM_POW << std::endl;
    std::cout << "  CYCLE_PARTICIPATION_POW: " << node_scoring.CYCLE_PARTICIPATION_POW << std::endl;
    std::cout << "Global:" << std::endl;
    std::cout << "  eps: " << eps << std::endl;
    std::cout << "---------------------------" << std::endl;
} 