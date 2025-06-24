CXX = g++
CXXFLAGS = -Iinclude -Wall -Wextra -std=c++20 -O3 -MMD -MP -fopenmp -Wno-reorder -march=native

# Folders
SRC_DIR = src
BUILD_DIR = build

# Executable names
SOLVER = solver
SPLITUTIL = splitutil

# Source files
SOLVER_SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/solver.cpp $(SRC_DIR)/dsu.cpp $(SRC_DIR)/config.cpp
SPLITUTIL_SRCS = $(SRC_DIR)/splitutil.cpp $(SRC_DIR)/dsu.cpp

# Object files (in build/)
SOLVER_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOLVER_SRCS))
SPLITUTIL_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SPLITUTIL_SRCS))

# Dependency files
DEPS = $(SOLVER_OBJS:.o=.d) $(SPLITUTIL_OBJS:.o=.d)

# Default target
all: $(SOLVER) $(SPLITUTIL)

# Build the solver executable
$(SOLVER): $(SOLVER_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build the splitutil executable
$(SPLITUTIL): $(SPLITUTIL_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each .cpp into build/*.o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Include dependency files
-include $(DEPS)

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(SOLVER) $(SPLITUTIL) *.d *.o

.PHONY: all clean
