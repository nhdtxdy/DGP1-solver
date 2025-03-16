CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O3

# Executable names
SOLVER = solver
SPLITUTIL = splitutil

# Source files for each target.
# The solver executable uses main.cpp (the entry point) along with the other components.
SOLVER_SRCS = main.cpp solver.cpp dsu.cpp
# The splitutil executable uses its own main in splitutil.cpp.
SPLITUTIL_SRCS = splitutil.cpp dsu.cpp

# Object files for each target.
SOLVER_OBJS = $(SOLVER_SRCS:.cpp=.o)
SPLITUTIL_OBJS = $(SPLITUTIL_SRCS:.cpp=.o)

# Default target builds both executables.
all: $(SOLVER) $(SPLITUTIL)

# Build the solver executable.
$(SOLVER): $(SOLVER_OBJS)
	$(CXX) $(CXXFLAGS) -o $(SOLVER) $(SOLVER_OBJS)

# Build the splitutil executable.
$(SPLITUTIL): $(SPLITUTIL_OBJS)
	$(CXX) $(CXXFLAGS) -o $(SPLITUTIL) $(SPLITUTIL_OBJS)

# Pattern rule to compile .cpp files to .o files.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files.
clean:
	rm -f $(SOLVER) $(SPLITUTIL) $(SOLVER_OBJS) $(SPLITUTIL_OBJS)

.PHONY: all clean