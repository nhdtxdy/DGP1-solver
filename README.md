# DGP1 Solver

A high-performance C++20 implementation for the one–dimensional **Distance Geometry Problem (DGP1)**.  
Given an undirected weighted graph \(G=(V,E,w)\) the task is to assign an integer (or real) coordinate \(P[v]\) to every vertex so that

```
|P[u] - P[v]| = w(u,v) ,   for every edge (u,v)∈E
```
The program either prints **one** valid assignment or – with a flag – **all** assignments that satisfy the constraints. If printing all assignments is chosen, it ALWAYS RUNS IN O(2^N).

---

## 1. Building the project

The repository contains a portable `Makefile` that produces two executables:  
• `solver`      – the main DGP1 solver (built from `src/main.cpp`).  
• `splitutil`   – a helper tool that splits a disconnected instance into connected components.

To build everything from scratch run

```bash
make clean   # optional – removes previous objects/binaries
make         # produces ./solver and ./splitutil
```

The build uses

* C++20
* `-O3`, `-fopenmp` (multi-threaded Dijkstra, bounds optimization, etc.)
* `-Iinclude` so you do **not** need to install additional libraries.

### Integer vs. floating-point weights
By default the macro `USE_INTEGER_WEIGHTS` is defined in `include/config.h` which makes the solver work with **signed 32-bit integers**.  
Comment out this line if you need floating-point edge weights (double precision).  Remember to do a full rebuild afterwards.

---

## 2. Command-line syntax

```
./solver <input_file> [output_file]
        [--config=<config.ini>]
        [--root=<strategy>] [--neighbors=<strategy>]
        [--optimizations=opt1,opt2,...]
        [--list-all-solutions]
        [--export-mzn=<file.mzn>]   # MiniZinc
        [--export-gurobi=<file.lp>] # Gurobi LP
```

Argument summary

* `<input_file>`   – AMPL `.dat` graph (see §4).
* `[output_file]`   – optional destination; `stdout` is used if omitted.
* `--config`        – optional INI-style file that overrides runtime parameters (see §5).
* `--root`, `--neighbors` choose the ordering heuristics when exploring the DFS tree.
* `--optimizations` activates one or more pruning / acceleration techniques (comma separated).
* `--list-all-solutions` enumerates every feasible assignment instead of stopping at the first.
* `--export-mzn / --export-gurobi` dump the instance as a MiniZinc or Gurobi model and **exit** (no solving).

### 2.1 Node-ordering heuristics
Available values for `root` **and** `neighbors` (default is `default`):

* `highest-order`   – vertex with the largest degree first.
* `highest-cycle`   – vertex contained in the largest number of cycles first.
* `highest-score`   – hybrid scoring function (degree, cycles, weight sum, etc.).

### 2.2 Optimisation flags
The code currently recognises the following keywords:

| Flag               | Description |
|--------------------|-------------|
| `bridges`          | Detects graph bridges, solves each 2-edge-connected component independently, then merges by translation. |
| `randomize`        | 50 % chance of flipping the sign of every tree-edge during DFS, giving a form of **Las-Vegas** exploration that can shorten search paths. |
| `ssp`              | Enables the **Subset-Sum Pruning (SSP)** technique based on on-the-fly bit-set knapsacks. Works only with integer weights. |
| `ssp-preprocess`   | Performs an *O(n²)* preprocessing pass that memoises SSP results between all root/ancestor pairs. |
| `triangle`         | Uses the triangle inequality to discard infeasible cycles early. |
| `bounds`           | Maintains running lower/upper bounds for every vertex at every DFS depth (propagated by shortest-path distances). |

You can mix any subset, e.g.:

```bash
./solver graph.dat --optimizations=bridges,ssp,randomize,bounds
```

---

## 3. Examples

1. Solve and print **all** solutions, choosing the highest degree vertex as root:

```bash
./solver tests/graph_10_15_100.dat result.txt \
        --root=highest-order --list-all-solutions
```

2. Heavily tuned run with multiple optimisations and custom neighbour order:

```bash
./solver tests/graph_400_600_1000.dat \
        --root=highest-cycle --neighbors=highest-cycle \
        --optimizations=bridges,randomize,triangle,ssp
```

3. Produce a MiniZinc model of an instance (no solving):

```bash
./solver graph.dat --export-mzn=graph.mzn
```

---

## 4. Input format
The solver expects the **AMPL `.dat`** format used in the DIMACS DGP library:

```
param n := <num_vertices> ;
param : E : c I :=
  u1 v1  weight1 1
  u2 v2  weight2 1
  ...
;
```

*Vertices are numbered from **1** to **n**.*  Lines starting with `#` are treated as comments and ignored.

You can generate random connected instances with `gentest.py`:

```bash
python gentest.py
```
The script also stores the ground-truth positions for verification.

If your graph is disconnected, run

```bash
./splitutil graph.dat
```
which creates separate `graph_comp<i>.dat` files – one for each component.

---

## 5. Runtime configuration file
`--config=<file>` allows fine-tuning without recompilation.  The file is parsed as `key=value` pairs, e.g.

```
# Sample config.ini
RANDOMIZE_ARRAY_SIZE = 10000000
DEGREE_PROD            = 1.0
CYCLE_COUNT_PROD       = 2.0
CYCLE_PARTICIPATION_POW = 1.5
```

See `include/config.h` for the full list of adjustable parameters.  Values not specified keep their defaults.

---

## 6. Testing
A collection of benchmark instances is provided in the `tests/` directory together with

* `test_accuracy.py` – quickly checks that the solver's output satisfies every edge constraint.
* `visualize.py`     – small utility to plot vertex coordinates on a line.



