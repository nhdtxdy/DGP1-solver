import subprocess
import tempfile
import os
from gentest import generate_dgp1_instance, save_graph_to_dat, save_true_values
import random
import math
import sys
import shutil
import readline

def run_solver(graph_path):
    return subprocess.run(["./solver", graph_path, "/dev/null", "--optimizations=bridges,triangle"]).returncode

def test_solver_on_many_instances(num_tests=5, keep_failed=True):
    failed_tests = 0

    for test_id in range(num_tests):
        n = random.randint(7, 7)
        m = random.randint(math.ceil(n * 1.4), math.floor(n * 1.7))
        M = 10
        n, edges, edge_weights, vertex_positions = generate_dgp1_instance(n, m, M)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False, dir='/tmp') as tmp_graph:
            save_graph_to_dat(n, edges, edge_weights, tmp_graph.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir='/tmp') as tmp_truth:
            save_true_values(vertex_positions, tmp_truth.name)

        ret = run_solver(tmp_graph.name)

        if ret != 0:
            print(f"[FAIL] Solver failed on test n={n} m={m}")
            failed_tests += 1
            if keep_failed:
                # Move files to current dir for inspection
                new_graph = f"failed_graph_n{n}_m{m}_t{test_id}.dat"
                new_truth = f"true_values_n{n}_m{m}_t{test_id}.txt"
                shutil.copyfile(tmp_graph.name, new_graph)
                shutil.copyfile(tmp_truth.name, new_truth)
            else:
                os.remove(tmp_graph.name)
                os.remove(tmp_truth.name)
        else:
            print(f"[OK] Solver passed on test n={n} m={m}")
            os.remove(tmp_graph.name)
            os.remove(tmp_truth.name)


    print(f"\nTest summary: {num_tests - failed_tests}/{num_tests} passed.")

if __name__ == "__main__":
    ntpn = int(input('Input # of tests: '))
    test_solver_on_many_instances(ntpn)