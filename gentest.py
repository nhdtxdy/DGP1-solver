import random
import itertools

def generate_dgp1_instance(n, m, M):
    assert m >= n - 1 and M >= max(10, n // 2), "Invalid input conditions"

    vertex_positions = {v: random.randint(-M, M) for v in range(n)}

    parent = list(range(n))
    rank = [0] * n

    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        else:
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
        return True

    edges = set()
    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(1, n):
        while True:
            u = nodes[i]
            v = random.choice(nodes[:i])
            if union(u, v):
                edges.add((min(u, v) + 1, max(u, v) + 1))
                break

    while len(edges) < m:
        u, v = random.sample(range(n), 2)
        if u == v:
            continue
        u, v = min(u, v) + 1, max(u, v) + 1
        if (u, v) not in edges:
            edges.add((u, v))

    edge_weights = {
        (u, v): abs(vertex_positions[u - 1] - vertex_positions[v - 1])
        for u, v in edges
    }

    return n, edges, edge_weights, vertex_positions

def save_graph_to_dat(n, edges, edge_weights, filename="graph.dat"):
    with open(filename, "w") as f:
        f.write("# DGP1 instance in AMPL .dat format\n")
        f.write("# graph.dat\n\n")
        f.write(f"param n := {n} ;\n\n")
        f.write("param : E : c I :=\n")
        for (u, v), weight in sorted(edge_weights.items()):
            f.write(f"  {u} {v}  {weight:.3f} 1\n")
        f.write(";\n")

def save_true_values(vertex_positions, filename="true_values.txt"):
    """Saves the true value (position) of each vertex to a file."""
    with open(filename, "w") as f:
        for v in sorted(vertex_positions.keys()):
            # Convert from 0-based to 1-based indexing for printing
            f.write(f"Vertex {v+1} -> {vertex_positions[v]}\n")

def gen_test(n, m, M, filename):
    n, edges, edge_weights, vertex_positions = generate_dgp1_instance(n, m, M)
    save_graph_to_dat(n, edges, edge_weights, filename)
    save_true_values(vertex_positions, "true_values.txt")
