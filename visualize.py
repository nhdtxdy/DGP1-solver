import networkx as nx
import matplotlib.pyplot as plt

def read_dfs_data(filename="log.txt"):
    """Reads DFS tree and back edges from log file and converts to adjacency lists."""
    dfs_tree_adj = {}
    back_adj = {}
    
    with open(filename, "r") as file:
        lines = file.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line == "DFS_TREE_ADJ":
            current_section = dfs_tree_adj
            continue
        elif line == "BACK_EDGES":
            current_section = back_adj
            continue
        
        if current_section is not None:
            parts = line.split(":")
            node = int(parts[0])
            neighbors = list(map(int, parts[1].split())) if len(parts) > 1 and parts[1].strip() else []
            current_section[node] = neighbors

    return dfs_tree_adj, back_adj

def visualize_dfs_tree(dfs_tree_adj, back_adj):
    """Visualizes the DFS tree with back edges colored differently."""
    G = nx.DiGraph()

    # Add tree edges
    for node, neighbors in dfs_tree_adj.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor, color="black")

    # Add back edges
    for node, neighbors in back_adj.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor, color="red")

    # Define edge colors
    edge_colors = [G[u][v]["color"] for u, v in G.edges()]

    print('here')

    # Use a tree layout
    pos = nx.spring_layout(G, seed=42)

    print('ok')

    # Draw graph
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, edge_color=edge_colors, width=2, arrows=True)

    # Legend
    plt.text(0.1, 0.02, "Back Edges (Red)", color="red", transform=plt.gcf().transFigure)
    plt.text(0.1, 0.05, "Tree Edges (Black)", color="black", transform=plt.gcf().transFigure)

    plt.title("DFS Tree with Back Edges Highlighted")
    plt.show()

# Auto-read and visualize DFS Tree
dfs_tree_adj, back_adj = read_dfs_data()
visualize_dfs_tree(dfs_tree_adj, back_adj)