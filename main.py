import time
from multiprocessing import Pool, cpu_count
import igraph as ig

from collections import Counter

def read_adjlist(filename: str):
    adjacency_list = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines or comments
            parts = list(map(int, line.split()))
            node = parts[0]
            neighbors = parts[1:]
            adjacency_list[node] = neighbors

    # Flatten adjacency list into edge tuples
    edges = []
    for node, neighbors in adjacency_list.items():
        edges.extend((node, neighbor) for neighbor in neighbors)

    # Create the graph
    G = ig.Graph(edges=edges, directed=False)
    return G


def compute_stress_centrality(G):
    stress = Counter()

    for source in range(G.vcount()):
        paths = G.get_shortest_paths(source, to=None, mode="ALL", output="vpath")
        for path in paths:
            # Exclude the source and target nodes themselves
            for node in path[1:-1]:
                stress[node] += 1

    return [stress[i] for i in range(G.vcount())]

def compute_betweenness(args):
    graph, u, v = args
    myG = graph.copy()
    myG.add_edge(u, v)
    betw = myG.betweenness()
    stress = compute_stress_centrality(myG)
    max_betw = max(betw)
    max_stress = max(stress)
    return max_betw, max_stress, (u,v)

def do_pool():
    G = read_adjlist("graph1.adjlist")
    start_time = time.time()
    # Calculate betweenness centrality
    missing_edges = [
        (i, j)
        for i in range(G.vcount())
        for j in range(i + 1, G.vcount())
        if not G.are_adjacent(i, j)
    ]
    args = [(G, u, v) for u, v in missing_edges]

    num_processes = min(cpu_count(), len(missing_edges))
    with Pool(processes=num_processes) as pool:
        start_time = time.time()
        results = pool.map(compute_betweenness, args)
        # total_time = sum(times)
        best_betw = min(results, key=lambda k: k[0])
        best_stress = min(results, key=lambda k: k[1])
        print(f"best betw => {best_betw}")
        print(f"best stress => {best_stress}")

    end_time = time.time()
    print(f"Running time => {end_time - start_time}")

if __name__ == "__main__":
    do_pool()
