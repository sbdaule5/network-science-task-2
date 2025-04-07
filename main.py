import sys
import time
from itertools import chain
from multiprocessing import Pool, cpu_count

import igraph as ig


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


def compute_max_stress_centrality(G):
    vcount = G.vcount()
    stress = [0] * vcount

    for i in range(vcount):
        paths = G.get_all_shortest_paths(
            i, to=range(i + 1, vcount), mode="ALL"
        )

        # Process all intermediate nodes in one sweep
        for node in chain.from_iterable(path[1:-1] for path in paths if len(path) > 2):
            stress[node] += 1

    return max(stress)


def compute_centralities(args):
    graph, u, v = args
    myG = graph.copy()
    myG.add_edge(u, v)
    betw = myG.betweenness()
    max_betw = max(betw)
    max_stress = compute_max_stress_centrality(myG)
    return max_betw, max_stress, (u, v)


def do_pool(filename):
    G = read_adjlist(filename)
    print(f"Original maximum betweenness centrality is {max(G.betweenness())}")
    print(f"Original maximum stress centrality is {compute_max_stress_centrality(G)}")
    start_time = time.time()
    # Calculate betweenness centrality
    missing_edges = [
        (i, j)
        for i in range(G.vcount())
        for j in range(i + 1, G.vcount())
        if not G.are_adjacent(i, j)
    ]
    args = [(G, u, v) for u, v in missing_edges]
    print(len(args))

    num_processes = min(cpu_count(), len(missing_edges))
    with Pool(processes=num_processes) as pool:
        start_time = time.time()
        # results = pool.map(compute_centralities, args)
        results = [compute_centralities(arg) for arg in args[:1000]]
        best_betw = min(results, key=lambda k: k[0])
        best_stress = min(results, key=lambda k: k[1])
        print(f"Optimization for betweenness centrality => {best_betw}")
        print(f"Optimization for stress centrality => {best_stress}")

    end_time = time.time()
    print(f"Running time => {end_time - start_time}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python script.py <arg>")
        sys.exit(1)

    arg = sys.argv[1]
    do_pool(arg)
