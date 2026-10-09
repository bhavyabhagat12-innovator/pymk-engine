def mutual_friends(graph, u, v):
    return graph.adj[u] & graph.adj[v]
