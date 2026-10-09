import math


def common_neighbors(graph, u, v):
    return len(graph.adj[u] & graph.adj[v])


def jaccard(graph, u, v):
    union = len(graph.adj[u] | graph.adj[v])
    if union == 0:
        return 0.0
    return len(graph.adj[u] & graph.adj[v]) / union


def adamic_adar(graph, u, v):
    score = 0.0
    for w in graph.adj[u] & graph.adj[v]:
        d = len(graph.adj[w])
        if d > 1:  # log(1) = 0 would cause division by zero
            score += 1 / math.log(d)
    return score


def preferential_attachment(graph, u, v):
    return len(graph.adj[u]) * len(graph.adj[v])


HEURISTICS = {
    "common_neighbors": common_neighbors,
    "jaccard": jaccard,
    "adamic_adar": adamic_adar,
    "preferential_attachment": preferential_attachment,
}
