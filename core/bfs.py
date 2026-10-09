from collections import deque


def shortest_distance(graph, src, dst):
    """Degrees of separation between two users.
    Returns -1 if they are not connected."""
    if src == dst:
        return 0
    visited = {src}
    queue = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        for nbr in graph.adj[node]:
            if nbr == dst:
                return dist + 1
            if nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, dist + 1))
    return -1


def friends_of_friends(graph, u):
    """Users at distance exactly 2 from u (BFS limited to depth 2).
    These are the candidates for 'People You May Know'."""
    direct = graph.adj[u]
    visited = {u} | direct
    candidates = set()
    for friend in direct:
        for fof in graph.adj[friend]:
            if fof not in visited:
                visited.add(fof)
                candidates.add(fof)
    return candidates
