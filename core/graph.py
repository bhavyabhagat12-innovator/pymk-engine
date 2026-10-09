from collections import defaultdict


class Graph:
    """Undirected graph stored as an adjacency list.

    adj maps each user ID to the set of that user's friends.
    """

    def __init__(self):
        self.adj = defaultdict(set)

    def add_edge(self, u, v):
        # Friendship is mutual, so add both directions
        self.adj[u].add(v)
        self.adj[v].add(u)

    def load(self, path):
        """Read an edge-list file where each line is: user1 user2"""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                u, v = map(int, line.split())
                self.add_edge(u, v)

    def neighbors(self, u):
        return self.adj[u]

    def degree(self, u):
        return len(self.adj[u])

    def num_nodes(self):
        return len(self.adj)

    def num_edges(self):
        # Each edge is stored twice (u->v and v->u), so divide by 2
        return sum(len(n) for n in self.adj.values()) // 2
