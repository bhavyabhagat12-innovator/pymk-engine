import heapq
from core.bfs import friends_of_friends
from core.heuristics import HEURISTICS


def recommend(graph, u, k=10, method="adamic_adar"):
    score_fn = HEURISTICS[method]
    heap = []  # min-heap of (score, node), size at most k

    for cand in friends_of_friends(graph, u):
        score = score_fn(graph, u, cand)
        if len(heap) < k:
            heapq.heappush(heap, (score, cand))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, cand))

    # Sort descending by score
    top = sorted(heap, reverse=True)
    return [
        {
            "user": node,
            "score": round(score, 4),
            "mutual_friends": sorted(graph.adj[u] & graph.adj[node]),
        }
        for score, node in top
    ]
