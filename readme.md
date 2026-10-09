import sys, os, time, random, heapq, tracemalloc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import Graph
from core.mutual import mutual_friends
from core.bfs import shortest_distance, friends_of_friends
from core.heuristics import HEURISTICS
from core.recommender import recommend

random.seed(42)
DATA = "data/facebook_combined.txt"


def avg_ms(fn, items):
    start = time.perf_counter()
    for x in items:
        fn(x)
    return (time.perf_counter() - start) / len(items) * 1000


# 1. Load time and memory
tracemalloc.start()
t0 = time.perf_counter()
g = Graph()
g.load(DATA)
load_s = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print("== Load ==")
print(f"Load time: {load_s:.3f} s | Peak memory: {peak / 1024 / 1024:.1f} MB")
print(f"Nodes: {g.num_nodes()} | Edges: {g.num_edges()}")

users = list(g.adj)
sample = random.sample(users, 200)
pairs = [(random.choice(users), random.choice(users)) for _ in range(200)]

# 2. Query times
print("\n== Query times (average per query) ==")
print(f"Mutual friends: {avg_ms(lambda p: mutual_friends(g, *p), pairs):.4f} ms")
print(f"BFS distance:   {avg_ms(lambda p: shortest_distance(g, *p), pairs):.4f} ms")
print(f"Friends-of-friends: {avg_ms(lambda u: friends_of_friends(g, u), sample):.4f} ms")

# 3. Heuristic comparison (speed)
print("\n== Recommend top-10 by heuristic ==")
for name in HEURISTICS:
    t = avg_ms(lambda u: recommend(g, u, 10, name), sample)
    print(f"{name:25s} {t:.3f} ms")


# 4. Heap top-K vs full sort (selection step only)
def topk_heap(scores, k):
    heap = []
    for item in scores:
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)


def topk_sort(scores, k):
    return sorted(scores, reverse=True)[:k]


big_users = sorted(users, key=g.degree, reverse=True)[:50]
score_lists = []
for u in big_users:
    score_lists.append([(HEURISTICS["adamic_adar"](g, u, c), c)
                        for c in friends_of_friends(g, u)])

print("\n== Heap vs full sort (50 highest-degree users, K=10) ==")
t_heap = avg_ms(lambda s: topk_heap(s, 10), score_lists)
t_sort = avg_ms(lambda s: topk_sort(s, 10), score_lists)
print(f"Heap top-K: {t_heap:.4f} ms | Full sort: {t_sort:.4f} ms")
print(f"Avg candidates per user: {sum(map(len, score_lists)) / len(score_lists):.0f}")

# 5. Quality: hide 10% of edges, then try to recover them
print("\n== Quality evaluation (hide 10% of edges) ==")
edges = [(u, v) for u in g.adj for v in g.adj[u] if u < v]
random.shuffle(edges)
cut = len(edges) // 10
hidden, kept = edges[:cut], edges[cut:]

train = Graph()
for u, v in kept:
    train.add_edge(u, v)

hidden_of = {}
for u, v in hidden:
    hidden_of.setdefault(u, set()).add(v)
    hidden_of.setdefault(v, set()).add(u)

eval_users = [u for u in hidden_of if u in train.adj]
eval_users = random.sample(eval_users, min(300, len(eval_users)))

K = 10
for name in HEURISTICS:
    precisions, recalls = [], []
    for u in eval_users:
        recs = recommend(train, u, K, name)
        hits = sum(1 for r in recs if r["user"] in hidden_of[u])
        precisions.append(hits / K)
        recalls.append(hits / len(hidden_of[u]))
    print(f"{name:25s} precision@{K}: {sum(precisions)/len(precisions):.4f} "
          f"| recall@{K}: {sum(recalls)/len(recalls):.4f}")
