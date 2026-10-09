from fastapi import FastAPI, HTTPException
from core.graph import Graph
from core.mutual import mutual_friends
from core.bfs import shortest_distance
from core.heuristics import HEURISTICS
from core.recommender import recommend

app = FastAPI(title="People You May Know Engine")

graph = Graph()
graph.load("data/facebook_combined.txt")  # loaded once at startup


def check_user(u):
    if u not in graph.adj:
        raise HTTPException(status_code=404, detail=f"User {u} not found")


@app.get("/stats")
def stats():
    n, e = graph.num_nodes(), graph.num_edges()
    hubs = sorted(graph.adj, key=graph.degree, reverse=True)[:10]
    return {
        "nodes": n,
        "edges": e,
        "avg_degree": round(2 * e / n, 2),
        "density": round(2 * e / (n * (n - 1)), 5),
        "top_hubs": [{"user": h, "degree": graph.degree(h)} for h in hubs],
    }


@app.get("/mutual/{u}/{v}")
def mutual(u: int, v: int):
    check_user(u); check_user(v)
    m = sorted(mutual_friends(graph, u, v))
    return {"count": len(m), "mutual_friends": m}


@app.get("/distance/{u}/{v}")
def distance(u: int, v: int):
    check_user(u); check_user(v)
    return {"distance": shortest_distance(graph, u, v)}


@app.get("/recommend/{u}")
def recommend_api(u: int, k: int = 10, method: str = "adamic_adar"):
    check_user(u)
    if method not in HEURISTICS:
        raise HTTPException(400, f"method must be one of {list(HEURISTICS)}")
    return recommend(graph, u, k, method)
