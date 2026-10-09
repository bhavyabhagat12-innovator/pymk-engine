# People You May Know Engine

A friend recommendation system built on the Stanford ego-Facebook graph
(4,039 users, 88,234 friendships). It computes mutual friends, degrees of
separation, and ranked "People You May Know" suggestions.

## DSA Concepts Used
- Adjacency list (dict of sets)
- BFS (shortest distance, friends-of-friends)
- Min-heap Top-K selection
- Set operations (mutual friends)
- Link-prediction heuristics: Common Neighbors, Jaccard, Adamic-Adar,
  Preferential Attachment

## How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py          # dashboard
uvicorn api.main:app --reload           # API (docs at /docs)
python -m pytest                        # tests
python benchmarks/bench.py              # benchmarks
```

## Benchmark Results
== Load ==
Load time: 0.484 s | Peak memory: 15.3 MB
Nodes: 4039 | Edges: 88234

== Query times (average per query) ==
Mutual friends: 0.0029 ms
BFS distance:   4.1052 ms
Friends-of-friends: 0.3358 ms

== Recommend top-10 by heuristic ==
common_neighbors          1.365 ms
jaccard                   5.162 ms
adamic_adar               2.162 ms
preferential_attachment   0.664 ms

== Heap vs full sort (50 highest-degree users, K=10) ==
Heap top-K: 0.0727 ms | Full sort: 0.1422 ms
Avg candidates per user: 690

== Quality evaluation (hide 10% of edges) ==
common_neighbors          precision@10: 0.2620 | recall@10: 0.5462
jaccard                   precision@10: 0.2490 | recall@10: 0.4828
adamic_adar               precision@10: 0.2660 | recall@10: 0.5639
preferential_attachment   precision@10: 0.0897 | recall@10: 0.1311

## Dataset
Stanford SNAP ego-Facebook (J. McAuley and J. Leskovec, "Learning to
Discover Social Circles in Ego Networks", NIPS 2012).
https://snap.stanford.edu/data/ego-Facebook.html
