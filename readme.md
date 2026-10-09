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
(paste the contents of benchmarks/results.txt here)

## Dataset
Stanford SNAP ego-Facebook (J. McAuley and J. Leskovec, "Learning to
Discover Social Circles in Ego Networks", NIPS 2012).
https://snap.stanford.edu/data/ego-Facebook.html
