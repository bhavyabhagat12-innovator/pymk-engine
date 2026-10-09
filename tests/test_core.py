from core.graph import Graph
from core.mutual import mutual_friends
from core.bfs import shortest_distance, friends_of_friends
from core.heuristics import common_neighbors, jaccard, adamic_adar
from core.recommender import recommend


def build():
    # 1-2, 1-3, 2-3, 2-4, 3-4, 4-5, 6 isolated pair 6-7
    g = Graph()
    for u, v in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (6, 7)]:
        g.add_edge(u, v)
    return g


def test_mutual():
    g = build()
    assert mutual_friends(g, 1, 4) == {2, 3}
    assert mutual_friends(g, 1, 5) == set()


def test_distance():
    g = build()
    assert shortest_distance(g, 1, 1) == 0
    assert shortest_distance(g, 1, 2) == 1
    assert shortest_distance(g, 1, 5) == 3
    assert shortest_distance(g, 1, 6) == -1


def test_friends_of_friends():
    g = build()
    assert friends_of_friends(g, 1) == {4}


def test_heuristics():
    g = build()
    assert common_neighbors(g, 1, 4) == 2
    assert jaccard(g, 1, 4) == 2 / 3   
    assert adamic_adar(g, 1, 4) > 0


def test_recommend():
    g = build()
    result = recommend(g, 1, k=3)
    assert result[0]["user"] == 4
    assert result[0]["mutual_friends"] == [2, 3]
