import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
from core.graph import Graph
from core.bfs import shortest_distance
from core.heuristics import HEURISTICS
from core.recommender import recommend

st.set_page_config(page_title="People You May Know", layout="wide")


@st.cache_resource
def load_graph():
    g = Graph()
    g.load("data/facebook_combined.txt")
    return g


g = load_graph()

st.title("People You May Know Engine")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Recommendations", "Mutual Friends", "Network Stats", "Ego Network", "Performance"]
)

with tab1:
    c1, c2, c3 = st.columns(3)
    user = c1.number_input("User ID", min_value=0, value=0, step=1)
    method = c2.selectbox("Heuristic", list(HEURISTICS))
    k = c3.slider("Top K", 1, 30, 10)

    if user not in g.adj:
        st.error("User not found")
    else:
        recs = recommend(g, user, k, method)
        df = pd.DataFrame(recs)
        df["mutual_count"] = df["mutual_friends"].apply(len)
        st.dataframe(df, use_container_width=True)

with tab2:
    a = st.number_input("User A", min_value=0, value=0, step=1, key="a")
    b = st.number_input("User B", min_value=0, value=1, step=1, key="b")
    if a in g.adj and b in g.adj:
        m = sorted(g.adj[a] & g.adj[b])
        st.metric("Mutual friends", len(m))
        st.metric("Degrees of separation", shortest_distance(g, a, b))
        st.write(m)
    else:
        st.error("User not found")

with tab3:
    n, e = g.num_nodes(), g.num_edges()
    s1, s2, s3 = st.columns(3)
    s1.metric("Users", n)
    s2.metric("Friendships", e)
    s3.metric("Avg degree", round(2 * e / n, 2))
    degrees = [g.degree(x) for x in g.adj]
    st.plotly_chart(px.histogram(degrees, nbins=50, title="Degree distribution"),
                    use_container_width=True)

with tab4:
    import networkx as nx
    import plotly.graph_objects as go

    c1, c2 = st.columns(2)
    center = c1.number_input("User", min_value=0, value=0, step=1, key="ego_user")
    method4 = c2.selectbox("Heuristic", list(HEURISTICS), key="ego_method")

    if center not in g.adj:
        st.error("User not found")
    else:
        top = recommend(g, center, 1, method4)
        target = top[0]["user"] if top else None
        shared = set(top[0]["mutual_friends"]) if top else set()

        nodes = {center} | g.adj[center]
        if target is not None:
            nodes.add(target)
        G = nx.Graph()
        for a in nodes:
            for b in g.adj[a] & nodes:
                G.add_edge(a, b)

        pos = nx.spring_layout(G, seed=42)

        ex, ey = [], []
        for a, b in G.edges():
            ex += [pos[a][0], pos[b][0], None]
            ey += [pos[a][1], pos[b][1], None]

        def color(n):
            if n == center: return "red"
            if n == target: return "gold"
            if n in shared: return "limegreen"
            return "lightgray"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(width=0.3, color="gray"), hoverinfo="none"))
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode="markers",
            marker=dict(size=[16 if n in (center, target) else 8 for n in G.nodes()],
                        color=[color(n) for n in G.nodes()]),
            text=[f"User {n}" for n in G.nodes()], hoverinfo="text"))
        fig.update_layout(showlegend=False, height=650,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red: selected user. Gold: top recommendation. "
                   "Green: mutual friends. Gray: other friends.")
with tab5:
    st.subheader("Graph Performance")

    # Load
    l1, l2, l3 = st.columns(3)
    l1.metric("Load time", "0.484 s")
    l2.metric("Peak memory", "15.3 MB")
    l3.metric("Graph size", "4,039 nodes / 88,234 edges")

    # Query times
    st.markdown("#### Query times (average per query)")
    q = pd.DataFrame({
        "Operation": ["Mutual friends", "Friends-of-friends (BFS depth 2)", "BFS distance (full)"],
        "Time (ms)": [0.0029, 0.3358, 4.1052],
    })
    st.plotly_chart(
        px.bar(q, x="Operation", y="Time (ms)", text="Time (ms)", log_y=True),
        use_container_width=True,
    )
    st.caption("Log scale: set intersection is orders of magnitude faster than full BFS.")

    # Heuristic speed
    st.markdown("#### Heuristic speed (top-10 recommendation)")
    sp = pd.DataFrame({
        "Heuristic": ["preferential_attachment", "common_neighbors", "adamic_adar", "jaccard"],
        "Time (ms)": [0.664, 1.365, 2.162, 5.162],
    })
    st.plotly_chart(px.bar(sp, x="Heuristic", y="Time (ms)", text="Time (ms)"),
                    use_container_width=True)

    # Heap vs sort
    st.markdown("#### Heap Top-K vs full sort (about 690 candidates, K=10)")
    hs = pd.DataFrame({"Method": ["Heap Top-K", "Full sort"], "Time (ms)": [0.0727, 0.1422]})
    st.plotly_chart(px.bar(hs, x="Method", y="Time (ms)", text="Time (ms)"),
                    use_container_width=True)

    # Quality
    st.markdown("#### Recommendation quality (hide 10% of edges, K=10)")
    ql = pd.DataFrame({
        "Heuristic": ["adamic_adar", "common_neighbors", "jaccard", "preferential_attachment"],
        "Precision@10": [0.2660, 0.2620, 0.2490, 0.0897],
        "Recall@10": [0.5639, 0.5462, 0.4828, 0.1311],
    })
    qm = ql.melt(id_vars="Heuristic", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(qm, x="Heuristic", y="Score", color="Metric", barmode="group"),
                    use_container_width=True)
    st.success("Adamic-Adar gives the best quality; preferential attachment is far worse "
               "because it ignores mutual friends.")