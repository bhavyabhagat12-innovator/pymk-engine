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
tab1, tab2, tab3 = st.tabs(["Recommendations", "Mutual Friends", "Network Stats"])

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
