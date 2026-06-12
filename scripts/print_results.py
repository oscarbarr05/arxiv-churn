"""Print the three-setup comparison + dataset numbers (fills REPORT.md)."""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import features as F  # noqa: E402
import model as M     # noqa: E402

corpus = pd.read_csv(ROOT / "data/raw/papers.csv")
hist = pd.read_csv(ROOT / "data/raw/author_papers.csv",
                   parse_dates=["published"])
hist = hist.rename(columns={"queried_author": "author"})
hist = F.add_author_position(hist)
df = F.build_author_features(hist)
y = df["churned"]
X = df[F.FEATURE_COLUMNS]
print(f"N_PAPERS={len(corpus)} N_ROWS={len(hist)} N_AUTHORS={len(df)} "
      f"CHURN_RATE={y.mean():.1%}")

pre = hist[hist.published <= F.CUTOFF]
in_set = set(df.author)
G = nx.Graph()
G.add_nodes_from(in_set)
for _, grp in pre.groupby("arxiv_id"):
    names = sorted(set(grp.author) & in_set)
    G.add_edges_from(combinations(names, 2))
deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, seed=42)
pr = nx.pagerank(G)
df["degree_centrality"] = df.author.map(deg).fillna(0)
df["betweenness"] = df.author.map(btw).fillna(0)
df["pagerank"] = df.author.map(pr).fillna(0)
print(f"GRAPH: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]
rf = lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=42)

def run(est, Xc, name):
    s = cross_validate(est, Xc, y, cv=cv, scoring=scoring)
    vals = {m: np.mean(s[f"test_{m}"]) for m in scoring}
    print(name, {k: round(v, 3) for k, v in vals.items()})

selected = M.consensus_selection(X, y)
X_ext = df[F.FEATURE_COLUMNS + F.NETWORK_FEATURE_COLUMNS]
selected_ext = M.consensus_selection(X_ext, y)
print("SELECTED:", selected)
print("SELECTED_EXT:", selected_ext)

run(rf(), df[selected], "A")
run(Pipeline([("s", StandardScaler()), ("p", PCA(3, random_state=42)),
              ("rf", rf())]), X, "B")
run(rf(), df[selected + F.NETWORK_FEATURE_COLUMNS], "C")
