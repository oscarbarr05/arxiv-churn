"""Precompute visualization data -> app/viz.json (served by GET /viz).

Four meaningful views, all from the real trained artifacts:
  * PCA   : 295 researchers in 2D, colored by churn (+ params so the dashboard
            can project a live slider profile into the same space).
  * SVD   : researchers and candidate papers in the recommender's latent space.
  * importance : RandomForest feature importances (the 8 served features).
  * network    : co-authorship graph (spring layout, churn-colored).

This runs author-side (like build_model.py); the container only serves the
resulting viz.json. Run:  python scripts/make_viz.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import features as F          # noqa: E402

OUT = ROOT / "app" / "viz.json"
SEED = 42


def norm01(a: np.ndarray) -> np.ndarray:
    """Scale an array to [0,1] per column (for SVG plotting)."""
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(axis=0), a.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (a - lo) / rng


# ----------------------------------------------------------- load everything
corpus = pd.read_csv(ROOT / "data/raw/papers.csv")
hist = pd.read_csv(ROOT / "data/raw/author_papers.csv", parse_dates=["published"])
hist = hist.rename(columns={"queried_author": "author"})
hist = F.add_author_position(hist)
df = F.build_author_features(hist)

model_bundle = joblib.load(ROOT / "app/model.pkl")
rec_bundle = joblib.load(ROOT / "app/recommender.pkl")
selected = model_bundle["features"]                 # the 8 served features
churn_probs = rec_bundle["churn_probs"]             # Series indexed by author

# ------------------------------------------------------------------- 1) PCA
X = df[selected]
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
pca = PCA(n_components=3, random_state=SEED).fit(Xs)   # 3 comps: 2D + 3D views
coords = pca.transform(Xs)
probs = churn_probs.reindex(df["author"]).fillna(0.0).to_numpy()
pca_view = {
    "points": [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]),
         "z": float(coords[i, 2]),
         "churn": int(df["churned"].iloc[i]), "prob": round(float(probs[i]), 3),
         "author": df["author"].iloc[i]}
        for i in range(len(df))
    ],
    "feature_order": list(selected),
    "scaler_mean": [float(v) for v in scaler.mean_],
    "scaler_scale": [float(v) for v in scaler.scale_],
    "pca_mean": [float(v) for v in pca.mean_],
    "components": [[float(v) for v in row] for row in pca.components_],  # 3x8
    "explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
}

# ------------------------------------------------------------------- 2) SVD
# author and paper vectors live in the same k-dim latent space; recommend uses
# cosine similarity, so we L2-normalize then take the top-2 factors for a 2D
# projection whose angles approximate the real (k-dim) similarity.
A = np.asarray(rec_bundle["author_factors"], dtype=float)
A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
authors = rec_bundle["authors"]
churn_map = {a: int(df.set_index("author")["churned"].get(a, 0)) for a in authors}
svd_authors = [
    {"author": authors[i], "x": float(A[i, 0]), "y": float(A[i, 1]),
     "churn": churn_map[authors[i]],
     "prob": round(float(churn_probs.get(authors[i], 0.0)), 3)}
    for i in range(len(authors))
]

cat_factors = np.asarray(rec_bundle["category_factors"], dtype=float)  # k x n_cat
cats = rec_bundle["categories"]
cand = rec_bundle["candidates"]
svd_papers = {}
for rec in cand.itertuples(index=False):
    idx = [cats.index(c) for c in str(rec.categories).split("|") if c in cats]
    if not idx:
        continue
    v = cat_factors[:, idx].mean(axis=1)
    v = v / (np.linalg.norm(v) + 1e-9)
    svd_papers[rec.arxiv_id] = {
        "x": float(v[0]), "y": float(v[1]),
        "title": rec.title, "categories": rec.categories,
    }
svd_view = {"authors": svd_authors, "papers": svd_papers,
            "explained": round(float(rec_bundle["explained_variance"]), 3)}

# ------------------------------------------------------- 3) feature importance
imp = model_bundle["model"].feature_importances_
importance_view = sorted(
    [{"feature": f, "value": round(float(v), 4)} for f, v in zip(selected, imp)],
    key=lambda d: -d["value"],
)

# ------------------------------------------------------------ 4) co-authorship
pre = hist[hist.published <= F.CUTOFF]
node_list = list(df["author"])
idx_of = {a: i for i, a in enumerate(node_list)}
G = nx.Graph()
G.add_nodes_from(node_list)
for _, grp in pre.groupby("arxiv_id"):
    names = sorted(set(grp["author"]) & set(node_list))
    G.add_edges_from(combinations(names, 2))
pos = nx.spring_layout(G, seed=SEED, k=0.35)
deg = dict(G.degree())
P = norm01(np.array([pos[a] for a in node_list]))
network_view = {
    "nodes": [
        {"author": a, "x": float(P[i, 0]), "y": float(P[i, 1]),
         "churn": int(df["churned"].iloc[i]), "deg": int(deg.get(a, 0))}
        for i, a in enumerate(node_list)
    ],
    "edges": [[idx_of[u], idx_of[v]] for u, v in G.edges
              if u in idx_of and v in idx_of],
    "n_edges": G.number_of_edges(),
}

# -------------------------------------------------------------------- write
viz = {"pca": pca_view, "svd": svd_view,
       "importance": importance_view, "network": network_view}
OUT.write_text(json.dumps(viz), encoding="utf-8")
kb = OUT.stat().st_size / 1024
print(f"viz.json written ({kb:.0f} KB): "
      f"{len(pca_view['points'])} PCA pts, {len(svd_authors)} SVD authors, "
      f"{len(svd_papers)} papers, {network_view['n_edges']} edges")
print(f"PCA explains {sum(pca_view['explained']):.0%} in 2D; "
      f"SVD explains {svd_view['explained']:.0%}")
