"""Generate print-quality figures for REPORT.pdf (white background).

Outputs into report/:  fig_elbow.png, fig_pca_scatter.png, fig_svd.png,
fig_importance.png, fig_network.png — all from the real trained artifacts.
"""

import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import features as F          # noqa: E402
import recommender as R       # noqa: E402

OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
SEED = 42
RED, GREEN, BLUE, GOLD = "#d1495b", "#2e8b57", "#4878a8", "#e3a008"
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

# ----------------------------------------------------------- data + artifacts
hist = pd.read_csv(ROOT / "data/raw/author_papers.csv", parse_dates=["published"])
hist = hist.rename(columns={"queried_author": "author"})
hist = F.add_author_position(hist)
df = F.build_author_features(hist)
y = df["churned"].to_numpy()
model_bundle = joblib.load(ROOT / "app/model.pkl")
rec_bundle = joblib.load(ROOT / "app/recommender.pkl")
selected = model_bundle["features"]
churn_probs = rec_bundle["churn_probs"]

X = df[selected]
Xs = StandardScaler().fit_transform(X)

# --------------------------------------------------------------- 1) PCA elbow
pca_full = PCA(random_state=SEED).fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(range(1, len(cum) + 1), cum, "o-", color=BLUE, lw=2)
ax.axhline(0.9, ls="--", c="grey", lw=1, label="90% variance")
ax.set_xlabel("number of components"); ax.set_ylabel("cumulative explained variance")
ax.set_title("PCA elbow — cumulative explained variance"); ax.legend()
ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(OUT / "fig_elbow.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------ 2) PCA scatter
coords = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
ev = pca_full.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(coords[y == 0, 0], coords[y == 0, 1], s=26, c=GREEN, alpha=0.6, label="active")
ax.scatter(coords[y == 1, 0], coords[y == 1, 1], s=26, c=RED, alpha=0.7, label="churned")
ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% var)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% var)")
ax.set_title("Researchers in PCA space, colored by churn"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig_pca_scatter.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------- 3) SVD
matrix, m_authors, m_cats = R.build_interaction_matrix(hist, F.CUTOFF)
svd = R.fit_svd(matrix)
share = svd["sigma"] ** 2 / (svd["sigma"] ** 2).sum()
A = np.asarray(rec_bundle["author_factors"], float)
A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
authmap = {a: i for i, a in enumerate(rec_bundle["authors"])}
churn_a = np.array([int(df.set_index("author")["churned"].get(a, 0))
                    for a in rec_bundle["authors"]])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(1, len(share) + 1), share, color=BLUE)
axes[0].set_xlabel("latent factor"); axes[0].set_ylabel("variance share")
axes[0].set_title(f"SVD scree — {rec_bundle['explained_variance']*100:.0f}% in {len(share)} factors")
axes[1].scatter(A[churn_a == 0, 0], A[churn_a == 0, 1], s=18, c=GREEN, alpha=0.5, label="active author")
axes[1].scatter(A[churn_a == 1, 0], A[churn_a == 1, 1], s=18, c=RED, alpha=0.6, label="churned author")
cat = rec_bundle["category_factors"]; cats = rec_bundle["categories"]
for rec in rec_bundle["candidates"].itertuples(index=False):
    idx = [cats.index(c) for c in str(rec.categories).split("|") if c in cats]
    if idx:
        v = cat[:, idx].mean(axis=1); v = v / (np.linalg.norm(v) + 1e-9)
        axes[1].scatter(v[0], v[1], s=22, marker="D", c=GOLD, alpha=0.55)
axes[1].scatter([], [], marker="D", c=GOLD, label="candidate paper")
axes[1].set_xlabel("latent factor 1"); axes[1].set_ylabel("latent factor 2")
axes[1].set_title("Recommender latent space"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_svd.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------- 4) feature importance
imp = pd.Series(model_bundle["model"].feature_importances_, index=selected).sort_values()
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.barh(imp.index, imp.values, color=BLUE)
for i, v in enumerate(imp.values):
    ax.text(v + 0.005, i, f"{v*100:.1f}%", va="center", fontsize=9)
ax.set_xlabel("RandomForest importance (share of total)")
ax.set_title("Feature importance"); ax.set_xlim(0, imp.max() * 1.18)
fig.tight_layout(); fig.savefig(OUT / "fig_importance.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------- 5) co-authorship
pre = hist[hist.published <= F.CUTOFF]
nodes = list(df["author"]); in_set = set(nodes)
G = nx.Graph(); G.add_nodes_from(nodes)
for _, grp in pre.groupby("arxiv_id"):
    names = sorted(set(grp["author"]) & in_set)
    G.add_edges_from(combinations(names, 2))
pos = nx.spring_layout(G, seed=SEED, k=0.35)
deg = dict(G.degree())
colors = [RED if c else GREEN for c in df.set_index("author").loc[nodes, "churned"]]
sizes = [20 + 120 * deg[n] / max(deg.values()) for n in nodes]
fig, ax = plt.subplots(figsize=(8, 6.4))
nx.draw_networkx_edges(G, pos, alpha=0.18, ax=ax, edge_color="#888")
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85,
                       linewidths=0.3, edgecolors="white", ax=ax)
ax.scatter([], [], c=GREEN, label="active"); ax.scatter([], [], c=RED, label="churned")
ax.legend(loc="upper right"); ax.axis("off")
ax.set_title(f"Co-authorship network — {G.number_of_nodes()} researchers, "
             f"{G.number_of_edges()} edges (size = degree)")
fig.tight_layout(); fig.savefig(OUT / "fig_network.png", bbox_inches="tight"); plt.close(fig)

print("figures written to", OUT)
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name, f"{f.stat().st_size/1024:.0f} KB")
