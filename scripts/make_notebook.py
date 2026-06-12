"""Generate and execute notebooks/eda_and_selection.ipynb.

The notebook is the analytical audit trail: raw -> features -> 4 selection
methods -> PCA/SVD -> network -> model comparison. It imports app/features.py
and app/model.py (single source of truth) but is never imported by the app.
"""

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "eda_and_selection.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------- intro
md("""# arXiv Researcher Churn — EDA, Feature Selection, PCA/SVD & Network Analysis

**Final project — Introduction to Data Science (UPTP).**

**Entity:** a researcher publishing in `cs.CL` (computational linguistics) on arXiv.
**Churn:** the researcher submits **no paper after the cutoff date 2024-06-30** —
i.e. ~2 years of silence at evaluation time (June 2026).

### Why a temporal cutoff instead of a simple "days inactive" threshold?
A naive label like `days_inactive > 180` computed *today* leaks information:
recency would appear both inside the label and as a feature. We instead split
time itself: **features only see papers up to the cutoff, the label only looks
after it** — exactly how production churn systems at Netflix/Spotify frame the
problem ("given behaviour up to today, who goes silent over the next window?").

### Why 2 years?
Academic publishing is slower than music or e-commerce: the median gap between
consecutive papers for active cs.CL authors is well under a year (we verify
below), so a 2-year silence is a strong inactivity signal. We also check the
threshold's sensitivity and the resulting class balance (assignment Step 2).

Guiding principle of the whole unit: **better features beat better algorithms.**""")

code("""import sys, warnings
sys.path.insert(0, "../app")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import features as F
import model as M
import recommender as R

RNG = 42
np.random.seed(RNG)
plt.rcParams["figure.figsize"] = (9, 5)
pd.set_option("display.width", 140)""")

# ----------------------------------------------------------------- data
md("""## 1. Raw data

Two-stage collection (see `app/scraper.py`, rate-limited at 3 s/request per
arXiv's terms of use):

1. **Sampling frame** — 100 papers per quarter, 2014–2025, category `cs.CL`.
   Sampling across *all* years matters: a "most recent papers" query would
   contain almost no churned researchers (sampling bias).
2. **Full histories** — for every researcher appearing ≥ 2 times in the frame,
   one `au:"Name"` query retrieves their complete arXiv record, so recency and
   the label reflect **real** activity, not a sampling artifact.""")

code("""corpus = pd.read_csv("../data/raw/papers.csv", parse_dates=["published"])
hist = pd.read_csv("../data/raw/author_papers.csv", parse_dates=["published"])
hist = hist.rename(columns={"queried_author": "author"})
hist = F.add_author_position(hist)

print(f"corpus: {len(corpus)} papers  |  histories: {len(hist)} author-paper rows "
      f"for {hist.author.nunique()} researchers")
corpus.head(3)""")

code("""fig, ax = plt.subplots()
corpus.published.dt.year.value_counts().sort_index().plot(kind="bar", ax=ax, color="#4878a8")
ax.set_title("Corpus sample: papers per year (quarterly stratified sample)")
ax.set_xlabel("year"); ax.set_ylabel("papers")
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- label
md("""## 2. Churn label — definition, justification, class balance

`churned = 1` ⇔ no submission after **2024-06-30** (the cutoff). Features are
computed strictly from pre-cutoff history. Eligibility: ≥ 3 pre-cutoff papers
and ≥ 1 year of history (you cannot meaningfully call a brand-new author
"churned").""")

code("""df = F.build_author_features(hist, cutoff=F.CUTOFF)
y = df["churned"]
print(f"researchers: {len(df)}")
print(f"churn rate: {y.mean():.1%}  ({y.sum()} churned / {(1 - y).sum()} active)")

# median gap between consecutive papers (active authors) supports the threshold
active_gaps = df.loc[y == 0, "avg_gap_days"]
print(f"median avg-gap for ACTIVE authors: {active_gaps.median():.0f} days "
      f"-> 2 years of silence is far outside normal rhythm")""")

code("""# threshold sensitivity: how does the churn rate move with the cutoff?
rates = {}
for c in ["2023-06-30", "2023-12-31", "2024-06-30", "2024-12-31"]:
    d = F.build_author_features(hist, cutoff=pd.Timestamp(c))
    rates[c] = d.churned.mean()
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(rates.keys(), rates.values(), color=["#bbb", "#bbb", "#c0504d", "#bbb"])
ax.set_title("Churn-rate sensitivity to the cutoff date (chosen: 2024-06-30)")
ax.set_ylabel("churn rate"); ax.axhline(0.10, ls="--", c="k", lw=1)
plt.tight_layout(); plt.show()
print({k: f"{v:.1%}" for k, v in rates.items()})""")

md("""Class balance is comfortably above the ~10% danger zone, and we still pass
`class_weight="balanced"` to every model as a guard. **Decision:** keep the
2024-06-30 cutoff (≈ 2-year silence window).""")

# ----------------------------------------------------------------- features
md("""## 3. Engineered features (Step 3 — the most important step)

No raw API field enters the model. One-line justification per feature:

| feature | type | why it should predict churn |
|---|---|---|
| `recency_days_at_cutoff` | time (recency) | the classic #1 churn signal: silence already growing before the cutoff |
| `papers_per_year` | time (frequency) | low publication cadence = weak attachment to the field |
| `avg_gap_days` | time | habitual rhythm; long average gaps normalize disappearing |
| `recent_share_2y` | time (momentum) | share of output in the last 2 pre-cutoff years; fading momentum precedes exit |
| `solo_ratio` | ratio | solo authors lack collaborator "pull" to keep publishing |
| `first_author_ratio` | ratio | first authors are typically PhD students/postdocs — high attrition population |
| `categories_per_paper` | ratio | topical breadth per paper; narrow scope = fragile attachment |
| `avg_coauthors` | aggregation | team size proxies lab/network support (magnitude) |
| `max_gap_days` | aggregation | longest past silence; past breaks predict future breaks |
| `career_years` | aggregation | seniority; veterans behave differently from newcomers |
| `n_categories` | aggregation | total topical range across the career |
| `is_solo_researcher` | binary | >50% solo papers — isolation flag |
| `has_long_break` | binary | ever vanished >1.5 years before — relapse risk |
| `is_multidisciplinary` | binary | ≥4 categories — diversified researchers have more reasons to stay |

Recency, frequency and magnitude are all covered (assignment requirement).""")

code("""X = df[F.FEATURE_COLUMNS]
X.describe().T.round(2)""")

# ----------------------------------------------------------------- selection
md("""## 4. Feature selection — all four methods

1. **Filter**: variance threshold, correlation pruning (>0.9), ANOVA F-test
2. **Wrapper**: RFE with LogisticRegression (top 5)
3. **Embedded**: DecisionTree importances (max_depth=5)
4. **Embedded**: RandomForest importances (n_estimators=100)""")

code("""# --- 4.1 filter: variance threshold (constant features carry zero signal)
variances = X.var().sort_values()
print("lowest variances:")
print(variances.head(4).round(4))
kept_v = M.variance_prune(X)
const = [c for c in X.columns if c not in kept_v]
print("dropped as constant:", const or "none")
Xv = X[kept_v]""")

code("""# --- 4.1 filter: correlation matrix + pruning at |r| > 0.9
corr = Xv.corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
fig.colorbar(im); ax.set_title("Feature correlation matrix")
plt.tight_layout(); plt.show()

kept = M.correlation_prune(Xv)
dropped = [c for c in Xv.columns if c not in kept]
print("dropped by correlation pruning (>0.9):", dropped or "none")
Xp = Xv[kept]""")

code("""# --- 4.2-4.4: ANOVA + RFE + DecisionTree + RandomForest, one consolidated table
table = M.selection_scores(Xp, y)
votes = ((table.anova_rank <= 8).astype(int) + table.rfe_selected.astype(int)
         + (table.dt_rank <= 8).astype(int) + (table.rf_rank <= 8).astype(int))
table["votes"] = votes
table["decision"] = np.select([votes >= 3, votes == 2], ["KEEP", "OPTIONAL"], "DROP")
table""")

md("""### Reading the disagreements

* **Filters score features in isolation** — ANOVA loves anything monotonically
  related to churn but misses interactions; RFE (a wrapper) evaluates features
  *jointly* and can keep a feature that only matters in combination.
* **A single decision tree is unstable** — one resample can reshuffle its
  ranking; the **random forest averages 100 trees**, so when DT and RF
  disagree, we trust RF.
* The consensus rule (≥2 of 4 methods, ordered by RF importance) is exactly
  what `model.py::consensus_selection` ships to production — notebook and API
  cannot drift apart.""")

code("""selected = M.consensus_selection(X, y)
print(f"final feature set ({len(selected)}):")
selected""")

code("""# --- validate the final set: 5-fold stratified cross-validation
cv_selected = M.cross_validate_model(df[selected], y)
print("RandomForest on selected features (5-fold CV):")
for k, v in cv_selected.items():
    print(f"  {k:9s} {v:.3f}")""")

# ----------------------------------------------------------------- pca / svd
md("""## 5. Dimensionality reduction — PCA & SVD (Step 6)

The four methods above **select** existing features; PCA/SVD **create new
compressed ones**. PCA on standardized features ≈ SVD of the centered
covariance matrix — same math, different entry point.""")

code("""from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)
pca = PCA(random_state=RNG).fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(cum) + 1), cum, "o-", color="#4878a8")
ax.axhline(0.9, ls="--", c="k", lw=1, label="90% variance")
ax.set_xlabel("number of components"); ax.set_ylabel("cumulative explained variance")
ax.set_title("PCA elbow plot"); ax.legend()
plt.tight_layout(); plt.show()
print("first 5 components explain:", (cum[:5] * 100).round(1), "%")""")

code("""pcs = PCA(n_components=2, random_state=RNG).fit_transform(Xs)
fig, ax = plt.subplots(figsize=(8, 6))
for label, color, name in [(0, "#2e8b57", "active"), (1, "#c0504d", "churned")]:
    m = (y == label).to_numpy()
    ax.scatter(pcs[m, 0], pcs[m, 1], c=color, s=28, alpha=0.7, label=name)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
ax.set_title("Researchers in PCA space, colored by churn")
plt.tight_layout(); plt.show()""")

md("""Partial cluster separation along PC1 (dominated by recency/frequency
loadings) confirms the features carry real signal — separation in 2D *is* the
visual version of "strong features". The overlap region is the genuinely
uncertain population. Compressing 14 features into 2–3 components inevitably
loses the fine-grained information the forest exploits, which the model
comparison below quantifies.""")

code("""# SVD on the researcher x category interaction matrix (recommender input)
matrix, m_authors, m_cats = R.build_interaction_matrix(hist, F.CUTOFF)
svd = R.fit_svd(matrix)
print(f"interaction matrix: {matrix.shape[0]} researchers x {matrix.shape[1]} categories")
print(f"SVD with k={len(svd['sigma'])} factors explains {svd['explained_variance']:.1%} of variance")

fig, ax = plt.subplots(figsize=(7, 4))
share = svd["sigma"] ** 2 / (svd["sigma"] ** 2).sum()
ax.bar(range(1, len(share) + 1), share, color="#4878a8")
ax.set_title("SVD scree: variance share per latent factor (researcher x category)")
ax.set_xlabel("factor"); ax.set_ylabel("share")
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------- network
md("""## 6. Network analysis as feature enrichment (Step 8)

Co-authorship graph: nodes = researchers in the dataset, edge = they share at
least one pre-cutoff paper. Centralities become **new features**:

* `degree_centrality` — how many collaborators inside the community
* `betweenness` — bridge between research groups
* `pagerank` — influence weighted by influential collaborators""")

code("""pre = hist[hist.published <= F.CUTOFF]
in_set = set(df.author)
G = nx.Graph()
G.add_nodes_from(in_set)
from itertools import combinations
for _, grp in pre.groupby("arxiv_id"):
    names = sorted(set(grp.author) & in_set)
    G.add_edges_from(combinations(names, 2))
print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"{nx.number_connected_components(G)} components")

deg = nx.degree_centrality(G)
btw = nx.betweenness_centrality(G, seed=RNG)
pr = nx.pagerank(G)
df["degree_centrality"] = df.author.map(deg).fillna(0)
df["betweenness"] = df.author.map(btw).fillna(0)
df["pagerank"] = df.author.map(pr).fillna(0)""")

code("""pos = nx.spring_layout(G, seed=RNG, k=0.35)
colors = ["#c0504d" if c else "#2e8b57"
          for c in df.set_index("author").loc[list(G.nodes), "churned"]]
sizes = [60 + 900 * deg[n] for n in G.nodes]
fig, ax = plt.subplots(figsize=(10, 8))
nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85, ax=ax)
ax.set_title("Co-authorship network — red = churned, green = active "
             "(node size = degree)")
ax.axis("off"); plt.tight_layout(); plt.show()""")

code("""# re-run ALL FOUR selection methods with network features included
X_ext = df[F.FEATURE_COLUMNS + F.NETWORK_FEATURE_COLUMNS]
kept_ext = M.correlation_prune(X_ext[M.variance_prune(X_ext)])
table_ext = M.selection_scores(X_ext[kept_ext], y)
votes_ext = ((table_ext.anova_rank <= 8).astype(int) + table_ext.rfe_selected.astype(int)
             + (table_ext.dt_rank <= 8).astype(int) + (table_ext.rf_rank <= 8).astype(int))
table_ext["votes"] = votes_ext
table_ext["decision"] = np.select([votes_ext >= 3, votes_ext == 2],
                                  ["KEEP", "OPTIONAL"], "DROP")
table_ext""")

code("""selected_ext = M.consensus_selection(X_ext, y)
print("consensus set WITH network features:")
selected_ext""")

# ----------------------------------------------------------------- comparison
md("""## 7. Model comparison — three setups (Step 10 input)

| setup | features |
|---|---|
| A | consensus-selected **original** features |
| B | **PCA components** (scaler + PCA(3) inside the CV pipeline — no leakage) |
| C | original consensus set **+ network centralities** |""")

code("""from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
scoring = ["accuracy", "precision", "recall", "f1"]

def run_cv(estimator, Xc):
    s = cross_validate(estimator, Xc, y, cv=cv, scoring=scoring)
    return {m: np.mean(s[f"test_{m}"]) for m in scoring}

rf = lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=RNG)
pca_pipe = Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=3, random_state=RNG)),
                     ("rf", rf())])

results = pd.DataFrame({
    "A: selected original": run_cv(rf(), df[selected]),
    "B: PCA(3) components": run_cv(pca_pipe, X),
    "C: original + network": run_cv(rf(), df[selected + F.NETWORK_FEATURE_COLUMNS]),
}).T.round(3)
results""")

code("""fig, ax = plt.subplots(figsize=(8, 4.5))
results["f1"].plot(kind="barh", ax=ax, color=["#4878a8", "#bbb", "#2e8b57"])
ax.set_title("F1 by setup (5-fold CV)"); ax.set_xlabel("F1")
plt.tight_layout(); plt.show()""")

md("""## 8. Conclusions

* **Recency and frequency dominate** every selection method — the unit's motto
  ("better features beat better algorithms") holds: a plain RandomForest with
  good temporal features beats anything fancy on raw fields.
* **PCA components lose information** relative to the selected originals: PCA
  compresses for *variance*, not for *churn discrimination* — useful as a
  visual diagnostic, not as the production feature set.
* **Network centralities add marginal but real signal**: isolated researchers
  (low degree/PageRank) churn more — collaboration is retention, which is also
  the actionable lever for the product recommendations in `REPORT.md`.
* The served API uses setup **A** (portable: a client can compute every input
  from a researcher's public record alone, no graph recomputation needed);
  setup C is the offline "analyst's model". Full discussion in REPORT.md.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
})

NB_PATH.parent.mkdir(exist_ok=True)
nbf.write(nb, NB_PATH)
print(f"notebook written -> {NB_PATH}")

client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                        resources={"metadata": {"path": str(NB_PATH.parent)}})
client.execute()
nbf.write(nb, NB_PATH)
print("notebook executed top-to-bottom OK")
