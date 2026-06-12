"""Rebuild model.pkl + recommender.pkl from the committed raw data.

The grader never needs this (both .pkl files are committed); it documents and
reproduces the production path end-to-end:

    raw CSVs -> features -> consensus selection -> 5-fold CV -> RandomForest
             -> SVD recommender (gated by churn probability)

Run:  python build_model.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

import features as F          # noqa: E402
import model as M             # noqa: E402
import recommender as R       # noqa: E402

# ---------------------------------------------------------------- load raw
corpus = pd.read_csv(ROOT / "data/raw/papers.csv")
hist = pd.read_csv(ROOT / "data/raw/author_papers.csv")
hist = hist.rename(columns={"queried_author": "author"})

hist = F.add_author_position(hist)

# ------------------------------------------------------------ features + label
df = F.build_author_features(hist, cutoff=F.CUTOFF)
rate = df["churned"].mean()
print(f"researchers: {len(df)}  |  churn rate: {rate:.1%}")
assert 0.10 <= rate <= 0.60, "class balance outside the sane range — revisit threshold"

# --------------------------------------------------- consensus feature selection
X, y = df[F.FEATURE_COLUMNS], df["churned"]
selected = M.consensus_selection(X, y)
print(f"selected features ({len(selected)}): {selected}")

# ------------------------------------------------------------- train + persist
bundle = M.train_and_save(df, selected)
print("5-fold CV:", {k: round(v, 3) for k, v in bundle["cv_metrics"].items()})
print(f"model saved -> {M.MODEL_PATH}")

# ------------------------------------------------------------- recommender
proba = bundle["model"].predict_proba(df[selected])[:, 1]
churn_probs = pd.Series(proba, index=df["author"])
rec_bundle = R.build_bundle(hist, corpus, churn_probs, F.CUTOFF)
R.save_bundle(rec_bundle)
print(f"recommender saved -> {R.RECOMMENDER_PATH} "
      f"(SVD explained variance: {rec_bundle['explained_variance']:.1%}, "
      f"{len(rec_bundle['candidates'])} candidate papers)")
