import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import features as F
import model as M

hist = pd.read_csv(ROOT / "data/raw/author_papers.csv", parse_dates=["published"])
hist = hist.rename(columns={"queried_author": "author"})
hist = F.add_author_position(hist)
df = F.build_author_features(hist)
X, y = df[F.FEATURE_COLUMNS], df["churned"]

Xv = X[M.variance_prune(X)]
print("constant:", [c for c in X.columns if c not in Xv.columns])
kept = M.correlation_prune(Xv)
print("corr-dropped:", [c for c in Xv.columns if c not in kept])
table = M.selection_scores(Xv[kept], y)
print(table.to_string())
