"""Train / save / load the churn model. Bridge between analysis and the API.

The notebook runs the full audit (4 selection methods, PCA/SVD, network);
this module holds the *reproducible production path*: the same consensus
selection rule the notebook arrives at, the final RandomForest, and the
joblib persistence used by the FastAPI app.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
RANDOM_STATE = 42


def correlation_prune(X: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Drop one feature of every pair with |corr| > threshold (filter step)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return [c for c in X.columns if c not in drop]


def selection_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run the four selection methods; return one row per feature."""
    Xs = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns, index=X.index
    )

    # 1) filter: ANOVA F-test
    skb = SelectKBest(f_classif, k="all").fit(Xs, y)
    anova = pd.Series(skb.scores_, index=X.columns)

    # 2) wrapper: RFE with logistic regression, keep top 5
    rfe = RFE(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        n_features_to_select=5,
    ).fit(Xs, y)
    rfe_sel = pd.Series(rfe.support_, index=X.columns)

    # 3) decision tree importances
    dt = DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
    ).fit(X, y)
    dt_imp = pd.Series(dt.feature_importances_, index=X.columns)

    # 4) random forest importances
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    ).fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)

    table = pd.DataFrame(
        {
            "anova_F": anova.round(2),
            "anova_rank": anova.rank(ascending=False).astype(int),
            "rfe_selected": rfe_sel,
            "dt_importance": dt_imp.round(4),
            "dt_rank": dt_imp.rank(ascending=False).astype(int),
            "rf_importance": rf_imp.round(4),
            "rf_rank": rf_imp.rank(ascending=False).astype(int),
        }
    )
    return table.sort_values("rf_importance", ascending=False)


def consensus_selection(X: pd.DataFrame, y: pd.Series, keep: int = 8) -> list[str]:
    """Reproducible final feature set.

    1. correlation pruning (>0.9) removes redundant twins,
    2. a feature earns one vote per method that ranks it top-`keep`
       (ANOVA, DT, RF) or selects it (RFE),
    3. keep features with >= 2 votes, ordered by RF importance.
    """
    kept_cols = correlation_prune(X)
    Xp = X[kept_cols]
    table = selection_scores(Xp, y)
    votes = (
        (table["anova_rank"] <= keep).astype(int)
        + table["rfe_selected"].astype(int)
        + (table["dt_rank"] <= keep).astype(int)
        + (table["rf_rank"] <= keep).astype(int)
    )
    selected = table.index[votes >= 2].tolist()
    return sorted(selected, key=lambda c: -table.loc[c, "rf_importance"])


def cross_validate_model(
    X: pd.DataFrame, y: pd.Series, n_estimators: int = 300
) -> dict[str, float]:
    """5-fold stratified CV -> accuracy / precision / recall / F1."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"]
    )
    return {
        m: float(np.mean(scores[f"test_{m}"]))
        for m in ["accuracy", "precision", "recall", "f1"]
    }


def train_and_save(
    df: pd.DataFrame,
    feature_names: list[str],
    path: Path = MODEL_PATH,
) -> dict:
    """Fit the final RandomForest on the selected features and persist it."""
    X, y = df[feature_names], df["churned"]
    metrics = cross_validate_model(X, y)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    ).fit(X, y)
    bundle = {
        "model": model,
        "features": feature_names,
        "cv_metrics": metrics,
        "churn_definition": "no arXiv submission after 2024-06-30",
    }
    joblib.dump(bundle, path)
    return bundle


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_one(bundle: dict, payload: dict) -> tuple[bool, float]:
    """payload: feature_name -> value (extra keys ignored)."""
    row = pd.DataFrame([{f: payload[f] for f in bundle["features"]}])
    proba = float(bundle["model"].predict_proba(row)[0, 1])
    return proba >= 0.5, proba
