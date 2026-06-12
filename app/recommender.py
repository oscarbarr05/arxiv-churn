"""SVD collaborative-filtering recommender for researcher re-engagement.

Interaction matrix: researcher x arXiv category (how many of their pre-cutoff
papers touch each category). Truncated SVD (scipy.sparse.linalg.svds)
compresses it into latent "research interest" factors; an at-risk researcher
is then recommended RECENT papers written by ACTIVE authors whose category
profile is closest to the researcher's latent interests — the academic
analogue of "we miss you, here is new content in the topics you loved".

Business rule (per assignment): recommendations are ONLY returned when the
researcher's churn probability is >= 0.5; engaged researchers get a
"not at risk" message instead.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

RECOMMENDER_PATH = Path(__file__).resolve().parent / "recommender.pkl"
N_FACTORS = 12


def build_interaction_matrix(
    author_papers: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """researcher x category count matrix from pre-cutoff papers."""
    ap = author_papers.copy()
    ap["published"] = pd.to_datetime(ap["published"])
    pre = ap[ap["published"] <= cutoff]

    rows = []
    for rec in pre.itertuples(index=False):
        for cat in str(rec.categories).split("|"):
            if cat:
                rows.append({"author": rec.author, "category": cat})
    long = pd.DataFrame(rows)
    matrix = long.value_counts().unstack(fill_value=0)
    return matrix, list(matrix.index), list(matrix.columns)


def fit_svd(matrix: pd.DataFrame, k: int = N_FACTORS) -> dict:
    """Truncated SVD on the (log-damped, centered) interaction matrix."""
    dense = np.log1p(matrix.to_numpy(dtype=float))
    row_means = dense.mean(axis=1, keepdims=True)
    centered = dense - row_means
    k = min(k, min(centered.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(centered), k=k)
    order = np.argsort(sigma)[::-1]            # svds returns ascending
    U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]
    total_var = float((centered ** 2).sum())
    explained = float((sigma ** 2).sum()) / total_var if total_var else 0.0
    return {
        "author_factors": U * sigma,           # researchers in latent space
        "category_factors": Vt,                # categories in latent space
        "explained_variance": explained,
        "sigma": sigma,
    }


def build_bundle(
    author_papers: pd.DataFrame,
    corpus: pd.DataFrame,
    churn_probs: pd.Series,
    cutoff: pd.Timestamp,
) -> dict:
    """Everything /recommend needs, fully precomputed at build time."""
    matrix, authors, categories = build_interaction_matrix(author_papers, cutoff)
    svd = fit_svd(matrix)

    # candidate items: papers from the last year of the corpus whose authors
    # are still active (someone published after the cutoff)
    corpus = corpus.copy()
    corpus["published"] = pd.to_datetime(corpus["published"])
    ap = author_papers.copy()
    ap["published"] = pd.to_datetime(ap["published"])
    active_authors = set(ap.loc[ap["published"] > cutoff, "author"])
    recent = corpus[corpus["published"] >= corpus["published"].max()
                    - pd.Timedelta(days=365)]
    is_active = recent["authors"].apply(
        lambda s: any(a in active_authors for a in str(s).split("|"))
    )
    candidates = recent[is_active][
        ["arxiv_id", "title", "published", "authors", "categories"]
    ].reset_index(drop=True)

    return {
        "authors": authors,
        "categories": categories,
        "author_factors": svd["author_factors"],
        "category_factors": svd["category_factors"],
        "explained_variance": svd["explained_variance"],
        "candidates": candidates,
        "churn_probs": churn_probs.reindex(authors).fillna(0.0),
        "cutoff": str(cutoff.date()),
    }


def save_bundle(bundle: dict, path: Path = RECOMMENDER_PATH) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: Path = RECOMMENDER_PATH) -> dict:
    return joblib.load(path)


def _paper_vector(categories: str, bundle: dict) -> np.ndarray | None:
    """A paper's latent vector = mean of its categories' latent vectors."""
    idx = [
        bundle["categories"].index(c)
        for c in str(categories).split("|")
        if c in bundle["categories"]
    ]
    if not idx:
        return None
    return bundle["category_factors"][:, idx].mean(axis=1)


def recommend(bundle: dict, author: str, top_n: int = 5) -> dict:
    """Top-N re-engagement papers for an at-risk researcher."""
    if author not in bundle["authors"]:
        return {"error": f"unknown researcher: {author!r}",
                "hint": "GET /authors lists valid names"}

    prob = float(bundle["churn_probs"][author])
    if prob < 0.5:
        return {
            "author": author,
            "churn_probability": round(prob, 3),
            "at_risk": False,
            "message": "Researcher is not at churn risk (prob < 0.5); "
                       "no re-engagement needed.",
        }

    i = bundle["authors"].index(author)
    user_vec = bundle["author_factors"][i]
    norm_u = np.linalg.norm(user_vec) or 1.0

    scored = []
    for rec in bundle["candidates"].itertuples(index=False):
        if author in str(rec.authors).split("|"):
            continue                                    # never self-recommend
        vec = _paper_vector(rec.categories, bundle)
        if vec is None:
            continue
        score = float(user_vec @ vec / (norm_u * (np.linalg.norm(vec) or 1.0)))
        scored.append((score, rec))
    scored.sort(key=lambda t: -t[0])

    return {
        "author": author,
        "churn_probability": round(prob, 3),
        "at_risk": True,
        "recommendations": [
            {
                "arxiv_id": rec.arxiv_id,
                "title": rec.title,
                "published": str(rec.published)[:10],
                "categories": rec.categories,
                "similarity": round(score, 4),
            }
            for score, rec in scored[:top_n]
        ],
    }
