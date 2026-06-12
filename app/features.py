"""Feature engineering: raw papers -> one engineered row per researcher.

Imported by BOTH the notebook and model.py so analysis and serving can never
drift apart (single source of truth for transformations).

Temporal design (avoids label leakage)
--------------------------------------
We pick a CUTOFF date. Every feature is computed ONLY from papers submitted
on or before the cutoff. The churn label is computed ONLY from what happens
AFTER the cutoff:

    churned = 1  <=>  the researcher submitted no paper after CUTOFF

This mirrors how churn models work in production (Netflix/Spotify): you
observe behaviour up to "today" and predict inactivity over the NEXT window.
It also makes recency a legal feature — recency is measured *at* the cutoff,
the label lives strictly after it.

No raw API field is used directly as a feature: everything is a ratio,
time-derived, aggregated or binary transformation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2024-06-30")
MIN_PAPERS_PRE_CUTOFF = 3        # need history to compute gaps/ratios
MIN_HISTORY_YEARS = 1.0          # exclude brand-new authors (no churn risk yet)

# the exact column order the model trains on (pre-network features)
FEATURE_COLUMNS = [
    # --- time-based (recency + frequency) ---
    "recency_days_at_cutoff",
    "papers_per_year",
    "avg_gap_days",
    "recent_share_2y",
    # --- ratios ---
    "solo_ratio",
    "first_author_ratio",
    "categories_per_paper",
    # --- aggregations ---
    "avg_coauthors",
    "max_gap_days",
    "career_years",
    "n_categories",
    # --- binary flags ---
    "is_solo_researcher",
    "has_long_break",
    "is_multidisciplinary",
]

NETWORK_FEATURE_COLUMNS = ["degree_centrality", "betweenness", "pagerank"]


def add_author_position(hist: pd.DataFrame) -> pd.DataFrame:
    """Author-history rows (author, authors pipe-joined) -> + author_position."""
    hist = hist.copy()
    positions = []
    for rec in hist.itertuples(index=False):
        names = str(rec.authors).split("|")
        positions.append(names.index(rec.author) if rec.author in names else -1)
    hist["author_position"] = positions
    return hist


def explode_author_papers(papers: pd.DataFrame) -> pd.DataFrame:
    """papers (one row per paper, authors pipe-joined) -> one row per
    (author, paper) with the author's position in the author list."""
    rows = []
    for rec in papers.itertuples(index=False):
        names = str(rec.authors).split("|")
        for pos, name in enumerate(names):
            rows.append(
                {
                    "author": name,
                    "arxiv_id": rec.arxiv_id,
                    "published": rec.published,
                    "n_authors": rec.n_authors,
                    "primary_category": rec.primary_category,
                    "categories": rec.categories,
                    "author_position": pos,
                }
            )
    out = pd.DataFrame(rows)
    out["published"] = pd.to_datetime(out["published"])
    return out


def build_author_features(
    author_papers: pd.DataFrame,
    cutoff: pd.Timestamp = CUTOFF,
) -> pd.DataFrame:
    """One engineered row per researcher + the churn label.

    `author_papers` must have columns: author, arxiv_id, published,
    n_authors, primary_category, categories, author_position.
    """
    ap = author_papers.copy()
    ap["published"] = pd.to_datetime(ap["published"])

    pre = ap[ap["published"] <= cutoff]
    post = ap[ap["published"] > cutoff]
    active_after = set(post["author"].unique())

    rows = []
    for author, g in pre.groupby("author"):
        g = g.sort_values("published").drop_duplicates(subset="arxiv_id")
        n = len(g)
        if n < MIN_PAPERS_PRE_CUTOFF:
            continue

        first, last = g["published"].iloc[0], g["published"].iloc[-1]
        career_days = (last - first).days
        career_years = max(career_days / 365.25, 0.1)
        if (cutoff - first).days / 365.25 < MIN_HISTORY_YEARS:
            continue

        gaps = g["published"].diff().dt.days.dropna()
        avg_gap = float(gaps.mean()) if len(gaps) else career_days
        max_gap = float(gaps.max()) if len(gaps) else career_days

        cats: set[str] = set()
        for c in g["categories"]:
            cats.update(str(c).split("|"))
        n_cats = len(cats)

        solo = int((g["n_authors"] == 1).sum())
        first_auth = int((g["author_position"] == 0).sum())
        recent_2y = int((g["published"] >= cutoff - pd.Timedelta(days=730)).sum())

        rows.append(
            {
                "author": author,
                # time-based
                "recency_days_at_cutoff": float((cutoff - last).days),
                "papers_per_year": n / career_years,
                "avg_gap_days": avg_gap,
                "recent_share_2y": recent_2y / n,
                # ratios
                "solo_ratio": solo / n,
                "first_author_ratio": first_auth / n,
                "categories_per_paper": n_cats / n,
                # aggregations
                "avg_coauthors": float((g["n_authors"] - 1).mean()),
                "max_gap_days": max_gap,
                "career_years": career_years,
                "n_categories": float(n_cats),
                # binary
                "is_solo_researcher": int(solo / n > 0.5),
                "has_long_break": int(max_gap > 548),          # > 1.5 years
                "is_multidisciplinary": int(n_cats >= 4),
                # label: no submissions after the cutoff
                "churned": int(author not in active_after),
                # bookkeeping (NOT features)
                "n_papers_pre_cutoff": n,
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)
