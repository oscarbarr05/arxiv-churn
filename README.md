# arXiv Researcher Churn Predictor

A fully Dockerized churn-prediction pipeline for **academic researchers**:
predicts which arXiv (`cs.CL`) authors will **stop publishing**, explains why
(four feature-selection methods, PCA/SVD, co-authorship network analysis), and
acts on it by recommending **recent papers** to re-engage at-risk researchers —
the academic analogue of Netflix/Spotify retention systems.

**Data source:** the public [arXiv API](https://info.arxiv.org/help/api/index.html)
— **no API key, no login, no manual steps**. The repo ships with the raw data
(`data/raw/`) and trained artifacts (`app/model.pkl`, `app/recommender.pkl`)
committed, so everything works straight from a clean clone.

---

## Quick start (the only command needed)

```bash
docker-compose up --build
```

Then open **http://localhost:8000** for the **web dashboard** (predict with
sliders, browse at-risk researchers, get paper recommendations). The technical
interactive API docs stay at **http://localhost:8000/docs**.

### Try it

```bash
# health check
curl http://localhost:8000/health

# which inputs does the model expect? (+ churn definition + CV metrics)
curl http://localhost:8000/features

# predict: a dormant, isolated researcher profile
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "recency_days_at_cutoff": 600, "papers_per_year": 0.8, "avg_gap_days": 400,
  "recent_share_2y": 0.1, "solo_ratio": 0.6, "categories_per_paper": 0.5,
  "career_years": 6, "n_categories": 2}'
# -> {"churned": true, "churn_probability": ...}
# (only the 8 model features are required; see GET /features)

# list researchers at churn risk (valid names for /recommend)
curl "http://localhost:8000/authors?at_risk_only=true"

# recommend re-engagement papers for an at-risk researcher
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" \
  -d '{"author": "<NAME FROM /authors>", "top_n": 5}'
```

`/recommend` only returns papers when the researcher's churn probability is
**≥ 0.5**; engaged researchers get a "not at risk" message (assignment rule).

---

## Churn definition (and why it is leakage-free)

> **A researcher has churned if they submitted no arXiv paper after the
> cutoff date 2024-06-30** (≈ 2 years of silence at evaluation time).

Instead of a "days inactive > 180" label computed *today* — which would leak
recency into both the label and the features — we split time itself:

* **Features** are computed **only from papers up to the cutoff**
  (recency is measured *at* the cutoff, so it is a legal predictor).
* **The label** looks **only at what happens after** the cutoff.

This mirrors production churn systems: observe behaviour up to "today",
predict inactivity over the next window. Two years fits academia's rhythm:
active `cs.CL` authors publish with median gaps far below one year (verified
in the notebook), and the threshold's class balance is checked there too —
all models additionally use `class_weight="balanced"`.

## Data collection (two stages, `app/scraper.py`)

1. **Sampling frame:** 100 papers per quarter, 2014–2025 (~4 800 papers).
   Sampling across all years avoids the "recent papers only" bias that would
   exclude churned researchers from the dataset.
2. **Full histories:** for each researcher appearing ≥ 2 times in the frame,
   one `au:"Name"` query fetches their complete record, so the label reflects
   **real** activity. Rate-limited at 3 s/request per arXiv's terms of use;
   missing fields handled defensively; pagination supported.

## Expected `/predict` fields

All 14 engineered features (definitions in `app/features.py` and the
notebook). The model uses the consensus-selected subset reported by
`GET /features`; extra fields are accepted and ignored.

| field | meaning |
|---|---|
| `recency_days_at_cutoff` | days since last paper, measured at the cutoff |
| `papers_per_year` | publication frequency over the career |
| `avg_gap_days` / `max_gap_days` | typical / worst silence between papers |
| `recent_share_2y` | share of output in the last 2 pre-cutoff years |
| `solo_ratio` / `is_solo_researcher` | solo-authorship share / >50% flag |
| `first_author_ratio` | share of papers as first author |
| `categories_per_paper` / `n_categories` / `is_multidisciplinary` | topical breadth |
| `avg_coauthors` | average team size |
| `career_years` | seniority |
| `has_long_break` | ever silent > 1.5 years before the cutoff |

## Repository structure

```
arxiv-churn/
├── app/
│   ├── main.py            # FastAPI — THIN, endpoints only
│   ├── model.py           # consensus selection, train/save/load, CV
│   ├── features.py        # raw -> engineered features (shared with notebook)
│   ├── scraper.py         # arXiv fetcher — reusable, parameterized, no modeling
│   ├── recommender.py     # SVD collaborative filtering for /recommend
│   ├── model.pkl          # trained RandomForest bundle (committed)
│   └── recommender.pkl    # SVD factors + candidates + churn probs (committed)
├── notebooks/eda_and_selection.ipynb   # full analytical audit trail
├── data/raw/              # papers.csv + author_papers.csv (committed)
├── scripts/               # data collection + notebook generation (author-side)
├── build_model.py         # reproduce model.pkl + recommender.pkl from raw data
├── Dockerfile / docker-compose.yml / requirements.txt
├── README.md              # this file
└── REPORT.md              # written report (3–5 pages)
```

## Rebuilding from scratch (optional, not needed for grading)

```bash
python scripts/collect_data.py     # ~30 min (arXiv rate limit)
python build_model.py              # features -> selection -> CV -> .pkl files
python scripts/make_notebook.py    # regenerate + execute the notebook
```
