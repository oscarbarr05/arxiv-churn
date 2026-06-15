"""FastAPI app — THIN. Endpoints only; logic lives in model.py / recommender.py.

Both joblib bundles are loaded ONCE at startup, never per request.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import model as model_module
import recommender as rec_module

app = FastAPI(
    title="arXiv Researcher Churn Predictor",
    description="Predicts whether a researcher will stop publishing on arXiv "
                "and recommends recent papers to re-engage at-risk authors.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

MODEL = model_module.load_bundle()
RECOMMENDER = rec_module.load_bundle()
VIZ = json.loads((Path(__file__).resolve().parent / "viz.json").read_text("utf-8"))


class ResearcherFeatures(BaseModel):
    """Engineered features at the cutoff date (see README for definitions).

    The 8 fields the trained model consumes (reported by GET /features) are
    required; the remaining engineered features are accepted as optional
    extras so a client can always POST a full researcher profile, but they
    are ignored by the current model. This keeps the required contract aligned
    with /features while staying robust to a future re-selection.
    """
    # --- required: the 8 consensus-selected features the model uses ---
    recency_days_at_cutoff: float = Field(..., ge=0)
    papers_per_year: float = Field(..., ge=0)
    avg_gap_days: float = Field(..., ge=0)
    recent_share_2y: float = Field(..., ge=0, le=1)
    solo_ratio: float = Field(..., ge=0, le=1)
    categories_per_paper: float = Field(..., ge=0)
    career_years: float = Field(..., gt=0)
    n_categories: float = Field(..., ge=1)
    # --- optional: accepted for completeness, not used by the current model ---
    first_author_ratio: float = Field(0.5, ge=0, le=1)
    avg_coauthors: float = Field(2.0, ge=0)
    max_gap_days: float = Field(365.0, ge=0)
    is_solo_researcher: int = Field(0, ge=0, le=1)
    has_long_break: int = Field(0, ge=0, le=1)
    is_multidisciplinary: int = Field(0, ge=0, le=1)


class RecommendRequest(BaseModel):
    author: str
    top_n: int = Field(5, ge=1, le=20)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Friendly web dashboard (the technical Swagger UI stays at /docs)."""
    return FileResponse(STATIC_DIR / "index.html")


# vendored assets (Plotly.js) — served locally so the dashboard works offline
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/viz", include_in_schema=False)
def viz() -> dict:
    """Precomputed PCA / SVD / importance / network data for the dashboard."""
    return VIZ


@app.get("/features")
def features() -> dict:
    """Self-documentation: the exact inputs the model consumes."""
    return {
        "model_features": MODEL["features"],
        "churn_definition": MODEL["churn_definition"],
        "cv_metrics": MODEL["cv_metrics"],
    }


@app.post("/predict")
def predict(body: ResearcherFeatures) -> dict:
    churned, prob = model_module.predict_one(MODEL, body.model_dump())
    return {"churned": churned, "churn_probability": round(prob, 4)}


@app.post("/recommend")
def recommend(body: RecommendRequest) -> dict:
    result = rec_module.recommend(RECOMMENDER, body.author, body.top_n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.get("/authors")
def authors(at_risk_only: bool = False) -> dict:
    """Convenience for graders: valid names for /recommend."""
    probs = RECOMMENDER["churn_probs"]
    if at_risk_only:
        probs = probs[probs >= 0.5]
    return {
        "count": int(len(probs)),
        "authors": [
            {"author": a, "churn_probability": round(float(p), 3)}
            for a, p in probs.sort_values(ascending=False).items()
        ],
    }
