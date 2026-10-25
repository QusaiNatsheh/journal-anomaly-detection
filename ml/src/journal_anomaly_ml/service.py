"""HTTP scoring service. Run: uvicorn journal_anomaly_ml.service:app

Odoo sends a window of posted journal items (for example the last 12 months). The service computes features,
fits the detector on that window (each company is compared with its own history), and returns the most unusual
entries dated on or after `score_from`, with reasons.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .detectors import METHODS, score
from .explain import reasons
from .features import FEATURES, FeatureConfig, compute_features

MAX_LINES = 200_000
VERSION = "1.0.0"


class LineIn(BaseModel):
    entry_id: int
    date: date
    posted_at: datetime = Field(description="Local posting time of the entry")
    user: str = Field(max_length=100)
    journal: str = Field(max_length=50)
    partner: str | None = Field(None, max_length=200)
    account: str = Field(max_length=50)
    account_type: str = Field("expense", max_length=50)
    debit: float = Field(ge=0)
    credit: float = Field(ge=0)


class ScoreIn(BaseModel):
    lines: list[LineIn] = Field(max_length=MAX_LINES)
    score_from: date
    max_results: int = Field(50, ge=1, le=1000)
    method: str = "ensemble"
    approval_limit: float = Field(10_000.0, gt=0)
    weekend_days: list[int] = Field([4, 5], max_length=7)
    manual_journals: list[str] = Field(["general"], max_length=20)


class EntryOut(BaseModel):
    entry_id: int
    score: float
    percentile: float
    reasons: list[str]
    features: dict[str, float]


class ScoreOut(BaseModel):
    version: str
    method: str
    n_entries: int
    results: list[EntryOut]


app = FastAPI(title="Journal anomaly scoring service", version=VERSION)


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.post("/score", response_model=ScoreOut)
def score_entries(body: ScoreIn):
    if body.method not in METHODS:
        raise HTTPException(status_code=422, detail=f"method must be one of {METHODS}")
    if not body.lines:
        return ScoreOut(version=VERSION, method=body.method, n_entries=0, results=[])
    lines = pd.DataFrame([line.model_dump() for line in body.lines])
    config = FeatureConfig(
        approval_limit=body.approval_limit,
        weekend_days=tuple(body.weekend_days),
        manual_journals=tuple(body.manual_journals),
    )
    feats = compute_features(lines, config)
    scores = score(feats, body.method)
    percentile = pd.Series(scores).rank(pct=True).to_numpy()
    candidates = np.flatnonzero((feats["date"] >= pd.Timestamp(body.score_from)).to_numpy())
    top = candidates[np.argsort(-scores[candidates])][: body.max_results]
    texts = reasons(feats)
    results = [
        EntryOut(
            entry_id=int(feats.at[i, "entry_id"]),
            score=float(scores[i]),
            percentile=float(percentile[i]),
            reasons=texts[i],
            features={f: float(feats.at[i, f]) for f in FEATURES},
        )
        for i in top
    ]
    return ScoreOut(version=VERSION, method=body.method, n_entries=len(feats), results=results)
