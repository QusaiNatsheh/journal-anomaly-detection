import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from journal_anomaly_ml.detectors import METHODS, rule_score, score
from journal_anomaly_ml.evaluate import metrics
from journal_anomaly_ml.explain import reasons
from journal_anomaly_ml.features import compute_features
from journal_anomaly_ml.service import app


@pytest.fixture(scope="module")
def feats(lines):
    f = compute_features(lines)
    labels = lines.drop_duplicates("entry_id").set_index("entry_id")["anomaly"]
    f["anomaly"] = f["entry_id"].map(labels)
    return f


@pytest.mark.parametrize("method", METHODS)
def test_detectors_rank_anomalies_above_chance(feats, method):
    m = metrics(score(feats, method), feats["anomaly"])
    base_rate = feats["anomaly"].notna().mean()
    assert m["average_precision"] > 5 * base_rate


def test_ensemble_beats_rules(feats):
    ensemble = metrics(score(feats, "ensemble"), feats["anomaly"])["average_precision"]
    rules = metrics(rule_score(feats), feats["anomaly"])["average_precision"]
    assert ensemble > rules


def test_reasons_name_the_injected_signal(feats):
    texts = reasons(feats)
    night = np.flatnonzero((feats["anomaly"] == "night").to_numpy())
    assert any("night" in t for i in night for t in texts[i])
    dup = np.flatnonzero((feats["anomaly"] == "duplicate").to_numpy())
    assert any("Same amount" in t for i in dup for t in texts[i])


def test_score_endpoint(lines):
    client = TestClient(app)
    payload = {
        "lines": json.loads(lines.drop(columns=["anomaly", "ref"]).to_json(orient="records", date_format="iso")),
        "score_from": "2025-07-01",
        "max_results": 20,
    }
    res = client.post("/score", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 20
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    first_date = pd.Timestamp("2025-07-01")
    dates = lines.drop_duplicates("entry_id").set_index("entry_id")["date"]
    assert all(pd.Timestamp(dates[r["entry_id"]]) >= first_date for r in body["results"])
    labels = lines.drop_duplicates("entry_id").set_index("entry_id")["anomaly"]
    assert sum(labels[r["entry_id"]] is not None for r in body["results"]) >= 10


def test_score_validation():
    client = TestClient(app)
    bad = {"lines": [], "score_from": "2025-01-01", "method": "magic"}
    assert client.post("/score", json=bad).status_code == 422
    assert client.post("/score", json={"lines": [], "score_from": "2025-01-01"}).json()["results"] == []
