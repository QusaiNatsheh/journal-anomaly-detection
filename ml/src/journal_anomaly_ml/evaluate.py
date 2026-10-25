"""Compare detectors on synthetic data with injected anomalies. Labels are used here only.

Protocol: features use all history; detectors are fitted on and score the last 12 months (as the service does).
Settings are chosen on a validation dataset (seed 7) and reported on three test datasets (seeds 42, 43, 44).
Usage: python -m journal_anomaly_ml.evaluate
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .detectors import METHODS, rule_score, score
from .features import compute_features
from .generator import Generator, GeneratorConfig, entries_to_frames  # noqa: F401

VALIDATION_SEED = 7
TEST_SEEDS = (42, 43, 44)
SCORING_MONTHS = 12
GRID = {
    "iforest": [{"n_estimators": 300, "max_features": 1.0}, {"n_estimators": 300, "max_features": 0.5}],
    "lof": [{"n_neighbors": 20}, {"n_neighbors": 50}, {"n_neighbors": 100}],
    "autoencoder": [{"hidden": (12, 4, 12)}, {"hidden": (8,)}, {"hidden": (16, 8, 16)}],
}


def dataset(seed: int, cache: Path | None = None) -> pd.DataFrame:
    path = cache / f"features_seed{seed}.parquet" if cache else None
    if path and path.exists():
        return pd.read_parquet(path)
    lines = Generator(GeneratorConfig(seed=seed)).generate()
    feats = compute_features(lines)
    labels = lines.drop_duplicates("entry_id").set_index("entry_id")["anomaly"]
    feats["anomaly"] = feats["entry_id"].map(labels)
    cutoff = feats["date"].max() - pd.DateOffset(months=SCORING_MONTHS)
    feats = feats[feats["date"] > cutoff].reset_index(drop=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        feats.drop(columns=["accounts", "types"]).to_parquet(path)
    return feats


def metrics(s: np.ndarray, labels: pd.Series) -> dict:
    y = labels.notna().to_numpy()
    order = np.argsort(-s)
    n = len(s)
    top2 = order[: int(0.02 * n)]
    out = {
        "precision_at_50": float(y[order[:50]].mean()),
        "precision_at_100": float(y[order[:100]].mean()),
        "recall_at_100": float(y[order[:100]].sum() / y.sum()),
        "recall_at_2pct": float(y[top2].sum() / y.sum()),
        "false_alarms_per_1000_at_2pct": float((~y[top2]).sum() / n * 1000),
        "average_precision": float(average_precision_score(y, s)),
        "n_entries": int(n),
        "n_anomalies": int(y.sum()),
    }
    in_top = set(order[:100])
    out["recall_at_100_by_type"] = {
        t: float(np.mean([i in in_top for i in np.flatnonzero(labels.to_numpy() == t)]))
        for t in sorted(labels.dropna().unique())
    }
    return out


def tune(cache: Path | None) -> dict:
    feats = dataset(VALIDATION_SEED, cache)
    best = {}
    for method, options in GRID.items():
        results = [(metrics(score(feats, method, p), feats["anomaly"])["average_precision"], p) for p in options]
        best[method] = max(results, key=lambda r: r[0])[1]
        print(f"validation {method}: " + ", ".join(f"{p} AP={ap:.3f}" for ap, p in results))
    options = [
        {"members": members, "combine": combine, "member_params": dict(best)}
        for members in (
            ["lof", "autoencoder"],
            ["iforest", "lof", "autoencoder"],
            ["surprise_sum", "lof", "autoencoder"],
        )
        for combine in ("mean", "max")
    ]
    results = [(metrics(score(feats, "ensemble", p), feats["anomaly"])["average_precision"], p) for p in options]
    print("validation ensemble: " + ", ".join(f"{p['members']}/{p['combine']} AP={ap:.3f}" for ap, p in results))
    best["ensemble"] = max(results, key=lambda r: r[0])[1]
    return best


def run(cache: Path | None = None) -> dict:
    best = tune(cache)
    params = dict(best)
    per_seed = {m: [] for m in ["rules", *METHODS]}
    for seed in TEST_SEEDS:
        feats = dataset(seed, cache)
        per_seed["rules"].append(metrics(rule_score(feats), feats["anomaly"]))
        for m in METHODS:
            per_seed[m].append(metrics(score(feats, m, params.get(m), seed=seed), feats["anomaly"]))
    summary = {}
    for m, runs in per_seed.items():
        keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
        summary[m] = {
            k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in keys
        }
        types = runs[0]["recall_at_100_by_type"]
        summary[m]["recall_at_100_by_type"] = {
            t: float(np.mean([r["recall_at_100_by_type"].get(t, 0) for r in runs])) for t in types
        }
        summary[m]["n_entries"] = int(np.mean([r["n_entries"] for r in runs]))
        summary[m]["n_anomalies"] = int(np.mean([r["n_anomalies"] for r in runs]))
    return {"params": params, "summary": summary}


def log_mlflow(result: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("journal-anomaly")
    for method, s in result["summary"].items():
        with mlflow.start_run(run_name=method):
            mlflow.log_param("method", method)
            mlflow.log_param("params", json.dumps(result["params"].get(method, {}), default=str))
            for k, v in s.items():
                if isinstance(v, dict) and "mean" in v:
                    mlflow.log_metric(k, v["mean"])
                    mlflow.log_metric(f"{k}_std", v["std"])
            for t, v in s["recall_at_100_by_type"].items():
                mlflow.log_metric(f"recall_at_100_{t}", v)


def print_report(result: dict) -> None:
    s = result["summary"]
    print(
        f"\nTest datasets {TEST_SEEDS}: ~{s['rules']['n_entries']} entries scored, "
        f"~{s['rules']['n_anomalies']} anomalies each (mean ± std over seeds)"
    )
    cols = [
        "precision_at_50",
        "precision_at_100",
        "recall_at_100",
        "recall_at_2pct",
        "false_alarms_per_1000_at_2pct",
        "average_precision",
    ]
    print(f"{'':12}" + "".join(f"{c[:14]:>16}" for c in cols))
    for m, v in s.items():
        print(f"{m:12}" + "".join(f"{v[c]['mean']:9.3f} ±{v[c]['std']:5.3f}" for c in cols))
    types = list(s["rules"]["recall_at_100_by_type"])
    print("\nRecall in top 100 by anomaly type")
    print(f"{'':12}" + "".join(f"{t[:12]:>13}" for t in types))
    for m, v in s.items():
        print(f"{m:12}" + "".join(f"{v['recall_at_100_by_type'][t]:13.2f}" for t in types))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/generated/features"))
    parser.add_argument("--out", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args(argv)
    result = run(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    log_mlflow(result)
    print_report(result)


if __name__ == "__main__":
    main()
