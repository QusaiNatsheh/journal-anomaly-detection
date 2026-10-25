"""Unsupervised detectors. Each one is fitted on the entries it scores; no labels are used.

Every detector returns a score where higher means more unusual. The ensemble averages rank percentiles.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from .features import FEATURES

DEFAULT_PARAMS = {
    "surprise_sum": {},
    "iforest": {"n_estimators": 300, "max_features": 1.0},
    "lof": {"n_neighbors": 50},
    "autoencoder": {"hidden": (16, 8, 16), "max_iter": 400},
    # Chosen on the validation dataset by `evaluate` (see results/evaluation.json).
    "ensemble": {"members": ["surprise_sum", "lof", "autoencoder"], "combine": "max"},
}
METHODS = list(DEFAULT_PARAMS)


# Which tail of each feature is unusual. Everything not listed: high values are unusual.
TWO_SIDED = {"log_amount", "amount_z_account", "amount_z_pair", "amount_z_user"}
LOW_IS_UNUSUAL = {"days_to_month_end"}


def surprise(features: pd.DataFrame) -> np.ndarray:
    """Turn each feature into -log(tail probability) within the batch.

    A flag present on 1% of entries becomes about 4.6, so rare yes/no signals weigh as much as extreme amounts.
    """
    cols = []
    n = len(features)
    for name in FEATURES:
        x = features[name].to_numpy(float)
        if name in TWO_SIDED:
            x = np.abs(x - np.median(x))
        elif name in LOW_IS_UNUSUAL:
            x = -x
        # share of entries with a value at least as extreme (ties count), via ranks
        tail = pd.Series(x).rank(method="max", ascending=False).to_numpy() / n
        cols.append(-np.log(tail))
    return np.column_stack(cols)


def _matrix(features: pd.DataFrame) -> np.ndarray:
    return RobustScaler().fit_transform(surprise(features))


def _iforest(x, n_estimators, max_features, seed):
    model = IsolationForest(n_estimators=n_estimators, max_features=max_features, random_state=seed)
    return -model.fit(x).score_samples(x)


def _lof(x, n_neighbors, seed):
    model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(x) - 1))
    model.fit(x)
    return -model.negative_outlier_factor_


def _autoencoder(x, hidden, max_iter, seed):
    model = MLPRegressor(
        hidden_layer_sizes=tuple(hidden), max_iter=max_iter, early_stopping=True, random_state=seed, alpha=1e-3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, x)
    return ((model.predict(x) - x) ** 2).mean(axis=1)


def _rank(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(pct=True).to_numpy()


def score(features: pd.DataFrame, method: str = "ensemble", params: dict | None = None, seed: int = 0) -> np.ndarray:
    if len(features) < 10:
        return np.zeros(len(features))
    p = {**DEFAULT_PARAMS[method], **(params or {})}
    x = _matrix(features)
    if method == "surprise_sum":
        return surprise(features).sum(axis=1)
    if method == "iforest":
        return _iforest(x, p["n_estimators"], p["max_features"], seed)
    if method == "lof":
        return _lof(x, p["n_neighbors"], seed)
    if method == "autoencoder":
        return _autoencoder(x, p["hidden"], p["max_iter"], seed)
    if method == "ensemble":
        member_params = p.get("member_params", {})
        ranks = [_rank(score(features, m, member_params.get(m), seed)) for m in p["members"]]
        if p.get("combine", "mean") == "max":
            # An entry ranked very high by any member stays high; the mean breaks ties.
            return np.max(ranks, axis=0) + 0.01 * np.mean(ranks, axis=0)
        return np.mean(ranks, axis=0)
    raise ValueError(f"Unknown method {method}")


def rule_score(features: pd.DataFrame) -> np.ndarray:
    """Baseline: number of red-flag rules an auditor might write by hand."""
    f = features
    flags = [
        f["weekend"] == 1,
        f["night"] == 1,
        f["round_zeros"] >= 3,
        f["near_limit"] == 1,
        f["duplicate_14d"] >= 1,
        f["amount_z_account"].abs() > 3,
        f["sensitive_manual"] == 1,
    ]
    # Small tie-breaker so entries with equal flag counts are ordered by amount.
    return np.sum(flags, axis=0) + 0.001 * f["log_amount"].to_numpy()
