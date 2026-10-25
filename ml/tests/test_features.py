from datetime import date, datetime

import pandas as pd
from journal_anomaly_ml.features import FEATURES, compute_features


def _entry(entry_id, day, hour, amount, user="omar", debit_acct="6400", credit_acct="2000", journal="purchase"):
    posted = datetime(day.year, day.month, day.day, hour)
    common = {"entry_id": entry_id, "date": day, "posted_at": posted, "user": user, "journal": journal, "partner": "V1"}
    return [
        {**common, "account": debit_acct, "account_type": "expense", "debit": amount, "credit": 0.0},
        {**common, "account": credit_acct, "account_type": "liability_payable", "debit": 0.0, "credit": amount},
    ]


def test_features_use_only_past_entries():
    rows = []
    for i in range(30):
        rows += _entry(i + 1, date(2026, 1, 4 + i % 20), 10, 100.0 + i)
    rows += _entry(100, date(2026, 2, 5), 2, 50_000.0)  # Thursday, at night, round
    rows += _entry(101, date(2026, 2, 6), 10, 50_000.0)  # Friday (weekend), same as 100: duplicate
    f = compute_features(pd.DataFrame(rows)).set_index("entry_id")
    assert list(f[FEATURES].columns) == FEATURES
    assert f.loc[1, "duplicate_14d"] == 0 and f.loc[1, "amount_z_account"] == 0  # no history yet
    assert f.loc[100, "night"] == 1 and f.loc[100, "round_zeros"] == 4
    assert f.loc[100, "amount_z_account"] > 3
    assert f.loc[101, "weekend"] == 1 and f.loc[101, "duplicate_14d"] == 1
    assert f.loc[100, "duplicate_14d"] == 0  # the later copy does not leak backwards


def test_near_limit_and_manual_sensitive():
    rows = _entry(
        1, date(2026, 3, 2), 10, 9_900.0, user="khaled", debit_acct="3000", credit_acct="1010", journal="general"
    )
    rows[0]["account_type"], rows[1]["account_type"] = "equity", "asset_cash"
    f = compute_features(pd.DataFrame(rows)).iloc[0]
    assert f["near_limit"] == 1 and f["manual"] == 1 and f["sensitive_manual"] == 1
