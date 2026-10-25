"""Plain-language reasons for a flagged entry, from how far each feature is from the batch's normal range."""

import numpy as np
import pandas as pd

from .features import FEATURES

_TEMPLATES = {
    "amount_z_account": lambda r: (
        f"Amount is unusually {'high' if r['amount_z_account'] > 0 else 'low'} for account {r['main_account']}"
    ),
    "amount_z_pair": lambda r: f"Amount is unusual for this account combination ({r['signature']})",
    "amount_z_user": lambda r: f"Amount is unusual for user {r['user']}",
    # Only multiples of 1,000 read as "round" to a person; 2,150 is unusual for the batch but not a round amount.
    "round_zeros": lambda r: f"Round amount ({r['amount']:,.0f})" if r["round_zeros"] >= 3 else None,
    "near_limit": lambda r: f"Amount {r['amount']:,.2f} is just below the approval limit",
    "weekend": lambda r: f"Posted on a weekend day ({r['posted_at']:%A})",
    "night": lambda r: f"Posted at night ({r['posted_at']:%H:%M})",
    "hour_z_user": lambda r: f"Posting time {r['posted_at']:%H:%M} is unusual for user {r['user']}",
    "pair_rarity": lambda r: f"Rare account combination ({r['signature']})",
    "user_account_rarity": lambda r: f"User {r['user']} rarely posts to these accounts",
    "user_journal_rarity": lambda r: f"User {r['user']} rarely posts in the {r['journal']} journal",
    "sensitive_manual": lambda r: "Manual entry touching cash, equity, revenue, receivables or fixed assets",
    "duplicate_14d": lambda r: "Same amount, accounts and partner as another entry in the last 14 days",
    "posting_delay_days": lambda r: f"Posted {int(r['posting_delay_days'])} days after its accounting date",
    "days_to_month_end": lambda r: "Posted close to month end",
}


def reasons(features: pd.DataFrame, limit: int = 3) -> list[list[str]]:
    """For each entry, the features furthest from the batch median (robust z-score), as sentences."""
    x = features[FEATURES].to_numpy(float)
    median = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    spread = np.maximum(q75 - q25, np.maximum(np.std(x, axis=0), 1e-6))
    z = np.abs(x - median) / spread
    out = []
    records = features.to_dict("records")
    for i, row in enumerate(records):
        texts = []
        for j in np.argsort(-z[i]):
            if z[i, j] < 1.5 or len(texts) >= limit:
                break
            name = FEATURES[j]
            if name == "days_to_month_end" and row[name] > 3:
                continue
            template = _TEMPLATES.get(name)
            text = template(row) if template else None
            if text:
                texts.append(text)
        out.append(texts)
    return out
