"""Entry-level features, computed in posting order from earlier entries only.

The same code runs in evaluation and in the scoring service. Input is one row per journal item with columns:
entry_id, date, posted_at, user, journal, partner, account, account_type, debit, credit.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FEATURES = [
    "log_amount",
    "amount_z_account",
    "amount_z_pair",
    "amount_z_user",
    "round_zeros",
    "near_limit",
    "weekend",
    "night",
    "hour_z_user",
    "days_to_month_end",
    "pair_rarity",
    "user_account_rarity",
    "user_journal_rarity",
    "manual",
    "sensitive_manual",
    "duplicate_14d",
    "n_lines",
    "posting_delay_days",
    "benford_mad_account",
]

BENFORD = np.log10(1 + 1 / np.arange(1, 10))
MIN_HISTORY = 5
SENSITIVE_TYPES = {"asset_cash", "equity", "asset_fixed", "income", "asset_receivable"}


@dataclass
class FeatureConfig:
    approval_limit: float = 10_000.0
    weekend_days: tuple[int, ...] = (4, 5)  # Friday, Saturday
    manual_journals: tuple[str, ...] = ("general",)


class _Running:
    """Running mean and variance (Welford)."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def z(self, x: float) -> float:
        if self.n < MIN_HISTORY:
            return 0.0
        std = math.sqrt(self.m2 / (self.n - 1))
        return (x - self.mean) / max(std, 0.1)


@dataclass
class _History:
    total: int = 0
    pair: dict = field(default_factory=lambda: defaultdict(int))
    acct_amount: dict = field(default_factory=lambda: defaultdict(_Running))
    pair_amount: dict = field(default_factory=lambda: defaultdict(_Running))
    user_amount: dict = field(default_factory=lambda: defaultdict(_Running))
    user_hour: dict = field(default_factory=lambda: defaultdict(_Running))
    user_acct: dict = field(default_factory=lambda: defaultdict(int))
    user_total: dict = field(default_factory=lambda: defaultdict(int))
    user_journal: dict = field(default_factory=lambda: defaultdict(int))
    recent: dict = field(default_factory=lambda: defaultdict(list))
    digits: dict = field(default_factory=lambda: defaultdict(lambda: np.zeros(9)))


def _round_zeros(amount: float) -> int:
    if abs(amount - round(amount)) > 1e-6 or amount < 1:
        return 0
    n, zeros = int(round(amount)), 0
    while n % 10 == 0 and zeros < 6:
        n //= 10
        zeros += 1
    return zeros


def _first_digit(amount: float) -> int | None:
    if amount <= 0:
        return None
    return int(f"{amount:e}"[0])


def _rarity(count: int, total: int, k: int = 50) -> float:
    """Surprise of an event seen `count` times out of `total`, smoothed: higher means rarer."""
    return -math.log((count + 1) / (total + k))


def entry_table(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per entry with amount, signature and main account."""
    lines = lines.copy()
    lines["posted_at"] = pd.to_datetime(lines["posted_at"])
    lines["partner"] = lines["partner"].fillna("")
    grouped = lines.groupby("entry_id", sort=False)
    entries = grouped.agg(
        date=("date", "first"),
        posted_at=("posted_at", "first"),
        user=("user", "first"),
        journal=("journal", "first"),
        partner=("partner", "first"),
        amount=("debit", "sum"),
        line_count=("account", "size"),
    )
    entries["date"] = pd.to_datetime(entries["date"])
    # main account: the account of the largest debit line (first line when there is no debit)
    by_debit = lines.sort_values("debit", ascending=False, kind="stable").groupby("entry_id", sort=False)
    entries["main_account"] = by_debit["account"].first()
    entries["accounts"] = grouped["account"].agg(lambda s: tuple(sorted(set(s))))
    entries["types"] = grouped["account_type"].agg(lambda s: tuple(sorted(set(s))))
    debit_accts = lines[lines["debit"] > 0].groupby("entry_id")["account"].agg(lambda s: "+".join(sorted(set(s))))
    credit_accts = lines[lines["credit"] > 0].groupby("entry_id")["account"].agg(lambda s: "+".join(sorted(set(s))))
    entries["signature"] = (
        "D:" + debit_accts.reindex(entries.index).fillna("") + "|C:" + credit_accts.reindex(entries.index).fillna("")
    )
    entries = entries.reset_index()
    cols = [
        "entry_id",
        "date",
        "posted_at",
        "user",
        "journal",
        "partner",
        "amount",
        "main_account",
        "accounts",
        "types",
        "signature",
        "line_count",
    ]
    return entries[cols].sort_values(["posted_at", "entry_id"]).reset_index(drop=True)


def compute_features(lines: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Return the entry table with one column per feature."""
    cfg = config or FeatureConfig()
    entries = entry_table(lines)
    h = _History()
    feats = []
    for e in entries.itertuples(index=False):
        amount = e.amount
        log_amount = math.log1p(amount)
        hour = e.posted_at.hour + e.posted_at.minute / 60
        month_end = (e.date + pd.offsets.MonthEnd(0)).date()
        key_dup = (round(amount, 2), e.signature, e.partner)
        # 14 days: re-entered bills show up within days, while monthly recurring entries (rent, depreciation)
        # are 28-31 days apart and must not count as duplicates.
        recent = [d for d in h.recent[key_dup] if (e.date - d).days <= 14]
        user_total = h.user_total[e.user]
        digits = h.digits[e.main_account]
        benford_mad = float(np.abs(digits / digits.sum() - BENFORD).mean()) if digits.sum() >= 50 else 0.0
        manual = e.journal in cfg.manual_journals
        f = {
            "log_amount": log_amount,
            "amount_z_account": h.acct_amount[e.main_account].z(log_amount),
            "amount_z_pair": h.pair_amount[e.signature].z(log_amount),
            "amount_z_user": h.user_amount[e.user].z(log_amount),
            "round_zeros": float(_round_zeros(amount)),
            "near_limit": float(0.95 * cfg.approval_limit <= amount < cfg.approval_limit),
            "weekend": float(e.posted_at.weekday() in cfg.weekend_days),
            "night": float(hour < 6 or hour >= 22),
            "hour_z_user": abs(h.user_hour[e.user].z(hour)),
            "days_to_month_end": float((month_end - e.date.date()).days),
            "pair_rarity": _rarity(h.pair[e.signature], h.total),
            "user_account_rarity": max(_rarity(h.user_acct[(e.user, a)], user_total) for a in e.accounts),
            "user_journal_rarity": _rarity(h.user_journal[(e.user, e.journal)], user_total),
            "manual": float(manual),
            "sensitive_manual": float(manual and bool(SENSITIVE_TYPES & set(e.types))),
            "duplicate_14d": float(len(recent)),
            "n_lines": float(e.line_count),
            "posting_delay_days": float(max((e.posted_at.normalize() - e.date).days, 0)),
            "benford_mad_account": benford_mad,
        }
        feats.append(f)
        # update history with this entry
        h.total += 1
        h.pair[e.signature] += 1
        h.acct_amount[e.main_account].add(log_amount)
        h.pair_amount[e.signature].add(log_amount)
        h.user_amount[e.user].add(log_amount)
        h.user_hour[e.user].add(hour)
        for a in e.accounts:
            h.user_acct[(e.user, a)] += 1
        h.user_total[e.user] += 1
        h.user_journal[(e.user, e.journal)] += 1
        h.recent[key_dup] = [*recent, e.date]
        d = _first_digit(amount)
        if d:
            digits[d - 1] += 1
    return pd.concat([entries, pd.DataFrame(feats, columns=FEATURES)], axis=1)
