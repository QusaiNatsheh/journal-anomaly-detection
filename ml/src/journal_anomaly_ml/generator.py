"""Synthetic journal entries for one company, with injected anomalies whose type is known.

Normal activity: sales invoices, vendor bills, customer and vendor payments, payroll, rent, utilities,
depreciation, bank charges and manual adjustments, each posted by the user whose role covers it. The working
week is Sunday to Thursday (Jordan / Saudi Arabia), so Friday and Saturday are the weekend.

Labels are written to a separate column and are used only to evaluate detectors, never to train them.
Usage: python -m journal_anomaly_ml.generator --out data/generated/v1 --seed 42
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ACCOUNTS = {
    "1010": "Bank",
    "1100": "Accounts Receivable",
    "1500": "Fixed Assets",
    "1590": "Accumulated Depreciation",
    "2000": "Accounts Payable",
    "2100": "VAT Payable",
    "2110": "VAT Receivable",
    "2200": "Accrued Expenses",
    "3000": "Owner Equity",
    "4000": "Sales",
    "6100": "Salaries",
    "6200": "Rent",
    "6300": "Utilities",
    "6400": "Travel",
    "6500": "Office Supplies",
    "6600": "Consulting Fees",
    "6700": "Marketing",
    "6800": "Depreciation Expense",
    "6900": "Bank Charges",
}

ACCOUNT_TYPES = {
    "1010": "asset_cash",
    "1100": "asset_receivable",
    "1500": "asset_fixed",
    "1590": "asset_fixed",
    "2000": "liability_payable",
    "2100": "liability_current",
    "2110": "asset_current",
    "2200": "liability_current",
    "3000": "equity",
    "4000": "income",
}  # all other accounts are "expense"

# Expense accounts used on vendor bills: (account, median amount, spread of log amount, share of bills)
BILL_ACCOUNTS = [
    ("6300", 400, 0.4, 0.15),
    ("6400", 600, 0.6, 0.2),
    ("6500", 150, 0.5, 0.25),
    ("6600", 2500, 0.5, 0.15),
    ("6700", 1200, 0.7, 0.25),
]

# user -> journals they normally post in
USERS = {
    "rania": "sales clerk",
    "omar": "payables clerk",
    "lina": "payroll officer",
    "khaled": "accountant",
    "sami": "controller",
}

APPROVAL_LIMIT = 10_000.0
VAT_RATE = 0.16
WEEKEND = (4, 5)  # Friday, Saturday (Monday = 0)

ANOMALY_TYPES = [
    "duplicate",
    "unusual_pair",
    "weekend",
    "night",
    "round_amount",
    "amount_outlier",
    "below_limit",
    "wrong_user",
]


@dataclass
class GeneratorConfig:
    seed: int = 42
    start: str = "2024-10-01"
    months: int = 24
    customers: int = 120
    vendors: int = 60
    sales_per_month: int = 150
    bills_per_month: int = 80
    manual_per_month: int = 12
    anomaly_rate: float = 0.012
    # Normal behaviour that looks odd, so detectors cannot rely on single flags.
    normal_weekend_rate: float = 0.006
    normal_overtime_rate: float = 0.03
    normal_round_manual_rate: float = 0.15


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    return date(d.year + y, m + 1, 1)


class Generator:
    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.start = date.fromisoformat(config.start)
        self.end = _add_months(self.start, config.months) - timedelta(days=1)
        self.entries: list[dict] = []
        self.customer_size = self.rng.lognormal(0, 0.6, config.customers)
        self.vendor_account = [self._pick_bill_account() for _ in range(config.vendors)]

    # ------------------------------------------------------------------ helpers
    def _pick(self, items):
        return items[int(self.rng.integers(len(items)))]

    def _chance(self, p: float) -> bool:
        return bool(self.rng.random() < p)

    def _pick_bill_account(self):
        weights = np.array([a[3] for a in BILL_ACCOUNTS])
        return BILL_ACCOUNTS[int(self.rng.choice(len(BILL_ACCOUNTS), p=weights / weights.sum()))]

    def _workday(self, d: date) -> date:
        while d.weekday() in WEEKEND:
            d += timedelta(days=1)
        return d

    def _posting_time(self, d: date) -> datetime:
        """Posting timestamp: usually working hours, sometimes overtime, rarely a weekend day."""
        if self._chance(self.cfg.normal_weekend_rate):
            day = d + timedelta(days=int((WEEKEND[0] - d.weekday()) % 7))
        else:
            day = self._workday(d)
        if self._chance(self.cfg.normal_overtime_rate):
            hour = float(self.rng.uniform(17, 21))
        else:
            hour = float(np.clip(self.rng.normal(11.5, 2.0), 8, 17))
        return datetime(day.year, day.month, day.day) + timedelta(hours=hour)

    def _round(self, x: float) -> float:
        return float(np.round(x, 3))

    def _add(
        self,
        d: date,
        user: str,
        journal: str,
        lines: list[tuple[str, float, float]],
        partner=None,
        ref="",
        posted_at: datetime | None = None,
        anomaly: str | None = None,
    ) -> dict:
        entry = {
            "entry_id": len(self.entries) + 1,
            "date": d,
            "posted_at": posted_at or self._posting_time(d),
            "user": user,
            "journal": journal,
            "partner": partner,
            "ref": ref,
            "lines": [(a, self._round(dr), self._round(cr)) for a, dr, cr in lines],
            "anomaly": anomaly,
        }
        self.entries.append(entry)
        return entry

    # ------------------------------------------------------------------ normal activity
    def _month_days(self, m_start: date) -> list[date]:
        nxt = _add_months(m_start, 1)
        return [m_start + timedelta(days=i) for i in range((nxt - m_start).days)]

    def normal_month(self, m_start: date):
        days = [d for d in self._month_days(m_start) if d.weekday() not in WEEKEND]
        month_end = days[-1]
        for _ in range(self.rng.poisson(self.cfg.sales_per_month)):
            d = self._pick(days)
            c = int(self.rng.integers(self.cfg.customers))
            net = float(self.rng.lognormal(np.log(900 * self.customer_size[c]), 0.5))
            vat = net * VAT_RATE
            self._add(
                d,
                "rania",
                "sale",
                [("1100", net + vat, 0), ("4000", 0, net), ("2100", 0, vat)],
                partner=f"C{c}",
                ref="invoice",
            )
            pay_day = d + timedelta(days=int(self.rng.integers(10, 60)))
            if pay_day <= self.end:
                self._add(
                    self._workday(pay_day),
                    "rania",
                    "bank",
                    [("1010", net + vat, 0), ("1100", 0, net + vat)],
                    partner=f"C{c}",
                    ref="customer payment",
                )
        for _ in range(self.rng.poisson(self.cfg.bills_per_month)):
            d = self._pick(days)
            v = int(self.rng.integers(self.cfg.vendors))
            account, median, spread, _ = self.vendor_account[v]
            net = float(self.rng.lognormal(np.log(median), spread))
            vat = net * VAT_RATE
            self._add(
                d,
                "omar",
                "purchase",
                [(account, net, 0), ("2110", vat, 0), ("2000", 0, net + vat)],
                partner=f"V{v}",
                ref="vendor bill",
            )
            pay_day = d + timedelta(days=int(self.rng.integers(15, 45)))
            if pay_day <= self.end:
                self._add(
                    self._workday(pay_day),
                    "omar",
                    "bank",
                    [("2000", net + vat, 0), ("1010", 0, net + vat)],
                    partner=f"V{v}",
                    ref="vendor payment",
                )
        months_in = (m_start.year - self.start.year) * 12 + m_start.month - self.start.month
        payroll = 42_000 * (1.03 ** (months_in // 12)) * float(self.rng.uniform(0.98, 1.02))
        payday = [d for d in days if d.day >= 25][0]
        self._add(payday, "lina", "bank", [("6100", payroll, 0), ("1010", 0, payroll)], ref="payroll")
        self._add(days[0], "omar", "bank", [("6200", 3_500, 0), ("1010", 0, 3_500)], partner="LANDLORD", ref="rent")
        for d in self.rng.choice(len(days), size=4, replace=False):
            fee = float(self.rng.uniform(2, 25))
            self._add(days[int(d)], "rania", "bank", [("6900", fee, 0), ("1010", 0, fee)], ref="bank charges")
        self._add(month_end, "khaled", "general", [("6800", 2_150, 0), ("1590", 0, 2_150)], ref="depreciation")
        for _ in range(self.rng.poisson(self.cfg.manual_per_month)):
            # Manual adjustments cluster at month end, as accruals and reclassifications do.
            d = month_end - timedelta(days=int(self.rng.integers(0, 4)))
            d = self._workday(d) if d.month == month_end.month else month_end
            account = self._pick(["6300", "6400", "6600", "6700", "6500"])
            if self._chance(self.cfg.normal_round_manual_rate):
                amount = float(self._pick([500, 1000, 1500, 2000, 2500]))
            else:
                amount = float(self.rng.lognormal(np.log(1200), 0.7))
            user = "khaled" if self._chance(0.85) else "sami"
            self._add(d, user, "general", [(account, amount, 0), ("2200", 0, amount)], ref="accrual")
        if m_start.month in (1, 7):
            amount = float(self.rng.uniform(6_000, 30_000))
            self._add(
                self._pick(days),
                "sami",
                "purchase",
                [("1500", amount, 0), ("2110", amount * VAT_RATE, 0), ("2000", 0, amount * (1 + VAT_RATE))],
                partner="V-EQUIP",
                ref="equipment purchase",
            )

    # ------------------------------------------------------------------ anomalies
    def inject(self, n: int):
        normal = [e for e in self.entries if e["anomaly"] is None]
        kinds = [ANOMALY_TYPES[i % len(ANOMALY_TYPES)] for i in range(n)]
        self.rng.shuffle(kinds)
        for kind in kinds:
            getattr(self, f"_anomaly_{kind}")(normal)

    def _random_day(self) -> date:
        offset = int(self.rng.integers((self.end - self.start).days - 7))
        return self._workday(self.start + timedelta(days=offset))

    def _anomaly_duplicate(self, normal):
        src = self._pick([e for e in normal if e["journal"] == "purchase"])
        d = self._workday(src["date"] + timedelta(days=int(self.rng.integers(1, 6))))
        if d > self.end:
            d = src["date"]
        self._add(
            d,
            src["user"],
            src["journal"],
            [tuple(x) for x in src["lines"]],
            partner=src["partner"],
            ref=src["ref"],
            anomaly="duplicate",
        )

    def _anomaly_unusual_pair(self, normal):
        amount = float(self.rng.lognormal(np.log(3000), 0.6))
        lines = self._pick(
            [
                [("6600", amount, 0), ("1010", 0, amount)],  # consulting paid directly, no bill
                [("1010", amount, 0), ("4000", 0, amount)],  # cash sale with no invoice or VAT
                [("3000", amount, 0), ("1010", 0, amount)],  # money taken out to equity
                [("6400", amount, 0), ("1100", 0, amount)],  # expense booked against a customer balance
            ]
        )
        self._add(
            self._random_day(),
            self._pick(["omar", "khaled", "rania"]),
            "general",
            lines,
            ref="adjustment",
            anomaly="unusual_pair",
        )

    def _copy_with(self, normal, anomaly, posted_at_fn):
        src = self._pick([e for e in normal if e["journal"] in ("purchase", "general", "bank")])
        posted = posted_at_fn(src["date"])
        factor = float(self.rng.uniform(0.8, 1.25))  # a new amount, so the copy is not also a duplicate
        lines = [(a, dr * factor, cr * factor) for a, dr, cr in src["lines"]]
        self._add(
            src["date"],
            src["user"],
            src["journal"],
            lines,
            partner=src["partner"],
            ref=src["ref"],
            posted_at=posted,
            anomaly=anomaly,
        )

    def _anomaly_weekend(self, normal):
        def at(d):
            day = d + timedelta(days=int((self._pick(WEEKEND) - d.weekday()) % 7))
            return datetime(day.year, day.month, day.day) + timedelta(hours=float(self.rng.uniform(9, 20)))

        self._copy_with(normal, "weekend", at)

    def _anomaly_night(self, normal):
        def at(d):
            d = self._workday(d)
            hour = float(self.rng.uniform(23, 28)) % 24
            return datetime(d.year, d.month, d.day) + timedelta(hours=hour)

        self._copy_with(normal, "night", at)

    def _anomaly_round_amount(self, normal):
        amount = float(self._pick([20_000, 25_000, 40_000, 50_000, 75_000]))
        account = self._pick(["6600", "6700", "6400"])
        self._add(
            self._random_day(),
            self._pick(["khaled", "sami"]),
            "general",
            [(account, amount, 0), ("1010", 0, amount)],
            ref="adjustment",
            anomaly="round_amount",
        )

    def _anomaly_amount_outlier(self, normal):
        src = self._pick([e for e in normal if e["journal"] == "purchase" and e["ref"] == "vendor bill"])
        factor = float(self.rng.uniform(8, 20))
        lines = [(a, dr * factor, cr * factor) for a, dr, cr in src["lines"]]
        self._add(
            src["date"],
            src["user"],
            "purchase",
            lines,
            partner=src["partner"],
            ref=src["ref"],
            anomaly="amount_outlier",
        )

    def _anomaly_below_limit(self, normal):
        amount = float(self.rng.uniform(0.95, 0.999) * APPROVAL_LIMIT)
        account = self._pick(["6600", "6700", "6400", "6500"])
        self._add(
            self._random_day(),
            self._pick(["khaled", "omar"]),
            "general",
            [(account, amount, 0), ("2200", 0, amount)],
            ref="accrual",
            anomaly="below_limit",
        )

    def _anomaly_wrong_user(self, normal):
        src = self._pick([e for e in normal if e["journal"] in ("sale", "purchase")])
        user = self._pick([u for u in ("lina", "rania", "omar") if u != src["user"]])
        factor = float(self.rng.uniform(0.7, 1.3))
        self._add(
            src["date"],
            user,
            src["journal"],
            [(a, dr * factor, cr * factor) for a, dr, cr in src["lines"]],
            partner=src["partner"],
            ref=src["ref"],
            anomaly="wrong_user",
        )

    # ------------------------------------------------------------------ assemble
    def generate(self) -> pd.DataFrame:
        for m in range(self.cfg.months):
            self.normal_month(_add_months(self.start, m))
        self.entries = [e for e in self.entries if e["date"] <= self.end]
        self.inject(int(len(self.entries) * self.cfg.anomaly_rate))
        for e in self.entries:  # balance any rounding gap on the last line
            diff = round(sum(dr - cr for _, dr, cr in e["lines"]), 3)
            if diff:
                a, dr, cr = e["lines"][-1]
                e["lines"][-1] = (a, dr, round(cr + diff, 3)) if cr else (a, round(dr - diff, 3), cr)
        entries = sorted(self.entries, key=lambda e: e["posted_at"])
        for i, e in enumerate(entries, start=1):
            e["entry_id"] = i
        return entries_to_frames(entries)


def entries_to_frames(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for e in entries:
        for account, debit, credit in e["lines"]:
            rows.append(
                {
                    "entry_id": e["entry_id"],
                    "date": e["date"],
                    "posted_at": e["posted_at"],
                    "user": e["user"],
                    "journal": e["journal"],
                    "partner": e["partner"],
                    "ref": e["ref"],
                    "account": account,
                    "account_type": ACCOUNT_TYPES.get(account, "expense"),
                    "debit": debit,
                    "credit": credit,
                    "anomaly": e["anomaly"],
                }
            )
    return pd.DataFrame(rows)


def save(lines: pd.DataFrame, out: Path, config: GeneratorConfig):
    out.mkdir(parents=True, exist_ok=True)
    lines.to_parquet(out / "lines.parquet", index=False)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2))


def load(path: Path) -> pd.DataFrame:
    lines = pd.read_parquet(Path(path) / "lines.parquet")
    lines["date"] = pd.to_datetime(lines["date"]).dt.date
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/generated/v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = GeneratorConfig(seed=args.seed)
    lines = Generator(config).generate()
    save(lines, args.out, config)
    entries = lines.drop_duplicates("entry_id")
    print(f"{len(entries)} entries, {len(lines)} lines, {entries['anomaly'].notna().sum()} anomalies")
    print(entries["anomaly"].value_counts().to_string())


if __name__ == "__main__":
    main()
