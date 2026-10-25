"""Export the last months of a synthetic dataset as JSON for loading into Odoo (scripts/load_demo_data.py).

Usage: python -m journal_anomaly_ml.export_demo --out data/generated/demo.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .generator import ACCOUNT_TYPES, ACCOUNTS, Generator, GeneratorConfig


def export(seed: int, months: int) -> dict:
    lines = Generator(GeneratorConfig(seed=seed)).generate()
    cutoff = pd.Timestamp(lines["date"].max()) - pd.DateOffset(months=months)
    lines = lines[pd.to_datetime(lines["date"]) > cutoff]
    entries = []
    for entry_id, g in lines.groupby("entry_id", sort=True):
        first = g.iloc[0]
        entries.append(
            {
                "id": int(entry_id),
                "date": first["date"].isoformat(),
                "posted_at": pd.Timestamp(first["posted_at"]).isoformat(),
                "user": first["user"],
                "journal": first["journal"],
                "partner": first["partner"] if pd.notna(first["partner"]) else None,
                "ref": first["ref"],
                "anomaly": first["anomaly"] if pd.notna(first["anomaly"]) else None,
                "lines": [[r.account, r.debit, r.credit] for r in g.itertuples()],
            }
        )
    return {
        "timezone": "Asia/Riyadh",
        "accounts": [{"code": c, "name": n, "type": ACCOUNT_TYPES.get(c, "expense")} for c, n in ACCOUNTS.items()],
        "users": sorted(lines["user"].unique().tolist()),
        "entries": entries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/generated/demo.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--months", type=int, default=12)
    args = parser.parse_args(argv)
    data = export(args.seed, args.months)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data))
    n_anomalies = sum(e["anomaly"] is not None for e in data["entries"])
    print(f"{len(data['entries'])} entries ({n_anomalies} injected anomalies) -> {args.out}")


if __name__ == "__main__":
    main()
