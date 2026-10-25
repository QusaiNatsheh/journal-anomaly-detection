from journal_anomaly_ml.generator import ANOMALY_TYPES, Generator, GeneratorConfig


def test_entries_balance_and_amounts_are_positive(lines):
    totals = lines.groupby("entry_id")[["debit", "credit"]].sum()
    assert ((totals["debit"] - totals["credit"]).abs() < 0.002).all()
    assert (lines["debit"] >= 0).all() and (lines["credit"] >= 0).all()


def test_all_anomaly_types_are_injected(lines):
    entries = lines.drop_duplicates("entry_id")
    assert set(entries["anomaly"].dropna()) == set(ANOMALY_TYPES)
    assert 0.005 < entries["anomaly"].notna().mean() < 0.03


def test_generation_is_reproducible():
    a = Generator(GeneratorConfig(seed=9, months=2)).generate()
    b = Generator(GeneratorConfig(seed=9, months=2)).generate()
    assert a.equals(b)
