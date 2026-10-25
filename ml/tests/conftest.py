import pandas as pd
import pytest
from journal_anomaly_ml.generator import Generator, GeneratorConfig


@pytest.fixture(scope="session")
def lines() -> pd.DataFrame:
    return Generator(GeneratorConfig(seed=5, months=12, sales_per_month=80, bills_per_month=40)).generate()
