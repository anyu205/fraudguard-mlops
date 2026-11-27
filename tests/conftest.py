import numpy as np
import pandas as pd
import pytest


def make_inference_df(n_rows: int = 6) -> pd.DataFrame:
    """Create a small valid synthetic inference DataFrame with 30 schema columns."""
    rng = np.random.default_rng(42)
    times = np.array([0.0, 7200.0, 43200.0, 86399.0, 90000.0, 120000.0])[:n_rows]
    if n_rows > len(times):
        times = np.linspace(0.0, 150000.0, n_rows)

    amounts = np.array([0.0, 25.5, 100.0, 250.75, 15.0, 500.0])[:n_rows]
    if n_rows > len(amounts):
        amounts = rng.uniform(0.0, 500.0, size=n_rows).round(2)

    data: dict[str, np.ndarray] = {"Time": times}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0.0, 1.0, size=n_rows)
    data["Amount"] = amounts
    return pd.DataFrame(data)


@pytest.fixture
def valid_inference_df() -> pd.DataFrame:
    """Return a valid synthetic inference DataFrame (no Class column)."""
    return make_inference_df()


@pytest.fixture
def valid_training_df(valid_inference_df: pd.DataFrame) -> pd.DataFrame:
    """Return a valid synthetic training DataFrame with binary Class labels."""
    df = valid_inference_df.copy()
    df["Class"] = [0, 1, 0, 1, 0, 0]
    return df
