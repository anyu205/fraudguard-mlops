from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"Time", "Amount", "Class"}
SECONDS_PER_DAY = 86_400
NIGHT_WINDOW_SECONDS = 5 * 3_600  # 00:00 -> 05:00


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare time-ordered fraud datasets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/creditcard.csv"),
        help="Path to raw creditcard.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to write processed parquet files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible drift injection",
    )
    return parser.parse_args()


def load_and_sort_data(input_path: Path) -> pd.DataFrame:
    """Load raw CSV, validate required columns, and sort ascending by Time."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")
    else:
        # print(type(df["Class"]))
        df["Class"] = df["Class"].astype(int)
        df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
        # print("Data is sorted by Time value!")
        return df


def split_by_time(
    df: pd.DataFrame,
    v1_ratio: float = 0.4,
    v2_ratio: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split sorted dataset chronologically into v1, v2, and v3."""
    if not (v1_ratio > 0 and v2_ratio > 0 and v1_ratio + v2_ratio < 1):
        raise ValueError("Value of v1 or v2 is not correct!")

    idx1 = int(len(df) * v1_ratio)
    idx2 = int(len(df) * (v1_ratio + v2_ratio))

    # .copy() isolates each slice in memory to avoid SettingWithCopyWarning and side effects.
    # .reset_index(drop=True) renumbers rows 0..N-1 without adding an extra "index" column to Parquet.
    v1 = df.iloc[:idx1].copy().reset_index(drop=True)
    v2 = df.iloc[idx1:idx2].copy().reset_index(drop=True)
    v3 = df.iloc[idx2:].copy().reset_index(drop=True)

    return v1, v2, v3


def inject_drift(
    df: pd.DataFrame,
    amount_scale: float = 2.5,
    noise_std: float = 15.0,
    night_shift_ratio: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Create controlled Amount and night-time distribution drift."""
    df2 = df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(df2))
    df2["Amount"] = np.clip(
        df2["Amount"] * amount_scale + noise, a_min=0.0, a_max=None
    ).round(2)
    if not (0.0 <= night_shift_ratio <= 1.0):
        raise ValueError("night_shift_ratio must be between 0.0 and 1.0")
    n_shift = int(len(df2) * night_shift_ratio)
    selected_idx = rng.choice(len(df2), size=n_shift, replace=False)

    df2.loc[selected_idx, "Time"] = (
        df2.loc[selected_idx, "Time"] // SECONDS_PER_DAY
    ) * SECONDS_PER_DAY + rng.integers(0, NIGHT_WINDOW_SECONDS, size=n_shift)
    df2 = df2.sort_values("Time", kind="mergesort").reset_index(drop=True)

    return df2


def summarize_split(name: str, df: pd.DataFrame) -> dict[str, object]:
    """Compute summary statistics for one dataset split."""
    return {
        "split": name,
        "rows": len(df),
        "fraud_rate_pct": round(float((df["Class"] == 1).mean() * 100), 3),
        "night_rate_pct": round(
            float(((df["Time"] % SECONDS_PER_DAY) < NIGHT_WINDOW_SECONDS).mean() * 100),
            3,
        ),
        "amount_mean": round(float((df["Amount"]).mean()), 2),
        "amount_median": round(float((df["Amount"]).median()), 2),
        "time_min": (df["Time"]).min(),
        "time_max": (df["Time"]).max(),
    }


def save_and_report(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save splits to Parquet files and print a summary table."""
    summary = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        output_path = output_dir / f"{name}.parquet"
        df.to_parquet(name, index=False)
        summary.append(summarize_split(output_path, df))
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    df = load_and_sort_data(args.input)
    v1, v2, v3_raw = split_by_time(df)
    v3_drift = inject_drift(v3_raw, seed=args.seed)

    splits = {
        "transactions_v1": v1,
        "transactions_v2": v2,
        "transactions_drift_v3": v3_drift,
    }
    save_and_report(splits, args.output_dir)


if __name__ == "__main__":
    main()
