import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 60 * 60
NIGHT_WINDOW_SECONDS = 5 * 60 * 60

# Lock exact column order (and drop raw Time / target Class) to avoid train-serve skew.
FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
    "amount_log",
    "hour_sin",
    "hour_cos",
    "is_night",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build stateless fraud-detection features and return columns in FEATURE_COLUMNS order."""
    # Copy without resetting index so caller slices keep alignment with y labels.
    df_cpy = df.copy()

    # Compress heavy right-skewed Amount; log1p(x) = ln(1 + x) stays safe when Amount == 0.
    df_cpy["amount_log"] = np.log1p(df_cpy["Amount"])

    # Cyclical encoding keeps 23:59 adjacent to 00:00; both sin & cos are needed for 1-to-1 mapping.
    hour_of_day = (df_cpy["Time"] % SECONDS_PER_DAY) / 3600.0
    df_cpy["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    df_cpy["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    # Explicit binary flag for late-night transactions (00:00 -> 05:00).
    df_cpy["is_night"] = (
        (df_cpy["Time"] % SECONDS_PER_DAY) < NIGHT_WINDOW_SECONDS
    ).astype(int)

    # Selecting FEATURE_COLUMNS drops raw monotonic Time and optional Class in one step.
    return df_cpy[list(FEATURE_COLUMNS)]
