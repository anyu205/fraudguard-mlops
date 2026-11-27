import numpy as np
import pandas as pd

from src.features.engineer import FEATURE_COLUMNS, build_features


def test_build_features_does_not_mutate_input(valid_training_df: pd.DataFrame) -> None:
    """build_features must leave the caller's DataFrame untouched."""
    original = valid_training_df.copy(deep=True)
    _ = build_features(valid_training_df)

    pd.testing.assert_frame_equal(valid_training_df, original)


def test_output_columns_order_and_index_preservation(
    valid_inference_df: pd.DataFrame, valid_training_df: pd.DataFrame
) -> None:
    """Output matches FEATURE_COLUMNS exactly, drops Time/Class, and preserves index."""
    sliced_train = valid_training_df.iloc[2:]
    out_train = build_features(sliced_train)
    out_inf = build_features(valid_inference_df)

    assert len(FEATURE_COLUMNS) == 33
    assert list(out_train.columns) == list(FEATURE_COLUMNS)
    assert list(out_inf.columns) == list(FEATURE_COLUMNS)
    assert "Time" not in out_train.columns
    assert "Class" not in out_train.columns
    assert out_train.index.equals(sliced_train.index)


def test_boundary_values_and_cyclical_time_encoding(
    valid_inference_df: pd.DataFrame,
) -> None:
    """Zero Amount stays finite, night flag toggles cleanly, and 23:55 is near 00:05."""
    df = valid_inference_df.iloc[:4].copy()
    df["Amount"] = [0.0, 50.0, 100.0, 10.0]
    # 00:05, 02:00 (night), 12:00 (day), 23:55
    df["Time"] = [300.0, 7_200.0, 43_200.0, 86_100.0]

    out = build_features(df)

    assert np.isfinite(out.to_numpy()).all()
    assert out.loc[0, "amount_log"] == 0.0
    assert out.loc[1, "is_night"] == 1
    assert out.loc[2, "is_night"] == 0

    # Distance between 00:05 and 23:55 on the unit circle should be very small (~0.0436).
    d_sin = out.loc[0, "hour_sin"] - out.loc[3, "hour_sin"]
    d_cos = out.loc[0, "hour_cos"] - out.loc[3, "hour_cos"]
    assert np.hypot(d_sin, d_cos) < 0.05


def test_batch_vs_single_row_invariance(valid_training_df: pd.DataFrame) -> None:
    """Features for any single row must match batch output to guarantee zero row leakage."""
    batch_out = build_features(valid_training_df)

    for idx in range(len(valid_training_df)):
        single_out = build_features(valid_training_df.iloc[[idx]])
        pd.testing.assert_frame_equal(batch_out.iloc[[idx]], single_out)
