import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data_validation.schema import validate_inference_data, validate_training_data


def test_valid_training_and_inference_data_pass(
    valid_inference_df: pd.DataFrame, valid_training_df: pd.DataFrame
) -> None:
    """Valid data passes and integer Amount/Time values are coerced to float."""
    int_coerced_df = valid_inference_df.copy()
    int_coerced_df["Amount"] = int_coerced_df["Amount"].astype(int)
    int_coerced_df["Time"] = int_coerced_df["Time"].astype(int)

    inf_out = validate_inference_data(int_coerced_df)
    train_out = validate_training_data(valid_training_df)

    assert inf_out.shape == valid_inference_df.shape
    assert pd.api.types.is_float_dtype(inf_out["Amount"])
    assert pd.api.types.is_float_dtype(inf_out["Time"])
    assert train_out.shape == valid_training_df.shape


@pytest.mark.parametrize(("col", "bad_val"), [("Amount", -1.0), ("Time", -0.5)])
def test_negative_amount_or_time_raises(
    valid_training_df: pd.DataFrame, col: str, bad_val: float
) -> None:
    """Negative Amount or Time violates non-negative checks."""
    bad_df = valid_training_df.copy()
    bad_df.loc[0, col] = bad_val

    with pytest.raises(pa.errors.SchemaErrors):
        validate_training_data(bad_df)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_nan_or_inf_raises(valid_inference_df: pd.DataFrame, bad_val: float) -> None:
    """NaN and infinite float values are rejected by nullable=False and FINITE_CHECK."""
    bad_df = valid_inference_df.copy()
    bad_df.loc[1, "V1"] = bad_val

    with pytest.raises(pa.errors.SchemaErrors):
        validate_inference_data(bad_df)


def test_missing_extra_column_and_invalid_class_raise(
    valid_training_df: pd.DataFrame,
) -> None:
    """Missing columns, unexpected columns (strict=True), and non-binary Class fail."""
    missing_col_df = valid_training_df.drop(columns=["V28"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_training_data(missing_col_df)

    extra_col_df = valid_training_df.assign(extra_col=1.0)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_training_data(extra_col_df)

    bad_class_df = valid_training_df.copy()
    bad_class_df.loc[0, "Class"] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate_training_data(bad_class_df)


def test_lazy_validation_collects_multiple_column_errors(
    valid_training_df: pd.DataFrame,
) -> None:
    """lazy=True reports violations across multiple columns in a single exception."""
    bad_df = valid_training_df.copy()
    bad_df.loc[0, "Amount"] = -10.0
    bad_df.loc[1, "V1"] = np.inf
    bad_df.loc[2, "Class"] = 3

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        validate_training_data(bad_df)

    failed_columns = set(exc_info.value.failure_cases["column"].dropna())
    assert {"Amount", "V1", "Class"}.issubset(failed_columns)
