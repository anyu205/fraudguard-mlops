import numpy as np
import pandas as pd
import pandera.pandas as pa

FINITE_CHECK = pa.Check(np.isfinite, error="must be finite")

v_columns = {
    f"V{i}": pa.Column(float, checks=FINITE_CHECK, nullable=False) for i in range(1, 29)
}

inference_schema = pa.DataFrameSchema(
    columns={
        "Time": pa.Column(
            float, checks=[pa.Check.ge(0.0), FINITE_CHECK], nullable=False
        ),
        "Amount": pa.Column(
            float, checks=[pa.Check.ge(0.0), FINITE_CHECK], nullable=False
        ),
        **v_columns,
    },
    strict=True,
    coerce=True,
)

training_schema = inference_schema.add_columns(
    {
        "Class": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        )
    }
)


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input data features without target column"""
    # lazy = True -> Let Pandera collect all errors of all columns instead stopping at first error column.
    return inference_schema.validate(df, lazy=True)


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate input data features with targetout column"""
    return training_schema.validate(df, lazy=True)
