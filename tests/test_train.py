from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.train import (
    compute_scale_pos_weight,
    evaluate_model,
    load_config,
    run_training,
    split_chronological,
)
from tests.conftest import make_inference_df


def test_load_config_valid_and_errors(tmp_path: Path) -> None:
    """load_config parses YAML dict and rejects missing or non-dict configs."""
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text("data:\n  train_ratio: 0.7\n", encoding="utf-8")
    assert load_config(cfg_path) == {"data": {"train_ratio": 0.7}}

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")

    bad_yaml = tmp_path / "list.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(bad_yaml)


def test_split_chronological_sorts_by_time_and_validates(
    valid_training_df: pd.DataFrame,
) -> None:
    """split_chronological sorts unsorted rows by Time before splitting without leakage."""
    shuffled = valid_training_df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    # Force explicit labels in chronological order so both 60/40 splits contain 0 and 1.
    ordered = shuffled.sort_values("Time").reset_index(drop=True)
    ordered["Class"] = [0, 1, 0, 0, 1, 0]
    unsorted = ordered.iloc[[5, 2, 0, 4, 1, 3]].reset_index(drop=True)

    train_df, val_df = split_chronological(unsorted, train_ratio=0.5)

    assert len(train_df) == 3
    assert len(val_df) == 3
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert train_df["Time"].is_monotonic_increasing
    assert val_df["Time"].is_monotonic_increasing

    with pytest.raises(ValueError):
        split_chronological(unsorted, train_ratio=1.0)
    with pytest.raises(ValueError):
        split_chronological(unsorted.drop(columns=["Time"]), train_ratio=0.5)


def test_compute_scale_pos_weight() -> None:
    """compute_scale_pos_weight computes n_neg / n_pos and guards against zero positives."""
    y = pd.Series([0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)

    with pytest.raises(ValueError):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))


def test_evaluate_model_metrics_and_zero_division() -> None:
    """evaluate_model returns expected keys and handles zero positive predictions."""
    y_true = pd.Series([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])  # All below threshold 0.5

    metrics = evaluate_model(y_true, y_prob, threshold=0.5)

    assert set(metrics.keys()) == {"pr_auc", "roc_auc", "precision", "recall", "f1"}
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_run_training_end_to_end_and_gate(tmp_path: Path) -> None:
    """run_training saves artifact when PR-AUC passes and raises when gate fails."""
    n_rows = 80
    df = make_inference_df(n_rows=n_rows)
    # Create clean separable fraud signal on V1 so synthetic training hits PR-AUC > 0.8.
    labels = np.zeros(n_rows, dtype=int)
    fraud_idx = [5, 15, 25, 35, 45, 55, 62, 68, 74, 78]
    labels[fraud_idx] = 1
    df.loc[fraud_idx, "V1"] = 6.0
    df.loc[df.index.difference(fraud_idx), "V1"] = -1.0
    df["Class"] = labels

    parquet_path = tmp_path / "synthetic.parquet"
    df.to_parquet(parquet_path, index=False)

    model_path = tmp_path / "artifacts" / "model.joblib"
    cfg_path = tmp_path / "config.yaml"
    cfg = {
        "data": {
            "input_path": str(parquet_path),
            "output_path": str(model_path),
            "train_ratio": 0.7,
        },
        "model": {
            "type": "lightgbm",
            "max_scale_pos_weight": 2.0,
            "log_every": 5,
            "params": {
                "random_state": 42,
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "num_leaves": 7,
                "min_child_samples": 2,
                "verbose": -1,
            },
        },
        "evaluation": {"min_pr_auc": 0.8, "threshold": 0.5},
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    metrics = run_training(cfg_path)
    assert metrics["pr_auc"] >= 0.8
    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert loaded["threshold"] == 0.5
    assert len(loaded["feature_columns"]) == 33

    # Verify Evaluation Gate rejects models below min_pr_auc.
    cfg["evaluation"]["min_pr_auc"] = 1.01
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="PR_AUC is lower"):
        run_training(cfg_path)
