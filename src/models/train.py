import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from lightgbm.callback import CallbackEnv
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_validation.schema import validate_training_data
from src.features.engineer import FEATURE_COLUMNS, build_features

CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load training YAML configuration from disk.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config is not a dict, got {type(cfg).__name__}")
    return cfg


def split_chronological(
    df: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort data by Time and split chronologically into train and validation sets.

    Args:
        df: Validated training DataFrame containing Time and Class columns.
        train_ratio: Fraction of earliest rows assigned to the training split (0 < r < 1).

    Returns:
        Tuple of (train_df, val_df) preserving chronological order without shuffling.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(
            f"Train ratio must be in range (0, 1), current ratio is {train_ratio}"
        )
    dataset = df.copy()

    missing_cols = {"Time", "Class"} - set(dataset.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols, key=str)}!")

    dataset = dataset.sort_values("Time", kind="mergesort").reset_index(drop=True)

    val_idx = int(len(dataset) * train_ratio)
    train_df = dataset.iloc[:val_idx].copy().reset_index(drop=True)
    val_df = dataset.iloc[val_idx:].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Train and validation splits must both be non-empty")

    if train_df["Class"].nunique() < 2 or val_df["Class"].nunique() < 2:
        raise ValueError(
            "Train and validation splits must both contain classes 0 and 1"
        )

    return train_df, val_df


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute class imbalance ratio (n_neg / n_pos) from training labels only.

    Args:
        y_train: Binary target Series from the training split.

    Returns:
        Positive class weight ratio (number of negatives / number of positives).
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if not n_pos:
        raise ValueError("There is no positive sample in output label.")
    weight = float(n_neg / n_pos)
    return weight


def evaluate_model(
    y_true: pd.Series, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Evaluate fraud detection probabilities on imbalanced validation labels.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Predicted fraud probabilities for the positive class.
        threshold: Probability cutoff used to convert probabilities into binary labels.

    Returns:
        Dictionary with pr_auc, roc_auc, precision, recall, and f1 scores.
    """

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _build_milestone_table(
    rows: list[dict[str, float]],
    live_row: dict[str, float] | None = None,
) -> Table:
    """Build a Rich table displaying checkpointed and live LightGBM metrics."""
    table = Table(title="Training Progress", expand=False)
    table.add_column("Iter", justify="right", style="cyan")
    table.add_column("LR", justify="right", style="magenta")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Train PR-AUC", justify="right", style="green")
    table.add_column("Val PR-AUC", justify="right", style="bold green")
    table.add_column("Elapsed (s)", justify="right", style="dim")
    table.add_column("ETA (s)", justify="right", style="yellow")

    display_rows = list(rows)
    if live_row is not None and (not rows or rows[-1]["iteration"] != live_row["iteration"]):
        display_rows.append(live_row)

    for r in display_rows:
        table.add_row(
            f"{int(r['iteration'])}",
            f"{r['lr']:.4f}",
            f"{r.get('train_binary_logloss', float('nan')):.4f}",
            f"{r.get('val_binary_logloss', float('nan')):.4f}",
            f"{r.get('train_average_precision', float('nan')):.4f}",
            f"{r.get('val_average_precision', float('nan')):.4f}",
            f"{r['elapsed_sec']:.2f}",
            f"{r['eta_sec']:.2f}",
        )
    return table


@contextmanager
def training_progress_observer(
    total_iterations: int,
    log_every: int = 25,
    history: list[dict[str, float]] | None = None,
    console: Console | None = None,
) -> Iterator[Callable[[CallbackEnv], None]]:
    """Provide a live Rich progress bar and milestone table callback for LightGBM.

    Args:
        total_iterations: Expected total boosting rounds (n_estimators).
        log_every: Record a permanent table row every N rounds (plus first and last).
        history: Optional list populated with per-iteration metric records.
        console: Optional Rich Console instance for rendering.

    Yields:
        LightGBM-compatible callback function updating the live UI after each round.
    """
    if log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {log_every}")

    active_console = console or Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(),
        console=active_console,
    )
    task_id = progress.add_task("Training LightGBM", total=total_iterations)
    milestone_rows: list[dict[str, float]] = []
    start_time = time.perf_counter()

    with Live(
        Group(progress, _build_milestone_table(milestone_rows)),
        console=active_console,
        refresh_per_second=10,
    ) as live:

        def _callback(env: CallbackEnv) -> None:
            step = env.iteration + 1
            total_steps = max(env.end_iteration - env.begin_iteration, 1)
            completed = max(step - env.begin_iteration, 1)
            elapsed = time.perf_counter() - start_time
            eta = (elapsed / completed) * max(total_steps - completed, 0)
            lr = float(env.params.get("learning_rate", 0.0))

            metrics_map = {
                f"{dataset}_{metric}": float(value)
                for dataset, metric, value, _ in (env.evaluation_result_list or [])
            }
            record: dict[str, float] = {
                "iteration": float(step),
                "lr": lr,
                "elapsed_sec": elapsed,
                "eta_sec": eta,
                **metrics_map,
            }
            if history is not None:
                history.append(record)

            is_boundary = step == env.begin_iteration + 1 or step == env.end_iteration
            if is_boundary or step % log_every == 0:
                milestone_rows.append(record)

            progress.update(
                task_id,
                completed=completed,
                total=total_steps,
            )
            live.update(Group(progress, _build_milestone_table(milestone_rows, record)))

        _callback.order = 10  # type: ignore[attr-defined]
        _callback.before_iteration = False  # type: ignore[attr-defined]
        yield _callback


def _render_evaluation_panel(
    console: Console,
    metrics: dict[str, float],
    min_pr_auc: float,
    output_path: Path,
) -> None:
    """Render final validation metrics and Evaluation Gate status as a Rich table."""
    passed = metrics["pr_auc"] >= min_pr_auc
    status = "[bold green]PASSED[/]" if passed else "[bold red]FAILED[/]"

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, val in metrics.items():
        style = "bold green" if key == "pr_auc" and passed else ""
        table.add_row(key, f"{val:.4f}", style=style)

    summary = Group(
        table,
        f"Evaluation Gate (PR-AUC >= {min_pr_auc:.2f}): {status}",
        f"Artifact Path: [dim]{output_path}[/]",
    )
    console.print(Panel(summary, title="Validation Summary", expand=False))


def run_training(config_path: Path = CONFIG_PATH) -> dict[str, float]:
    """Run validation, chronological split, LightGBM training, evaluation, and saving.

    Args:
        config_path: Path to the training YAML configuration file.

    Returns:
        Dictionary of validation evaluation metrics.
    """
    console = Console()
    cfg = load_config(config_path)

    input_path = Path(cfg["data"]["input_path"])
    output_path = Path(cfg["data"]["output_path"])
    train_ratio = float(cfg["data"]["train_ratio"])

    df = pd.read_parquet(input_path)
    df = validate_training_data(df)
    train_df, val_df = split_chronological(df, train_ratio)
    X_train = build_features(train_df)
    y_train = train_df["Class"]
    X_val = build_features(val_df)
    y_val = val_df["Class"]

    model_cfg = cfg["model"]
    if model_cfg.get("type") != "lightgbm":
        raise ValueError(
            f"Unsupported model type: {model_cfg.get('type')}; expected 'lightgbm'"
        )
    params = dict(model_cfg["params"])
    imbalance_ratio = compute_scale_pos_weight(y_train)
    max_weight = float(model_cfg.get("max_scale_pos_weight", 3.0))
    effective_weight = min(imbalance_ratio, max_weight)

    console.print(
        Panel(
            f"Train rows: [bold]{len(X_train):,}[/] (fraud={int((y_train == 1).sum())}) | "
            f"Val rows: [bold]{len(X_val):,}[/] (fraud={int((y_val == 1).sum())})\n"
            f"Imbalance ratio: [yellow]{imbalance_ratio:.2f}[/] | "
            f"Effective scale_pos_weight (cap={max_weight}): [bold cyan]{effective_weight:.2f}[/]",
            title="FraudGuard Baseline Training",
            expand=False,
        )
    )

    log_every = int(model_cfg.get("log_every", 25))
    n_estimators = int(params.get("n_estimators", 100))
    history: list[dict[str, float]] = []

    model = LGBMClassifier(**params, scale_pos_weight=effective_weight)
    with training_progress_observer(
        total_iterations=n_estimators,
        log_every=log_every,
        history=history,
        console=console,
    ) as observer:
        model.fit(
            X_train,
            y_train,
            eval_X=(X_train, X_val),
            eval_y=(y_train, y_val),
            eval_names=["train", "val"],
            eval_metric=["binary_logloss", "average_precision"],
            callbacks=[observer],
        )

    console.print()
    y_prob = model.predict_proba(X_val)[:, 1]
    threshold = float(cfg["evaluation"]["threshold"])

    metrics = evaluate_model(y_val, y_prob, threshold)
    min_pr_auc = float(cfg["evaluation"]["min_pr_auc"])
    _render_evaluation_panel(console, metrics, min_pr_auc, output_path)

    if metrics["pr_auc"] < min_pr_auc:
        raise ValueError(
            f"PR_AUC is lower than minimum threshold, current is {metrics['pr_auc']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": threshold,
        "metrics": metrics,
    }

    joblib.dump(artifact, output_path)

    return metrics


if __name__ == "__main__":
    run_training()
