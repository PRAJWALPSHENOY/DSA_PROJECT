from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from game_solver.data.dataset_io import feature_columns, load_dataframe


def run_transfer_experiment(
    dataset_dir: str | Path,
    output_dir: str | Path,
    seed: int = 7,
    max_epochs: int = 40,
) -> dict:
    dataset_root = Path(dataset_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ttt_df = load_dataframe(dataset_root / "tic_tac_toe_states.csv")
    c4_df = load_dataframe(dataset_root / "connect_four_states.csv")

    x_ttt = ttt_df[feature_columns(ttt_df)].to_numpy(dtype=np.float32)
    y_ttt = ttt_df["label"].to_numpy(dtype=np.float32)

    x_c4 = c4_df[feature_columns(c4_df)].to_numpy(dtype=np.float32)
    y_c4 = c4_df["label"].to_numpy(dtype=np.float32)

    target_dim = int(max(x_ttt.shape[1], x_c4.shape[1]))
    x_ttt_padded = _pad_features(x_ttt, target_dim)
    x_c4_padded = _pad_features(x_c4, target_dim)

    x_train, x_val, y_train, y_val = train_test_split(
        x_c4_padded,
        y_c4,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    source_model = _build_partial_model(random_state=seed)
    source_model.partial_fit(x_ttt_padded, y_ttt)

    transfer_model = _build_partial_model(random_state=seed)
    transfer_model.partial_fit(x_train[: min(8, len(x_train))], y_train[: min(8, len(y_train))])
    transfer_model.coefs_ = [coef.copy() for coef in source_model.coefs_]
    transfer_model.intercepts_ = [bias.copy() for bias in source_model.intercepts_]

    scratch_model = _build_partial_model(random_state=seed)
    scratch_model.partial_fit(x_train[: min(8, len(x_train))], y_train[: min(8, len(y_train))])

    transfer_curve = []
    scratch_curve = []

    for _epoch in range(max_epochs):
        transfer_model.partial_fit(x_train, y_train)
        scratch_model.partial_fit(x_train, y_train)

        transfer_mse = mean_squared_error(y_val, transfer_model.predict(x_val))
        scratch_mse = mean_squared_error(y_val, scratch_model.predict(x_val))

        transfer_curve.append(float(transfer_mse))
        scratch_curve.append(float(scratch_mse))

    threshold = min(transfer_curve[-1], scratch_curve[-1]) * 1.05

    transfer_epoch = _first_below(transfer_curve, threshold)
    scratch_epoch = _first_below(scratch_curve, threshold)

    metrics = {
        "target_threshold_mse": float(threshold),
        "transfer_best_mse": float(min(transfer_curve)),
        "scratch_best_mse": float(min(scratch_curve)),
        "epochs_to_threshold_transfer": transfer_epoch,
        "epochs_to_threshold_scratch": scratch_epoch,
        "transfer_curve": transfer_curve,
        "scratch_curve": scratch_curve,
        "note": "scikit-learn MLP does not support freezing selected layers; this uses warm-start transfer instead.",
    }

    metrics_path = out / "transfer_learning_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_epochs + 1), transfer_curve, label="transfer (TTT -> C4)")
    plt.plot(range(1, max_epochs + 1), scratch_curve, label="scratch (C4)")
    plt.axhline(y=threshold, color="gray", linestyle="--", label="threshold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Transfer Learning Convergence")
    plt.legend()
    plt.tight_layout()
    plot_path = out / "transfer_learning_curve.png"
    plt.savefig(plot_path)
    plt.close()

    metrics["metrics_path"] = str(metrics_path)
    metrics["plot_path"] = str(plot_path)
    return metrics


def _build_partial_model(random_state: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=64,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )


def _pad_features(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[1] == target_dim:
        return x
    padded = np.zeros((x.shape[0], target_dim), dtype=np.float32)
    padded[:, : x.shape[1]] = x
    return padded


def _first_below(curve: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(curve, start=1):
        if value <= threshold:
            return idx
    return None
