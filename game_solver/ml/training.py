from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, top_k_accuracy_score
from sklearn.model_selection import train_test_split

from game_solver.data.dataset_io import feature_columns, load_dataframe
from game_solver.ml.models import HeuristicModel, MoveOrderingModel


def train_models_for_game(
    game_name: str,
    dataset_dir: str | Path,
    model_dir: str | Path,
    random_state: int = 7,
    full_data_refit: bool = False,
) -> dict:
    dataset_root = Path(dataset_dir)
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    states_df = load_dataframe(dataset_root / f"{game_name}_states.csv")
    moves_df = load_dataframe(dataset_root / f"{game_name}_moves.csv")

    feature_cols = feature_columns(states_df)
    x_states = states_df[feature_cols].to_numpy(dtype=np.float32)
    y_states = states_df["label"].to_numpy(dtype=np.float32)

    x_train, x_val, x_test, y_train, y_val, y_test, small_sample_mode = _safe_three_way_split(
        x_states,
        y_states,
        random_state=random_state,
    )

    heuristic_model = HeuristicModel(feature_dim=x_states.shape[1], random_state=random_state)
    heuristic_model.fit(x_train, y_train)

    val_pred = heuristic_model.predict_batch(x_val)
    test_pred = heuristic_model.predict_batch(x_test)

    state_metrics = {
        "val_mse": float(mean_squared_error(y_val, val_pred)),
        "test_mse": float(mean_squared_error(y_test, test_pred)),
        "final_fit_dataset": "full" if full_data_refit else "train_split",
    }

    if full_data_refit:
        heuristic_model = HeuristicModel(feature_dim=x_states.shape[1], random_state=random_state)
        heuristic_model.fit(x_states, y_states)

    heuristic_path = model_root / f"{game_name}_heuristic.pkl"
    heuristic_model.save(heuristic_path)

    move_metrics = _train_move_model(
        game_name,
        moves_df,
        feature_cols,
        model_root,
        random_state,
        full_data_refit=full_data_refit,
    )

    metrics = {
        "game": game_name,
        "state_samples": int(len(states_df)),
        "move_samples": int(len(moves_df)),
        "feature_dim": int(x_states.shape[1]),
        "small_sample_mode": small_sample_mode,
        "full_data_refit": bool(full_data_refit),
        "heuristic_model_path": str(heuristic_path),
        "heuristic": state_metrics,
        "move_order": move_metrics,
    }

    metrics_path = model_root / f"{game_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def train_all_games(
    games: list[str],
    dataset_dir: str | Path,
    model_dir: str | Path,
    random_state: int = 7,
    full_data_refit_games: list[str] | set[str] | None = None,
) -> list[dict]:
    refit_targets = {
        _normalize_game_name(game_name)
        for game_name in (full_data_refit_games if full_data_refit_games is not None else [])
    }

    outputs = []
    for game_name in games:
        outputs.append(
            train_models_for_game(
                game_name,
                dataset_dir,
                model_dir,
                random_state=random_state,
                full_data_refit=_normalize_game_name(game_name) in refit_targets,
            )
        )
    return outputs


def _train_move_model(
    game_name: str,
    moves_df: pd.DataFrame,
    feature_cols: list[str],
    model_root: Path,
    random_state: int,
    full_data_refit: bool,
) -> dict:
    x_moves = moves_df[feature_cols].to_numpy(dtype=np.float32)
    labels = moves_df["move_label"].astype(str).to_numpy()

    if len(np.unique(labels)) < 2:
        return {
            "trained": False,
            "reason": "insufficient unique move labels",
            "final_fit_dataset": "none",
        }

    x_train, x_val, x_test, y_train, y_val, y_test, small_sample_mode = _safe_three_way_split(
        x_moves,
        labels,
        random_state=random_state,
        stratify_first=True,
    )

    move_model = MoveOrderingModel(feature_dim=x_moves.shape[1], random_state=random_state)
    move_model.fit(x_train, y_train)

    train_label_set = set(str(label) for label in np.unique(y_train))

    val_mask = np.array([label in train_label_set for label in y_val], dtype=bool)
    test_mask = np.array([label in train_label_set for label in y_test], dtype=bool)

    x_val_eval = x_val[val_mask]
    y_val_eval = y_val[val_mask]
    x_test_eval = x_test[test_mask]
    y_test_eval = y_test[test_mask]

    val_accuracy = None
    test_accuracy = None
    top3_hit_rate = None

    if len(y_val_eval) > 0:
        encoded_val = move_model.encoder.transform(y_val_eval)
        val_pred = move_model.model.predict(x_val_eval)
        val_accuracy = float(accuracy_score(encoded_val, val_pred))

    if len(y_test_eval) > 0:
        encoded_test = move_model.encoder.transform(y_test_eval)
        test_pred = move_model.model.predict(x_test_eval)
        test_accuracy = float(accuracy_score(encoded_test, test_pred))

        test_proba = move_model.model.predict_proba(x_test_eval)
        top_k = min(3, test_proba.shape[1])
        top3_hit_rate = float(
            top_k_accuracy_score(
                encoded_test,
                test_proba,
                k=top_k,
                labels=np.arange(test_proba.shape[1]),
            )
        )

    move_metrics = {
        "trained": True,
        "small_sample_mode": small_sample_mode,
        "final_fit_dataset": "full" if full_data_refit else "train_split",
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "test_top3_hit_rate": top3_hit_rate,
        "val_seen_ratio": float(val_mask.mean()) if len(val_mask) > 0 else 0.0,
        "test_seen_ratio": float(test_mask.mean()) if len(test_mask) > 0 else 0.0,
    }

    if full_data_refit:
        move_model = MoveOrderingModel(feature_dim=x_moves.shape[1], random_state=random_state)
        move_model.fit(x_moves, labels)

    move_path = model_root / f"{game_name}_move_order.pkl"
    move_model.save(move_path)
    move_metrics["model_path"] = str(move_path)

    return move_metrics


def _safe_three_way_split(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
    stratify_first: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    # For tiny datasets, using the full set for all splits avoids split-time failures.
    if len(x) < 10:
        return x, x, x, y, y, y, True

    kwargs = {
        "test_size": 0.2,
        "random_state": random_state,
        "shuffle": True,
    }

    x_train = x
    x_temp = x
    y_train = y
    y_temp = y

    try:
        if stratify_first and len(np.unique(y)) > 1:
            x_train, x_temp, y_train, y_temp = train_test_split(x, y, stratify=y, **kwargs)
        else:
            x_train, x_temp, y_train, y_temp = train_test_split(x, y, **kwargs)
    except ValueError:
        try:
            x_train, x_temp, y_train, y_temp = train_test_split(x, y, **kwargs)
        except ValueError:
            return x, x, x, y, y, y, True

    try:
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            random_state=random_state,
            shuffle=True,
            stratify=y_temp if stratify_first and len(np.unique(y_temp)) > 1 else None,
        )
    except ValueError:
        try:
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp,
                y_temp,
                test_size=0.5,
                random_state=random_state,
                shuffle=True,
            )
        except ValueError:
            return x, x, x, y, y, y, True

    return x_train, x_val, x_test, y_train, y_val, y_test, False


def _normalize_game_name(game_name: str) -> str:
    return game_name.strip().lower()
