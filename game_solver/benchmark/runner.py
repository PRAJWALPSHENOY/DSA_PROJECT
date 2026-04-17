from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from game_solver.data.collectors import create_initial_state
from game_solver.data.features import move_to_label
from game_solver.engine.hybrid_engine import HybridEngine, HybridEngineConfig

CONFIGS = {
    "baseline": HybridEngineConfig(use_nn_eval=False, use_nn_order=False),
    "nn_eval": HybridEngineConfig(use_nn_eval=True, use_nn_order=False),
    "nn_order": HybridEngineConfig(use_nn_eval=False, use_nn_order=True),
    "full_hybrid": HybridEngineConfig(use_nn_eval=True, use_nn_order=True),
}


def run_benchmarks(
    games: list[str],
    depth: int,
    positions_per_game: int,
    match_games: int,
    model_dir: str | Path,
    output_dir: str | Path,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    match_rows: list[dict] = []

    raw_path = out / "raw_turn_metrics.csv"
    summary_path = out / "summary_metrics.csv"
    match_path = out / "match_metrics.csv"

    def _save_checkpoints():
        if raw_rows:
            pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        if match_rows:
            pd.DataFrame(match_rows).to_csv(match_path, index=False)

    for game_name in games:
        print(f"\n[{game_name.upper()}] Starting benchmark...")
        engines = {
            config_name: HybridEngine.from_model_dir(game_name=game_name, model_dir=model_dir, config=config)
            for config_name, config in CONFIGS.items()
        }

        positions = _sample_positions(game_name, positions_per_game, rng)
        print(f"[{game_name.upper()}] Sampled {len(positions)} positions. Evaluating baseline (depth={depth})...")

        baseline_results = []
        for position in positions:
            baseline_results.append(engines["baseline"].search(position, depth=depth))

        for config_name, engine in engines.items():
            print(f"[{game_name.upper()}] Evaluating {config_name} configuration...")
            config_rows = []
            for idx, position in enumerate(positions):
                result = engine.search(position, depth=depth)
                baseline = baseline_results[idx]

                baseline_best_label = None if baseline.best_move is None else move_to_label(position, baseline.best_move)
                predicted_label = None if result.best_move is None else move_to_label(position, result.best_move)
                top3_labels = [move_to_label(position, move) for move, _ in result.ranked_moves[:3]]
                baseline_rank_map = _rank_map(position, baseline.ranked_moves)
                predicted_rank_map = _rank_map(position, result.ranked_moves)
                rank_corr = _spearman_rank_correlation(baseline_rank_map, predicted_rank_map)
                nn_calls = result.stats.nn_eval_calls + result.stats.nn_order_calls
                nodes_saved = baseline.stats.nodes_explored - result.stats.nodes_explored

                row = {
                    "game": game_name,
                    "config": config_name,
                    "position_index": idx,
                    "score": result.score,
                    "nodes_explored": result.stats.nodes_explored,
                    "pruned_branches": result.stats.pruned_branches,
                    "pruning_efficiency": result.stats.pruning_efficiency,
                    "time_per_move_ms": result.stats.elapsed_ms,
                    "nn_eval_calls": result.stats.nn_eval_calls,
                    "nn_order_calls": result.stats.nn_order_calls,
                    "move_order_agreement": float(predicted_label == baseline_best_label),
                    "top3_hit_rate": float(baseline_best_label in top3_labels if baseline_best_label is not None else 0.0),
                    "exact_match_rate": float(predicted_label == baseline_best_label),
                    "move_rank_correlation": rank_corr,
                    "nodes_saved_vs_baseline": float(nodes_saved),
                    "nodes_saved_per_nn_call": (float(nodes_saved) / nn_calls) if nn_calls > 0 else None,
                    "decision_source": engine.decision_source(result),
                    "search_depth_in_1s": _search_depth_within_budget(position, engine, time_budget_ms=1000),
                }
                config_rows.append(row)
                raw_rows.append(row)
                if (idx + 1) % max(1, len(positions) // 5) == 0:
                    print(f"  - Position evaluations: {idx + 1}/{len(positions)}")

            df_config = pd.DataFrame(config_rows)
            summary_rows.append(
                {
                    "game": game_name,
                    "config": config_name,
                    "nodes_explored_per_move": float(df_config["nodes_explored"].mean()),
                    "pruning_efficiency_pct": float(df_config["pruning_efficiency"].mean()),
                    "move_order_agreement_pct": float(df_config["move_order_agreement"].mean() * 100.0),
                    "move_order_top3_hit_rate_pct": float(df_config["top3_hit_rate"].mean() * 100.0),
                    "move_order_exact_match_pct": float(df_config["exact_match_rate"].mean() * 100.0),
                    "move_rank_correlation": float(df_config["move_rank_correlation"].dropna().mean()) if not df_config["move_rank_correlation"].dropna().empty else None,
                    "nodes_saved_vs_baseline": float(df_config["nodes_saved_vs_baseline"].mean()),
                    "nodes_saved_per_nn_call": float(df_config["nodes_saved_per_nn_call"].dropna().mean()) if not df_config["nodes_saved_per_nn_call"].dropna().empty else None,
                    "time_per_move_ms": float(df_config["time_per_move_ms"].mean()),
                    "nn_inference_overhead_ms": float((df_config["time_per_move_ms"] * (df_config["nn_eval_calls"] + df_config["nn_order_calls"] > 0)).mean()),
                    "search_depth_in_fixed_time": float(df_config["search_depth_in_1s"].mean()),
                }
            )
            _save_checkpoints()

        for config_name in ("nn_eval", "nn_order", "full_hybrid"):
            print(f"[{game_name.upper()}] Starting head-to-head matches: {config_name} vs baseline ({match_games} games)...")
            win_rate = _play_match_series(
                game_name=game_name,
                depth=depth,
                games=match_games,
                first_engine=engines[config_name],
                second_engine=engines["baseline"],
                seed=seed,
            )
            match_rows.append(
                {
                    "game": game_name,
                    "config": config_name,
                    "metric": "win_rate_vs_baseline",
                    "value": win_rate,
                }
            )
            _save_checkpoints()

    # Final plot and wrap-up
    if summary_rows:
        _plot_summary(pd.DataFrame(summary_rows), out)

    return {
        "raw_metrics": str(raw_path),
        "summary_metrics": str(summary_path),
        "match_metrics": str(match_path),
    }


def _sample_positions(game_name: str, count: int, rng: random.Random):
    positions = []
    for _ in range(count):
        state = create_initial_state(game_name)
        rollout = rng.randint(0, 10)
        for _ply in range(rollout):
            if state.is_terminal():
                break
            legal = state.get_legal_moves()
            if not legal:
                break
            state = state.apply_move(rng.choice(legal))
        positions.append(state)
    return positions


def _search_depth_within_budget(state, engine: HybridEngine, time_budget_ms: float = 1000.0, max_depth: int = 8) -> int:
    achieved = 0
    for depth in range(1, max_depth + 1):
        result = engine.search(state=state, depth=depth)
        if result.stats.elapsed_ms > time_budget_ms:
            break
        achieved = depth
    return achieved


def _play_match_series(
    game_name: str,
    depth: int,
    games: int,
    first_engine: HybridEngine,
    second_engine: HybridEngine,
    seed: int,
) -> float:
    rng = random.Random(seed)
    first_wins = 0.0

    for game_idx in range(games):
        state = create_initial_state(game_name)
        first_as_player_one = (game_idx % 2 == 0)

        max_plies = 160 if game_name == "chess" else 80
        plies = 0

        while not state.is_terminal() and plies < max_plies:
            current_engine = None
            if first_as_player_one:
                current_engine = first_engine if state.current_player == 1 else second_engine
            else:
                current_engine = second_engine if state.current_player == 1 else first_engine

            result = current_engine.search(state=state, depth=depth)
            if result.best_move is None:
                legal = state.get_legal_moves()
                if not legal:
                    break
                state = state.apply_move(rng.choice(legal))
            else:
                state = state.apply_move(result.best_move)
            plies += 1

        winner = state.winner()
        if winner == 0:
            first_wins += 0.5
        elif first_as_player_one and winner == 1:
            first_wins += 1.0
        elif (not first_as_player_one) and winner == -1:
            first_wins += 1.0

        print(f"  - Match {game_idx + 1}/{games} finished (Plies: {plies}, Winner: {winner})")

    return first_wins / max(1, games)


def _plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    if summary_df.empty:
        return

    for game_name in summary_df["game"].unique():
        game_df = summary_df[summary_df["game"] == game_name]
        plt.figure(figsize=(10, 5))
        plt.bar(game_df["config"], game_df["nodes_explored_per_move"])
        plt.title(f"{game_name}: Nodes Explored per Move")
        plt.ylabel("Nodes")
        plt.tight_layout()
        plt.savefig(output_dir / f"{game_name}_nodes.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(game_df["config"], game_df["time_per_move_ms"])
        plt.title(f"{game_name}: Time per Move (ms)")
        plt.ylabel("Milliseconds")
        plt.tight_layout()
        plt.savefig(output_dir / f"{game_name}_time.png")
        plt.close()


def _rank_map(state, ranked_moves: list[tuple[object, float]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for rank, (move, _score) in enumerate(ranked_moves, start=1):
        mapping[move_to_label(state, move)] = rank
    return mapping


def _spearman_rank_correlation(reference: dict[str, int], candidate: dict[str, int]) -> float | None:
    common = sorted(set(reference.keys()) & set(candidate.keys()))
    if len(common) < 2:
        return None

    ref = np.array([float(reference[label]) for label in common], dtype=np.float64)
    cand = np.array([float(candidate[label]) for label in common], dtype=np.float64)

    corr = np.corrcoef(ref, cand)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)
