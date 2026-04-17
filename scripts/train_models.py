from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.ml.training import train_all_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heuristic + move-order networks")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["tic_tac_toe", "connect_four", "chess"],
        help="Games to train: tic_tac_toe connect_four chess",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts") / "datasets")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts") / "models")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--full-data-refit-games",
        nargs="*",
        default=["chess"],
        help="Games whose saved models are refit on full dataset after evaluation split training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    reports = train_all_games(
        games=args.games,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        random_state=args.seed,
        full_data_refit_games=args.full_data_refit_games,
    )

    print(json.dumps(reports, indent=2))
