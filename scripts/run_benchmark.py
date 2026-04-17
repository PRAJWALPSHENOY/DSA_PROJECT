from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.benchmark.runner import run_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs hybrid benchmark suite")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["tic_tac_toe", "connect_four", "chess"],
        help="Games to benchmark",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--positions-per-game", type=int, default=30)
    parser.add_argument("--match-games", type=int, default=20)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts") / "models")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "benchmarks")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    outputs = run_benchmarks(
        games=args.games,
        depth=args.depth,
        positions_per_game=args.positions_per_game,
        match_games=args.match_games,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(json.dumps(outputs, indent=2))
