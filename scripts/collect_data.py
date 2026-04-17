from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.data.collectors import CollectionConfig, collect_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play data collection + feature engineering")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["tic_tac_toe", "connect_four", "chess"],
        help="Games to collect: tic_tac_toe connect_four chess",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=250,
        help="Self-play games per game type (used for non-PGN collection).",
    )
    parser.add_argument("--search-depth", type=int, default=2, help="Depth used for policy in self-play")
    parser.add_argument("--epsilon", type=float, default=0.25, help="Random move probability during self-play")
    parser.add_argument("--max-plies", type=int, default=60, help="Max plies before truncating a game")
    parser.add_argument(
        "--chess-pgn",
        type=Path,
        default=None,
        help="Optional PGN file for chess import. If provided, chess dataset is imported from PGN.",
    )
    parser.add_argument(
        "--chess-max-games",
        type=int,
        default=None,
        help="Optional cap on imported chess games from PGN. Default imports all games.",
    )
    parser.add_argument(
        "--chess-max-moves",
        type=int,
        default=None,
        help="Optional full-move cap (exclusive). Default keeps all game lengths.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "datasets",
        help="CSV output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for game_name in args.games:
        config = CollectionConfig(
            game_name=game_name,
            num_games=args.num_games,
            search_depth=args.search_depth,
            epsilon=args.epsilon,
            max_plies=args.max_plies,
            seed=args.seed,
            chess_pgn_path=args.chess_pgn,
            chess_max_games=args.chess_max_games,
            chess_max_moves=args.chess_max_moves,
        )
        states_path, moves_path = collect_and_save(config, args.output_dir)
        print(f"[{game_name}] states -> {states_path}")
        print(f"[{game_name}] moves  -> {moves_path}")
