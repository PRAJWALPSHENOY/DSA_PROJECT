from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.ui.cli import run_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline minimax + alpha-beta CLI")
    parser.add_argument("--game", type=str, default="tic_tac_toe", help="tic_tac_toe | connect_four | chess")
    parser.add_argument("--depth", type=int, default=4, help="Search depth")
    parser.add_argument("--human-player", type=int, default=1, choices=[1, -1], help="1 = first player, -1 = second")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cli(game_name=args.game, depth=args.depth, human_player=args.human_player)
