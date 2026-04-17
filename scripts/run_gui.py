from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.ui.tk_gui import MainMenuApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Game Solver GUI Launcher")
    parser.add_argument("--model-dir", type=str, default="artifacts/models")
    parser.add_argument("--depth", type=int, default=4, help="Initial depth shown in GUI (can be changed in-app).")
    parser.add_argument(
        "--human-player",
        type=int,
        choices=[1, -1],
        default=1,
        help="Initial side: 1 for first (White/X), -1 for second (Black/O).",
    )
    parser.add_argument("--nn-eval", dest="nn_eval", action="store_true", help="Start GUI with NN evaluator enabled.")
    parser.add_argument("--no-nn-eval", dest="nn_eval", action="store_false", help="Start GUI with NN evaluator disabled.")
    parser.add_argument("--nn-order", dest="nn_order", action="store_true", help="Start GUI with NN ordering enabled.")
    parser.add_argument("--no-nn-order", dest="nn_order", action="store_false", help="Start GUI with NN ordering disabled.")
    parser.set_defaults(nn_eval=True, nn_order=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = MainMenuApp(
        model_dir=args.model_dir,
        default_depth=args.depth,
        default_human_player=args.human_player,
        default_nn_eval=args.nn_eval,
        default_nn_order=args.nn_order,
    )
    app.run()
