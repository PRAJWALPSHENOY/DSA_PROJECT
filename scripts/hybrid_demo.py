from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.data.collectors import create_initial_state
from game_solver.engine.hybrid_engine import HybridEngine, HybridEngineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate trained networks into alpha-beta")
    parser.add_argument("--game", type=str, default="connect_four")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--model-dir", type=str, default="artifacts/models")
    parser.add_argument("--nn-eval", action="store_true", help="Enable heuristic network")
    parser.add_argument("--nn-order", action="store_true", help="Enable move-ordering network")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    state = create_initial_state(args.game)
    engine = HybridEngine.from_model_dir(
        game_name=args.game,
        model_dir=args.model_dir,
        config=HybridEngineConfig(use_nn_eval=args.nn_eval, use_nn_order=args.nn_order),
    )

    result = engine.search(state=state, depth=args.depth)

    print(f"Game: {args.game}")
    print(f"Best move: {result.best_move}")
    print(f"Score: {result.score:.4f}")
    print(f"Nodes explored: {result.stats.nodes_explored}")
    print(f"Pruned branches: {result.stats.pruned_branches}")
    print(f"Pruning efficiency: {result.stats.pruning_efficiency:.2f}%")
    print(f"Top candidates: {engine.top_candidates(result, top_k=3)}")
    print(f"Decision source: {engine.decision_source(result)}")
