from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from game_solver.ml.transfer_learning import run_transfer_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning experiment")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts") / "datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / "benchmarks")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-epochs", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    report = run_transfer_experiment(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(report, indent=2))
