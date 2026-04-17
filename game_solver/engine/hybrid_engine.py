from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from game_solver.core.alpha_beta import SearchResult, alpha_beta_search
from game_solver.core.game_state import GameState
from game_solver.data.features import extract_features, move_to_label
from game_solver.ml.models import HeuristicModel, MoveOrderingModel


@dataclass
class HybridEngineConfig:
    use_nn_eval: bool = True
    use_nn_order: bool = True


class HybridEngine:
    def __init__(
        self,
        game_name: str,
        heuristic_model: HeuristicModel | None = None,
        move_order_model: MoveOrderingModel | None = None,
        config: HybridEngineConfig | None = None,
    ) -> None:
        self.game_name = game_name
        self.heuristic_model = heuristic_model
        self.move_order_model = move_order_model
        self.config = config or HybridEngineConfig()

    @classmethod
    def from_model_dir(
        cls,
        game_name: str,
        model_dir: str | Path,
        config: HybridEngineConfig | None = None,
    ) -> "HybridEngine":
        model_root = Path(model_dir)
        heuristic_path = model_root / f"{game_name}_heuristic.pkl"
        move_path = model_root / f"{game_name}_move_order.pkl"

        heuristic_model = HeuristicModel.load(heuristic_path) if heuristic_path.exists() else None
        move_order_model = MoveOrderingModel.load(move_path) if move_path.exists() else None

        return cls(
            game_name=game_name,
            heuristic_model=heuristic_model,
            move_order_model=move_order_model,
            config=config,
        )

    def search(self, state: GameState, depth: int) -> SearchResult:
        eval_fn = self._nn_eval if self.config.use_nn_eval and self.heuristic_model is not None else None
        order_fn = self._nn_order if self.config.use_nn_order and self.move_order_model is not None else None

        return alpha_beta_search(state=state, depth=depth, evaluator=eval_fn, move_orderer=order_fn)

    def decision_source(self, result: SearchResult) -> str:
        if result.stats.nn_order_calls > result.stats.nn_eval_calls:
            return "move_ordering_network"
        if result.stats.nn_eval_calls > 0:
            return "heuristic_network"
        return "handcrafted"

    def top_candidates(self, result: SearchResult, top_k: int = 3) -> list[tuple[Any, float]]:
        return result.ranked_moves[:top_k]

    def _nn_eval(self, state: GameState) -> float:
        assert self.heuristic_model is not None
        features = extract_features(state)
        return self.heuristic_model.predict(features)

    def _nn_order(self, state: GameState, moves: list[Any]) -> list[Any]:
        assert self.move_order_model is not None

        features = extract_features(state)
        labels = [move_to_label(state, move) for move in moves]
        ranked_labels = self.move_order_model.rank_labels(features, labels)

        buckets: dict[str, list[Any]] = {}
        for move in moves:
            buckets.setdefault(move_to_label(state, move), []).append(move)

        ranked_moves: list[Any] = []
        for label in ranked_labels:
            ranked_moves.extend(buckets.get(label, []))
            buckets.pop(label, None)

        for leftovers in buckets.values():
            ranked_moves.extend(leftovers)

        return ranked_moves
