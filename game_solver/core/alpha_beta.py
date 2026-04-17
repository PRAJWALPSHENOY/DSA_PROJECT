from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .game_state import GameState

EvaluatorFn = Callable[[GameState], float]
MoveOrderFn = Callable[[GameState, list[Any]], list[Any]]


@dataclass
class SearchStats:
    nodes_explored: int = 0
    pruned_branches: int = 0
    max_depth_reached: int = 0
    nn_eval_calls: int = 0
    nn_order_calls: int = 0
    elapsed_ms: float = 0.0

    @property
    def pruning_efficiency(self) -> float:
        total = self.nodes_explored + self.pruned_branches
        if total == 0:
            return 0.0
        return (self.pruned_branches / total) * 100.0


@dataclass
class SearchResult:
    best_move: Any | None
    score: float
    stats: SearchStats
    ranked_moves: list[tuple[Any, float]] = field(default_factory=list)


def alpha_beta_search(
    state: GameState,
    depth: int,
    evaluator: EvaluatorFn | None = None,
    move_orderer: MoveOrderFn | None = None,
) -> SearchResult:
    """Run negamax alpha-beta search and return best move + explainability stats."""
    start = time.perf_counter()
    stats = SearchStats()

    legal_moves = state.get_legal_moves()
    if depth <= 0 or state.is_terminal() or not legal_moves:
        score = _evaluate(state, evaluator, stats)
        stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResult(best_move=None, score=score, stats=stats, ranked_moves=[])

    if move_orderer is not None:
        legal_moves = move_orderer(state, legal_moves)
        stats.nn_order_calls += 1

    alpha = -math.inf
    beta = math.inf
    best_score = -math.inf
    best_move = legal_moves[0]
    ranked_moves: list[tuple[Any, float]] = []

    for move_index, move in enumerate(legal_moves):
        child = state.apply_move(move)
        score = -_alpha_beta(
            child,
            depth - 1,
            -beta,
            -alpha,
            evaluator,
            move_orderer,
            stats,
            current_ply=1,
        )
        ranked_moves.append((move, score))

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            stats.pruned_branches += max(0, len(legal_moves) - move_index - 1)
            break

    ranked_moves.sort(key=lambda pair: pair[1], reverse=True)
    stats.elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SearchResult(best_move=best_move, score=best_score, stats=stats, ranked_moves=ranked_moves)


def _alpha_beta(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    evaluator: EvaluatorFn | None,
    move_orderer: MoveOrderFn | None,
    stats: SearchStats,
    current_ply: int,
) -> float:
    stats.nodes_explored += 1
    stats.max_depth_reached = max(stats.max_depth_reached, current_ply)

    legal_moves = state.get_legal_moves()
    if depth == 0 or state.is_terminal() or not legal_moves:
        return _evaluate(state, evaluator, stats)

    if move_orderer is not None:
        legal_moves = move_orderer(state, legal_moves)
        stats.nn_order_calls += 1

    best_score = -math.inf

    for move_index, move in enumerate(legal_moves):
        child = state.apply_move(move)
        score = -_alpha_beta(
            child,
            depth - 1,
            -beta,
            -alpha,
            evaluator,
            move_orderer,
            stats,
            current_ply=current_ply + 1,
        )
        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            stats.pruned_branches += max(0, len(legal_moves) - move_index - 1)
            break

    return best_score


def _evaluate(state: GameState, evaluator: EvaluatorFn | None, stats: SearchStats) -> float:
    if evaluator is None:
        return state.handcrafted_eval()

    stats.nn_eval_calls += 1
    score = float(evaluator(state))
    return max(-1.0, min(1.0, score))
