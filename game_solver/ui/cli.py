from __future__ import annotations

from typing import Callable

from game_solver.core.alpha_beta import alpha_beta_search
from game_solver.core.game_state import GameState
from game_solver.games.chess_simplified import ChessSimplifiedState
from game_solver.games.connect_four import ConnectFourState
from game_solver.games.tic_tac_toe import TicTacToeState


def run_cli(
    game_name: str,
    depth: int,
    human_player: int,
    evaluator: Callable[[GameState], float] | None = None,
    move_orderer: Callable[[GameState, list], list] | None = None,
) -> None:
    state = _create_initial_state(game_name)

    print(f"\nStarting {state.name} | depth={depth} | human_player={human_player}")

    while not state.is_terminal():
        print("\n" + state.render())

        if state.current_player == human_player:
            legal = state.get_legal_moves()
            print(f"Legal moves: {[ _move_to_label(state, move) for move in legal ]}")
            move = _read_human_move(state)
            state = state.apply_move(move)
            continue

        result = alpha_beta_search(
            state=state,
            depth=depth,
            evaluator=evaluator,
            move_orderer=move_orderer,
        )

        if result.best_move is None:
            break

        print(
            "AI move:",
            _move_to_label(state, result.best_move),
            f"| score={result.score:.3f}",
            f"| nodes={result.stats.nodes_explored}",
            f"| pruned={result.stats.pruned_branches}",
            f"| time={result.stats.elapsed_ms:.1f}ms",
        )
        state = state.apply_move(result.best_move)

    print("\nFinal board:")
    print(state.render())

    winner = state.winner()
    if winner == 0:
        print("Result: Draw")
    elif winner == human_player:
        print("Result: You win")
    else:
        print("Result: AI wins")


def _create_initial_state(game_name: str) -> GameState:
    normalized = game_name.strip().lower()
    if normalized in {"ttt", "tic_tac_toe", "tic-tac-toe"}:
        return TicTacToeState.initial()
    if normalized in {"connect_four", "connect4", "connect-4", "c4"}:
        return ConnectFourState.initial()
    if normalized in {"chess", "chess_simplified"}:
        return ChessSimplifiedState.initial()
    raise ValueError(f"Unsupported game: {game_name}")


def _read_human_move(state: GameState):
    while True:
        token = input("Enter your move: ").strip()
        try:
            move = _label_to_move(state, token)
        except Exception as exc:
            print(f"Invalid move format: {exc}")
            continue

        if move not in state.get_legal_moves():
            print("Move is not legal in this position.")
            continue
        return move


def _move_to_label(state: GameState, move) -> str:
    if isinstance(state, TicTacToeState):
        return TicTacToeState.move_to_label(move)
    if isinstance(state, ConnectFourState):
        return ConnectFourState.move_to_label(move)
    if isinstance(state, ChessSimplifiedState):
        return ChessSimplifiedState.move_to_label(move)
    return str(move)


def _label_to_move(state: GameState, label: str):
    if isinstance(state, TicTacToeState):
        return TicTacToeState.label_to_move(label)
    if isinstance(state, ConnectFourState):
        return ConnectFourState.label_to_move(label)
    if isinstance(state, ChessSimplifiedState):
        return ChessSimplifiedState.label_to_move(label)
    raise ValueError("Unsupported state type")
