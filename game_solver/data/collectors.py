from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import chess.pgn
import pandas as pd

from game_solver.core.alpha_beta import alpha_beta_search
from game_solver.core.game_state import GameState
from game_solver.data.dataset_io import save_dataframe
from game_solver.data.features import extract_features, move_to_label
from game_solver.games.chess_simplified import ChessSimplifiedState
from game_solver.games.connect_four import ConnectFourState
from game_solver.games.tic_tac_toe import TicTacToeState


@dataclass
class CollectionConfig:
    game_name: str
    num_games: int
    search_depth: int = 2
    epsilon: float = 0.25
    max_plies: int = 60
    seed: int = 7
    chess_pgn_path: str | Path | None = None
    chess_max_games: int | None = None
    chess_max_moves: int | None = None


def collect_self_play(config: CollectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(config.seed)

    state_rows: list[dict] = []
    move_rows: list[dict] = []

    for game_idx in range(config.num_games):
        state = create_initial_state(config.game_name)
        trajectory: list[tuple[int, GameState, object, list[float]]] = []

        ply = 0
        while not state.is_terminal() and ply < config.max_plies:
            features = extract_features(state).tolist()
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                break

            if rng.random() < config.epsilon:
                selected_move = rng.choice(legal_moves)
            else:
                result = alpha_beta_search(state=state, depth=config.search_depth)
                selected_move = result.best_move if result.best_move is not None else rng.choice(legal_moves)

            trajectory.append((ply, state, selected_move, features))
            state = state.apply_move(selected_move)
            ply += 1

        outcome = state.winner()

        for ply, seen_state, selected_move, features in trajectory:
            label = float(outcome * seen_state.current_player)

            state_row = {
                "game": config.game_name,
                "game_id": game_idx,
                "ply": ply,
                "serialized": seen_state.serialize(),
                "label": label,
            }
            move_row = {
                "game": config.game_name,
                "game_id": game_idx,
                "ply": ply,
                "serialized": seen_state.serialize(),
                "label": label,
                "move_label": move_to_label(seen_state, selected_move),
            }

            for index, value in enumerate(features):
                col = f"f{index}"
                state_row[col] = float(value)
                move_row[col] = float(value)

            state_rows.append(state_row)
            move_rows.append(move_row)

    state_df = pd.DataFrame(state_rows)
    move_df = pd.DataFrame(move_rows)
    return state_df, move_df


def collect_and_save(config: CollectionConfig, output_dir: str | Path) -> tuple[Path, Path]:
    is_chess = config.game_name.strip().lower() in {"chess", "chess_simplified"}
    if is_chess and config.chess_pgn_path:
        state_df, move_df = collect_chess_from_pgn(config)
    else:
        state_df, move_df = collect_self_play(config)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    state_path = output / f"{config.game_name}_states.csv"
    move_path = output / f"{config.game_name}_moves.csv"

    save_dataframe(state_df, state_path)
    save_dataframe(move_df, move_path)

    return state_path, move_path


def collect_chess_from_pgn(config: CollectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.chess_pgn_path is None:
        raise ValueError("chess_pgn_path is required for PGN chess import")

    pgn_path = Path(config.chess_pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    state_rows: list[dict] = []
    move_rows: list[dict] = []

    accepted_games = 0
    source_index = 0

    max_games = config.chess_max_games
    if max_games is not None and max_games <= 0:
        max_games = None

    with _open_chess_text_stream(pgn_path) as handle:
        while True:
            if max_games is not None and accepted_games >= max_games:
                break

            game = chess.pgn.read_game(handle)
            if game is None:
                break

            source_index += 1
            plies = sum(1 for _ in game.mainline_moves())
            full_moves = (plies + 1) // 2

            if config.chess_max_moves is not None and full_moves >= config.chess_max_moves:
                continue

            outcome = _parse_result(game.headers.get("Result", "*"))
            if outcome is None:
                continue

            state = ChessSimplifiedState.initial()
            trajectory: list[tuple[int, GameState, object, list[float]]] = []
            skip_game = False

            for ply, move in enumerate(game.mainline_moves()):
                board = state.board

                # Keep dataset aligned with simplified chess rules used by the engine.
                if board.is_castling(move) or board.is_en_passant(move):
                    skip_game = True
                    break
                if move not in board.legal_moves:
                    skip_game = True
                    break

                features = extract_features(state).tolist()
                trajectory.append((ply, state, move, features))
                state = state.apply_move(move)

            if skip_game or not trajectory:
                continue

            game_id = accepted_games
            accepted_games += 1

            for ply, seen_state, selected_move, features in trajectory:
                label = float(outcome * seen_state.current_player)

                state_row = {
                    "game": config.game_name,
                    "game_id": game_id,
                    "source_game_index": source_index,
                    "ply": ply,
                    "serialized": seen_state.serialize(),
                    "label": label,
                }
                move_row = {
                    "game": config.game_name,
                    "game_id": game_id,
                    "source_game_index": source_index,
                    "ply": ply,
                    "serialized": seen_state.serialize(),
                    "label": label,
                    "move_label": move_to_label(seen_state, selected_move),
                }

                for index, value in enumerate(features):
                    col = f"f{index}"
                    state_row[col] = float(value)
                    move_row[col] = float(value)

                state_rows.append(state_row)
                move_rows.append(move_row)

    return pd.DataFrame(state_rows), pd.DataFrame(move_rows)


@contextmanager
def _open_chess_text_stream(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() == ".zst":
        import compression.zstd as zstd
        with zstd.open(path, mode="rt", encoding="utf-8", errors="replace") as handle:
            yield handle
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield handle


def _parse_result(result_token: str) -> int | None:
    if result_token == "1-0":
        return 1
    if result_token == "0-1":
        return -1
    if result_token == "1/2-1/2":
        return 0
    return None


def create_initial_state(game_name: str) -> GameState:
    normalized = game_name.strip().lower()
    if normalized in {"ttt", "tic_tac_toe", "tic-tac-toe"}:
        return TicTacToeState.initial()
    if normalized in {"connect_four", "connect4", "connect-4", "c4"}:
        return ConnectFourState.initial()
    if normalized in {"chess", "chess_simplified"}:
        return ChessSimplifiedState.initial()
    raise ValueError(f"Unsupported game name: {game_name}")
