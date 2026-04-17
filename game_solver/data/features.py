from __future__ import annotations

from math import tanh

import chess
import numpy as np

from game_solver.core.game_state import GameState
from game_solver.games.chess_simplified import ChessSimplifiedState
from game_solver.games.connect_four import COLS, ROWS, ConnectFourState
from game_solver.games.tic_tac_toe import TicTacToeState


def extract_features(state: GameState) -> np.ndarray:
    if isinstance(state, TicTacToeState):
        return _ttt_features(state)
    if isinstance(state, ConnectFourState):
        return _connect_four_features(state)
    if isinstance(state, ChessSimplifiedState):
        return _chess_features(state)
    raise TypeError(f"Unsupported state type: {type(state)}")


def move_to_label(state: GameState, move) -> str:
    if isinstance(state, TicTacToeState):
        return TicTacToeState.move_to_label(move)
    if isinstance(state, ConnectFourState):
        return ConnectFourState.move_to_label(move)
    if isinstance(state, ChessSimplifiedState):
        return ChessSimplifiedState.move_to_label(move)
    return str(move)


def label_to_move(state: GameState, label: str):
    if isinstance(state, TicTacToeState):
        return TicTacToeState.label_to_move(label)
    if isinstance(state, ConnectFourState):
        return ConnectFourState.label_to_move(label)
    if isinstance(state, ChessSimplifiedState):
        return ChessSimplifiedState.label_to_move(label)
    raise TypeError(f"Unsupported state type: {type(state)}")


def _ttt_features(state: TicTacToeState) -> np.ndarray:
    board = np.array(state.board, dtype=np.int8)
    own = (board == state.current_player).astype(np.float32)
    opp = (board == -state.current_player).astype(np.float32)
    empty = (board == 0).astype(np.float32)

    material = float(own.sum() - opp.sum()) / 9.0
    center_control = 1.0 if board[4] == state.current_player else (-1.0 if board[4] == -state.current_player else 0.0)

    threat_own = 0
    threat_opp = 0
    lines = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
    for a, b, c in lines:
        values = [board[a], board[b], board[c]]
        if values.count(state.current_player) == 2 and values.count(0) == 1:
            threat_own += 1
        if values.count(-state.current_player) == 2 and values.count(0) == 1:
            threat_opp += 1

    threat_count = float(threat_own - threat_opp) / 8.0
    mobility = float(len(state.get_legal_moves())) / 9.0
    open_lines = float(
        sum(1 for a, b, c in lines if all(board[idx] != -state.current_player for idx in (a, b, c)))
        - sum(1 for a, b, c in lines if all(board[idx] != state.current_player for idx in (a, b, c)))
    ) / 8.0

    extras = np.array([material, center_control, threat_count, mobility, open_lines], dtype=np.float32)
    return np.concatenate([own, opp, empty, extras])


def _connect_four_features(state: ConnectFourState) -> np.ndarray:
    board = np.array(state.board, dtype=np.int8)
    own = (board == state.current_player).astype(np.float32).flatten()
    opp = (board == -state.current_player).astype(np.float32).flatten()
    empty = (board == 0).astype(np.float32).flatten()

    material = float(own.sum() - opp.sum()) / float(ROWS * COLS)

    center_col = COLS // 2
    center_control = float(
        np.count_nonzero(board[:, center_col] == state.current_player)
        - np.count_nonzero(board[:, center_col] == -state.current_player)
    ) / ROWS

    threat_own = 0
    threat_opp = 0
    open_own = 0
    open_opp = 0

    windows = []
    for row in range(ROWS):
        for col in range(COLS - 3):
            windows.append([board[row, col + offset] for offset in range(4)])
    for row in range(ROWS - 3):
        for col in range(COLS):
            windows.append([board[row + offset, col] for offset in range(4)])
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            windows.append([board[row + offset, col + offset] for offset in range(4)])
    for row in range(ROWS - 3):
        for col in range(3, COLS):
            windows.append([board[row + offset, col - offset] for offset in range(4)])

    for window in windows:
        if window.count(state.current_player) == 3 and window.count(0) == 1:
            threat_own += 1
        if window.count(-state.current_player) == 3 and window.count(0) == 1:
            threat_opp += 1
        if all(value != -state.current_player for value in window):
            open_own += 1
        if all(value != state.current_player for value in window):
            open_opp += 1

    threat_count = float(threat_own - threat_opp) / max(1.0, float(len(windows)))
    mobility = float(len(state.get_legal_moves())) / COLS
    open_lines = float(open_own - open_opp) / max(1.0, float(len(windows)))

    extras = np.array([material, center_control, threat_count, mobility, open_lines], dtype=np.float32)
    return np.concatenate([own, opp, empty, extras])


def _chess_features(state: ChessSimplifiedState) -> np.ndarray:
    board = state.board
    own_color = chess.WHITE if state.current_player == 1 else chess.BLACK
    opp_color = not own_color

    planes = []
    for color in (own_color, opp_color):
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            plane = np.zeros(64, dtype=np.float32)
            for square in board.pieces(piece_type, color):
                plane[square] = 1.0
            planes.append(plane)

    own_material = _material(board, own_color)
    opp_material = _material(board, opp_color)
    material_balance = tanh((own_material - opp_material) / 16.0)

    center_squares = (chess.D4, chess.E4, chess.D5, chess.E5)
    center_control = float(
        sum(1 for sq in center_squares if board.is_attacked_by(own_color, sq))
        - sum(1 for sq in center_squares if board.is_attacked_by(opp_color, sq))
    ) / 4.0

    attacked_opp = 0
    attacked_own = 0
    for square, piece in board.piece_map().items():
        if piece.color == own_color and board.is_attacked_by(opp_color, square):
            attacked_own += 1
        if piece.color == opp_color and board.is_attacked_by(own_color, square):
            attacked_opp += 1
    threat_count = tanh((attacked_opp - attacked_own) / 8.0)

    mobility = float(len(state.get_legal_moves())) / 64.0

    board_swapped = board.copy(stack=False)
    board_swapped.turn = not board_swapped.turn
    opp_mobility = float(len(list(board_swapped.legal_moves))) / 64.0
    open_lines = mobility - opp_mobility

    extras = np.array([material_balance, center_control, threat_count, mobility, open_lines], dtype=np.float32)
    return np.concatenate([*planes, extras])


def _material(board: chess.Board, color: chess.Color) -> float:
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }
    return sum(len(board.pieces(piece_type, color)) * value for piece_type, value in values.items())
