from __future__ import annotations

from dataclasses import dataclass
from math import tanh

import chess

from game_solver.core.game_state import GameState

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


@dataclass(frozen=True)
class ChessSimplifiedState(GameState):
    board: chess.Board
    name: str = "chess"

    @staticmethod
    def initial() -> "ChessSimplifiedState":
        return ChessSimplifiedState(board=chess.Board())

    @property
    def current_player(self) -> int:
        return 1 if self.board.turn == chess.WHITE else -1

    def get_legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)


    def apply_move(self, move: chess.Move) -> "ChessSimplifiedState":
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")

        copied = self.board.copy(stack=False)
        copied.push(move)
        return ChessSimplifiedState(board=copied)

    def is_terminal(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def winner(self) -> int:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0
        return 1 if outcome.winner == chess.WHITE else -1

    def handcrafted_eval(self) -> float:
        won = self.winner()
        if won != 0:
            return float(won * self.current_player)
        if self.is_terminal():
            return 0.0

        white_material = _material_score(self.board, chess.WHITE)
        black_material = _material_score(self.board, chess.BLACK)
        material_score_white = white_material - black_material

        center_white = sum(1 for square in CENTER_SQUARES if self.board.is_attacked_by(chess.WHITE, square))
        center_black = sum(1 for square in CENTER_SQUARES if self.board.is_attacked_by(chess.BLACK, square))
        center_score_white = (center_white - center_black) * 0.2

        own_mobility = len(self.get_legal_moves())
        swapped = self.board.copy(stack=False)
        swapped.turn = not swapped.turn
        opp_mobility = len(list(swapped.legal_moves))
        mobility_score_white = (own_mobility - opp_mobility) * (0.01 if self.board.turn == chess.WHITE else -0.01)

        score_white = material_score_white + center_score_white + mobility_score_white
        score_current = score_white if self.board.turn == chess.WHITE else -score_white
        return tanh(score_current / 7.0)

    def render(self) -> str:
        return str(self.board)

    def serialize(self) -> str:
        return self.board.fen()

    @staticmethod
    def move_to_label(move: chess.Move) -> str:
        return move.uci()

    @staticmethod
    def label_to_move(label: str) -> chess.Move:
        return chess.Move.from_uci(label)


def _material_score(board: chess.Board, color: chess.Color) -> float:
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, color)) * value
    return score
