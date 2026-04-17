from __future__ import annotations

from dataclasses import dataclass

from game_solver.core.game_state import GameState

WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class TicTacToeState(GameState):
    board: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    current_player: int = 1
    name: str = "tic_tac_toe"

    @staticmethod
    def initial() -> "TicTacToeState":
        return TicTacToeState()

    def get_legal_moves(self) -> list[int]:
        return [idx for idx, value in enumerate(self.board) if value == 0]

    def apply_move(self, move: int) -> "TicTacToeState":
        if move < 0 or move > 8:
            raise ValueError(f"Invalid cell index: {move}")
        if self.board[move] != 0:
            raise ValueError(f"Cell already occupied: {move}")

        next_board = list(self.board)
        next_board[move] = self.current_player
        return TicTacToeState(board=tuple(next_board), current_player=-self.current_player)

    def is_terminal(self) -> bool:
        return self.winner() != 0 or all(value != 0 for value in self.board)

    def winner(self) -> int:
        for a, b, c in WIN_LINES:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return 0

    def handcrafted_eval(self) -> float:
        won = self.winner()
        if won != 0:
            return float(won * self.current_player)
        if self.is_terminal():
            return 0.0

        score = 0.0
        for line in WIN_LINES:
            values = [self.board[idx] for idx in line]
            own = values.count(self.current_player)
            opp = values.count(-self.current_player)

            if own > 0 and opp > 0:
                continue

            if opp == 0:
                score += (0.0, 0.15, 0.45, 1.0)[own]
            if own == 0:
                score -= (0.0, 0.15, 0.45, 1.0)[opp]

        if self.board[4] == self.current_player:
            score += 0.1
        elif self.board[4] == -self.current_player:
            score -= 0.1

        return max(-1.0, min(1.0, score / 2.0))

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for row in range(3):
            values = [symbols[self.board[row * 3 + col]] for col in range(3)]
            rows.append(" ".join(values))
        return "\n".join(rows)

    def serialize(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        board_token = "".join(symbols[value] for value in self.board)
        turn = "X" if self.current_player == 1 else "O"
        return f"{board_token}|{turn}"

    @staticmethod
    def move_to_label(move: int) -> str:
        return str(move)

    @staticmethod
    def label_to_move(label: str) -> int:
        return int(label)
