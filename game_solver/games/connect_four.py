from __future__ import annotations

from dataclasses import dataclass

from game_solver.core.game_state import GameState

ROWS = 6
COLS = 7


@dataclass(frozen=True)
class ConnectFourState(GameState):
    board: tuple[tuple[int, ...], ...] = (
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0),
    )
    current_player: int = 1
    name: str = "connect_four"

    @staticmethod
    def initial() -> "ConnectFourState":
        return ConnectFourState()

    def get_legal_moves(self) -> list[int]:
        return [col for col in range(COLS) if self.board[0][col] == 0]

    def apply_move(self, move: int) -> "ConnectFourState":
        if move < 0 or move >= COLS:
            raise ValueError(f"Invalid column: {move}")
        if self.board[0][move] != 0:
            raise ValueError(f"Column is full: {move}")

        next_board = [list(row) for row in self.board]
        for row in range(ROWS - 1, -1, -1):
            if next_board[row][move] == 0:
                next_board[row][move] = self.current_player
                break

        return ConnectFourState(board=tuple(tuple(row) for row in next_board), current_player=-self.current_player)

    def is_terminal(self) -> bool:
        return self.winner() != 0 or not self.get_legal_moves()

    def winner(self) -> int:
        board = self.board

        for row in range(ROWS):
            for col in range(COLS):
                player = board[row][col]
                if player == 0:
                    continue

                if col <= COLS - 4 and all(board[row][col + offset] == player for offset in range(4)):
                    return player
                if row <= ROWS - 4 and all(board[row + offset][col] == player for offset in range(4)):
                    return player
                if row <= ROWS - 4 and col <= COLS - 4 and all(board[row + offset][col + offset] == player for offset in range(4)):
                    return player
                if row <= ROWS - 4 and col >= 3 and all(board[row + offset][col - offset] == player for offset in range(4)):
                    return player

        return 0

    def handcrafted_eval(self) -> float:
        won = self.winner()
        if won != 0:
            return float(won * self.current_player)
        if self.is_terminal():
            return 0.0

        score = 0.0

        center_col = COLS // 2
        center_values = [self.board[row][center_col] for row in range(ROWS)]
        score += 0.07 * center_values.count(self.current_player)
        score -= 0.07 * center_values.count(-self.current_player)

        for row in range(ROWS):
            for col in range(COLS - 3):
                window = [self.board[row][col + offset] for offset in range(4)]
                score += _window_score(window, self.current_player)
                score -= _window_score(window, -self.current_player)

        for row in range(ROWS - 3):
            for col in range(COLS):
                window = [self.board[row + offset][col] for offset in range(4)]
                score += _window_score(window, self.current_player)
                score -= _window_score(window, -self.current_player)

        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                window = [self.board[row + offset][col + offset] for offset in range(4)]
                score += _window_score(window, self.current_player)
                score -= _window_score(window, -self.current_player)

        for row in range(ROWS - 3):
            for col in range(3, COLS):
                window = [self.board[row + offset][col - offset] for offset in range(4)]
                score += _window_score(window, self.current_player)
                score -= _window_score(window, -self.current_player)

        return max(-1.0, min(1.0, score / 4.0))

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for row in self.board:
            rows.append(" ".join(symbols[value] for value in row))
        rows.append("0 1 2 3 4 5 6")
        return "\n".join(rows)

    def serialize(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        board_token = "".join(symbols[self.board[row][col]] for row in range(ROWS) for col in range(COLS))
        turn = "X" if self.current_player == 1 else "O"
        return f"{board_token}|{turn}"

    @staticmethod
    def move_to_label(move: int) -> str:
        return str(move)

    @staticmethod
    def label_to_move(label: str) -> int:
        return int(label)


def _window_score(window: list[int], player: int) -> float:
    own = window.count(player)
    empty = window.count(0)
    opp = window.count(-player)

    if opp > 0 and own > 0:
        return 0.0
    if own == 4:
        return 1.0
    if own == 3 and empty == 1:
        return 0.4
    if own == 2 and empty == 2:
        return 0.15
    if own == 1 and empty == 3:
        return 0.05
    return 0.0
