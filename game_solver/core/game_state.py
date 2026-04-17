from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

Move = Any


class GameState(ABC):
    """Common interface used by minimax, data collection, and benchmarking."""

    name: str
    current_player: int

    @abstractmethod
    def get_legal_moves(self) -> list[Move]:
        raise NotImplementedError

    @abstractmethod
    def apply_move(self, move: Move) -> "GameState":
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def winner(self) -> int:
        """Return 1 for player-1 win, -1 for player-2 win, and 0 for draw/ongoing."""
        raise NotImplementedError

    @abstractmethod
    def handcrafted_eval(self) -> float:
        """Return score from the perspective of current_player."""
        raise NotImplementedError

    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> str:
        raise NotImplementedError
