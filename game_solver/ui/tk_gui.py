from __future__ import annotations

from typing import Any

import chess
import tkinter as tk
from tkinter import messagebox, ttk

from game_solver.engine.hybrid_engine import HybridEngine, HybridEngineConfig
from game_solver.games.chess_simplified import ChessSimplifiedState
from game_solver.games.connect_four import ConnectFourState
from game_solver.games.tic_tac_toe import TicTacToeState


TOP_BG = "#1c2b45"
BOARD_BG = "#0f1c34"
PANEL_BG = "#e4e8ef"
SURFACE_BG = "#c9cdd3"


class MainMenuApp:
    def __init__(
        self,
        model_dir: str = "artifacts/models",
        default_depth: int = 4,
        default_human_player: int = 1,
        default_nn_eval: bool = True,
        default_nn_order: bool = True,
    ) -> None:
        self.model_dir = model_dir

        self.root = tk.Tk()
        self.root.title("AI Game Solver - Explainable GUI")
        self.root.geometry("1120x780")
        self.root.configure(bg=SURFACE_BG)

        self.game_var = tk.StringVar(value="Tic-Tac-Toe")
        self.depth_var = tk.IntVar(value=max(1, min(8, int(default_depth))))
        side_default = "First (White/X)" if default_human_player == 1 else "Second (Black/O)"
        self.human_side_var = tk.StringVar(value=side_default)
        self.nn_eval_var = tk.BooleanVar(value=default_nn_eval)
        self.nn_order_var = tk.BooleanVar(value=default_nn_order)

        self.status_var = tk.StringVar(value="Start a game to play.")
        self.hint_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="No AI move yet.")
        self.top3_var = tk.StringVar(value="Top moves:\n-")

        self.current_game_key = "tic_tac_toe"
        self.state: Any = None
        self.engine: HybridEngine | None = None

        self.ttt_buttons: list[tk.Button] = []
        self.c4_cells: list[tuple[int, int, tk.Button]] = []
        self.chess_labels: dict[tuple[int, int], tk.Label] = {}

        self._build_layout()
        self.start_new_game()

    def _build_layout(self) -> None:
        top_bar = tk.Frame(self.root, bg=TOP_BG, padx=14, pady=10)
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text="Game:", fg="white", bg=TOP_BG, font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)

        game_combo = ttk.Combobox(
            top_bar,
            textvariable=self.game_var,
            values=["Tic-Tac-Toe", "Connect Four", "Chess"],
            state="readonly",
            width=16,
        )
        game_combo.pack(side=tk.LEFT, padx=(8, 16))
        game_combo.bind("<<ComboboxSelected>>", self._on_game_change)

        tk.Label(top_bar, text="Depth:", fg="white", bg=TOP_BG, font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
        tk.Spinbox(
            top_bar,
            from_=1,
            to=10,
            textvariable=self.depth_var,
            width=4,
            justify="center",
            font=("Segoe UI", 11),
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Button(
            top_bar,
            text="New Game",
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=18,
            pady=4,
            relief=tk.FLAT,
            command=self.start_new_game,
        ).pack(side=tk.RIGHT)

        content = tk.Frame(self.root, bg=SURFACE_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        left = tk.Frame(content, bg=SURFACE_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            left,
            textvariable=self.status_var,
            bg=SURFACE_BG,
            fg="#0f172a",
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 16, "bold"),
        ).pack(fill=tk.X, pady=(0, 10))

        self.board_shell = tk.Frame(left, bg=BOARD_BG, bd=1, relief=tk.SOLID)
        self.board_shell.pack(padx=8, pady=(0, 10), anchor="w")
        self.board_frame = tk.Frame(self.board_shell, bg=BOARD_BG)
        self.board_frame.pack(padx=10, pady=10)

        hint_label = tk.Label(
            left,
            textvariable=self.hint_var,
            bg=SURFACE_BG,
            fg="#5b6780",
            anchor="w",
            justify=tk.LEFT,
            font=("Segoe UI", 11),
        )
        hint_label.pack(fill=tk.X, pady=(4, 10))

        move_row = tk.Frame(left, bg=SURFACE_BG)
        move_row.pack(anchor="w", padx=8)

        tk.Label(move_row, text="Your Move:", bg=SURFACE_BG, fg="#0f172a", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        self.move_entry = tk.Entry(move_row, width=18, font=("Segoe UI", 14))
        self.move_entry.pack(side=tk.LEFT, padx=8)
        self.move_entry.bind("<Return>", self._on_submit)

        tk.Button(
            move_row,
            text="Submit",
            font=("Segoe UI", 12, "bold"),
            bg="#2563eb",
            fg="white",
            relief=tk.FLAT,
            padx=18,
            command=self._on_submit,
        ).pack(side=tk.LEFT)

        right = tk.LabelFrame(
            content,
            text="AI Brain Panel",
            bg=PANEL_BG,
            fg="#0f172a",
            font=("Segoe UI", 16, "bold"),
            padx=14,
            pady=12,
        )
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(16, 0))

        options = tk.LabelFrame(
            right,
            text="Game Options",
            bg=PANEL_BG,
            fg="#334155",
            font=("Segoe UI", 11, "bold"),
            padx=8,
            pady=8,
        )
        options.pack(fill=tk.X, pady=(0, 10))

        tk.Label(options, text="You Play As", bg=PANEL_BG, anchor="w", font=("Segoe UI", 10, "bold")).pack(fill=tk.X)
        ttk.Combobox(
            options,
            textvariable=self.human_side_var,
            values=["First (White/X)", "Second (Black/O)"],
            state="readonly",
            width=18,
        ).pack(anchor="w", pady=(2, 6))

        tk.Checkbutton(
            options,
            text="Use NN Evaluator",
            variable=self.nn_eval_var,
            bg=PANEL_BG,
            command=self._reload_engine,
        ).pack(anchor="w")
        tk.Checkbutton(
            options,
            text="Use NN Ordering",
            variable=self.nn_order_var,
            bg=PANEL_BG,
            command=self._reload_engine,
        ).pack(anchor="w")

        tk.Label(
            right,
            text="AI Decision Stats",
            bg=PANEL_BG,
            fg="#334155",
            anchor="w",
            font=("Consolas", 12, "bold"),
        ).pack(fill=tk.X, pady=(8, 0))

        tk.Label(
            right,
            textvariable=self.stats_var,
            bg=PANEL_BG,
            fg="#334155",
            justify=tk.LEFT,
            anchor="nw",
            font=("Consolas", 12),
        ).pack(fill=tk.X, pady=(4, 8))

        tk.Label(
            right,
            textvariable=self.top3_var,
            bg=PANEL_BG,
            fg="#0f172a",
            justify=tk.LEFT,
            anchor="nw",
            font=("Consolas", 11),
        ).pack(fill=tk.BOTH, expand=True)

    def _on_game_change(self, _event: Any) -> None:
        self.start_new_game()

    def _selected_game_key(self) -> str:
        display = self.game_var.get().strip().lower()
        if display == "connect four":
            return "connect_four"
        if display == "chess":
            return "chess"
        return "tic_tac_toe"

    def _human_player(self) -> int:
        return 1 if self.human_side_var.get().startswith("First") else -1

    def _depth_value(self) -> int:
        try:
            depth = int(self.depth_var.get())
        except (TypeError, ValueError):
            depth = 4
        depth = max(1, min(10, depth))
        self.depth_var.set(depth)
        return depth

    def _reload_engine(self) -> None:
        if self.current_game_key:
            self.engine = HybridEngine.from_model_dir(
                game_name=self.current_game_key,
                model_dir=self.model_dir,
                config=HybridEngineConfig(
                    use_nn_eval=self.nn_eval_var.get(),
                    use_nn_order=self.nn_order_var.get(),
                ),
            )

    def start_new_game(self) -> None:
        self.current_game_key = self._selected_game_key()

        if self.current_game_key == "tic_tac_toe":
            self.state = TicTacToeState.initial()
        elif self.current_game_key == "connect_four":
            self.state = ConnectFourState.initial()
        else:
            self.state = ChessSimplifiedState.initial()

        self._reload_engine()
        self._build_board_widgets()
        self._refresh_board()
        self.move_entry.delete(0, tk.END)

        human_symbol, ai_symbol = self._player_tokens()
        if self.current_game_key == "tic_tac_toe":
            self.hint_var.set(
                f"INPUT: Type 0-8. Layout: 0|1|2 / 3|4|5 / 6|7|8. You are {human_symbol}, AI is {ai_symbol}"
            )
        elif self.current_game_key == "connect_four":
            self.hint_var.set(
                f"INPUT: Type 0-6 for column. You are {human_symbol}, AI is {ai_symbol}"
            )
        else:
            self.hint_var.set(
                f"INPUT: Enter UCI like e2e4, g1f3. You are {human_symbol}, AI is {ai_symbol}"
            )

        self.status_var.set("Your turn! Type your move below." if self.state.current_player == self._human_player() else "AI is thinking...")
        self.stats_var.set("No AI move yet.")
        self.top3_var.set("Top moves:\n-")

        if self.state.current_player != self._human_player():
            self.root.after(80, self._ai_turn)

    def _player_tokens(self) -> tuple[str, str]:
        human_first = self._human_player() == 1
        if self.current_game_key == "chess":
            return ("White", "Black") if human_first else ("Black", "White")
        return ("X", "O") if human_first else ("O", "X")

    def _build_board_widgets(self) -> None:
        for child in self.board_frame.winfo_children():
            child.destroy()

        self.ttt_buttons = []
        self.c4_cells = []
        self.chess_labels = {}

        if self.current_game_key == "tic_tac_toe":
            for row in range(3):
                for col in range(3):
                    idx = row * 3 + col
                    btn = tk.Button(
                        self.board_frame,
                        width=6,
                        height=3,
                        relief=tk.RIDGE,
                        bd=1,
                        font=("Segoe UI", 26, "bold"),
                        bg="#1f2f4d",
                        activebackground="#2f4570",
                        command=lambda move=idx: self._on_board_click(move),
                    )
                    btn.grid(row=row, column=col, padx=2, pady=2)
                    self.ttt_buttons.append(btn)
            return

        if self.current_game_key == "connect_four":
            for row in range(6):
                for col in range(7):
                    btn = tk.Button(
                        self.board_frame,
                        width=4,
                        height=2,
                        relief=tk.RIDGE,
                        bd=1,
                        font=("Segoe UI", 15, "bold"),
                        bg="#10243f",
                        activebackground="#1d3a63",
                        command=lambda move=col: self._on_board_click(move),
                    )
                    btn.grid(row=row, column=col, padx=2, pady=2)
                    self.c4_cells.append((row, col, btn))

            for col in range(7):
                tk.Label(
                    self.board_frame,
                    text=str(col),
                    bg=BOARD_BG,
                    fg="#a3b3cf",
                    font=("Segoe UI", 10, "bold"),
                ).grid(row=6, column=col, pady=(4, 0))
            return

        for rank in range(8):
            for file_idx in range(8):
                light = (rank + file_idx) % 2 != 0
                bg = "#f0d9b5" if light else "#b58863"
                label = tk.Label(self.board_frame, text=" ", bg=bg, width=3, height=1, font=("Segoe UI Symbol", 26))
                label.grid(row=7 - rank, column=file_idx)
                self.chess_labels[(rank, file_idx)] = label

        files_row = tk.Frame(self.board_frame, bg=BOARD_BG)
        files_row.grid(row=8, column=0, columnspan=8, sticky="ew", pady=(4, 0))
        for file_idx, char in enumerate("abcdefgh"):
            tk.Label(files_row, text=char, width=4, bg=BOARD_BG, fg="#a3b3cf", font=("Segoe UI", 9, "bold")).grid(row=0, column=file_idx)

    def _refresh_board(self) -> None:
        if self.current_game_key == "tic_tac_toe":
            symbols = {1: ("X", "#ff5757"), -1: ("O", "#5cc8ff"), 0: ("", "#7b8fb5")}
            for idx, btn in enumerate(self.ttt_buttons):
                token = self.state.board[idx]
                text, color = symbols[token]
                if token == 0:
                    btn.config(text=str(idx), fg="#42587e")
                else:
                    btn.config(text=text, fg=color)
            return

        if self.current_game_key == "connect_four":
            for row, col, btn in self.c4_cells:
                token = self.state.board[row][col]
                if token == 1:
                    btn.config(text="●", fg="#ff5d5d")
                elif token == -1:
                    btn.config(text="●", fg="#ffd166")
                else:
                    btn.config(text="·", fg="#38547e")
            return

        black_symbols = {
            chess.PAWN: "♟",
            chess.KNIGHT: "♞",
            chess.BISHOP: "♝",
            chess.ROOK: "♜",
            chess.QUEEN: "♛",
            chess.KING: "♚",
        }
        white_symbols = {
            chess.PAWN: "♙",
            chess.KNIGHT: "♘",
            chess.BISHOP: "♗",
            chess.ROOK: "♖",
            chess.QUEEN: "♕",
            chess.KING: "♔",
        }

        for rank in range(8):
            for file_idx in range(8):
                square = chess.square(file_idx, rank)
                piece = self.state.board.piece_at(square)
                if piece is None:
                    self.chess_labels[(rank, file_idx)].config(text=" ")
                elif piece.color == chess.WHITE:
                    self.chess_labels[(rank, file_idx)].config(text=white_symbols[piece.piece_type])
                else:
                    self.chess_labels[(rank, file_idx)].config(text=black_symbols[piece.piece_type])

    def _on_board_click(self, move: Any) -> None:
        self.move_entry.delete(0, tk.END)
        self.move_entry.insert(0, self._format_move(move))
        self._apply_human_move(move)

    def _on_submit(self, _event: Any = None) -> None:
        raw = self.move_entry.get().strip()
        if not raw:
            return

        try:
            move = self._parse_move(raw)
        except ValueError as exc:
            messagebox.showerror("Invalid Move", str(exc), parent=self.root)
            return

        self._apply_human_move(move)

    def _parse_move(self, raw: str) -> Any:
        if self.current_game_key == "tic_tac_toe":
            if not raw.isdigit():
                raise ValueError("Tic-Tac-Toe move must be a number from 0 to 8.")
            move = int(raw)
            if move < 0 or move > 8:
                raise ValueError("Tic-Tac-Toe move must be between 0 and 8.")
            return move

        if self.current_game_key == "connect_four":
            if not raw.isdigit():
                raise ValueError("Connect Four move must be a column number from 0 to 6.")
            move = int(raw)
            if move < 0 or move > 6:
                raise ValueError("Connect Four column must be between 0 and 6.")
            return move

        token = raw.lower()
        try:
            return chess.Move.from_uci(token)
        except ValueError as exc:
            raise ValueError("Chess move must be UCI format, for example: e2e4") from exc

    def _apply_human_move(self, move: Any) -> None:
        if self.state.is_terminal():
            return

        if self.state.current_player != self._human_player():
            self.status_var.set("AI is thinking...")
            return

        legal_moves = self.state.get_legal_moves()
        if move not in legal_moves:
            messagebox.showerror(
                "Illegal Move",
                f"{self._format_move(move)} is not legal for the current position.",
                parent=self.root,
            )
            return

        self.move_entry.delete(0, tk.END)
        self.state = self.state.apply_move(move)
        self._refresh_board()

        if self.state.is_terminal():
            self._finish_game()
            return

        self.status_var.set("AI is thinking...")
        self.root.after(40, self._ai_turn)

    def _ai_turn(self) -> None:
        if self.state.is_terminal() or self.state.current_player == self._human_player() or self.engine is None:
            return

        depth = self._depth_value()
        result = self.engine.search(state=self.state, depth=depth)
        if result.best_move is None:
            self.status_var.set("No legal move found for AI.")
            return

        self._set_ai_panel(result, depth)
        self.state = self.state.apply_move(result.best_move)
        self._refresh_board()

        if self.state.is_terminal():
            self._finish_game()
        else:
            self.status_var.set("Your turn! Type your move below.")

    def _set_ai_panel(self, result: Any, requested_depth: int) -> None:
        source = self.engine.decision_source(result).replace("_", " ") if self.engine is not None else "unknown"
        best_move_text = self._format_move(result.best_move)
        self.stats_var.set(
            "\n".join(
                [
                    f"Depth Requested : {requested_depth}",
                    f"Depth Reached   : {result.stats.max_depth_reached}",
                    f"Nodes Explored  : {result.stats.nodes_explored}",
                    f"Pruned Branches : {result.stats.pruned_branches}",
                    f"Pruning Eff (%) : {result.stats.pruning_efficiency:.1f}",
                    f"Move Chosen     : {best_move_text}",
                    f"Position Score  : {result.score:.4f}",
                    f"Decision Time   : {result.stats.elapsed_ms:.2f} ms",
                    "",
                    f"[{source} mode]",
                ]
            )
        )

        ranked = result.ranked_moves[:3]
        if not ranked:
            self.top3_var.set("Top moves:\n-")
            return

        lines = [f"{idx + 1}. {self._format_move(move)} ({score:.4f})" for idx, (move, score) in enumerate(ranked)]
        self.top3_var.set("Top moves:\n" + "\n".join(lines))

    def _format_move(self, move: Any) -> str:
        if move is None:
            return "-"
        if self.current_game_key == "chess" and hasattr(move, "uci"):
            return move.uci()
        return str(move)

    def _finish_game(self) -> None:
        winner = self.state.winner()
        if winner == 0:
            message = "Draw"
        elif winner == self._human_player():
            message = "You win"
        else:
            message = "AI wins"

        self.status_var.set(f"Game ended: {message}")
        messagebox.showinfo("Result", message, parent=self.root)

    def run(self) -> None:
        self.root.mainloop()
