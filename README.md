# AI Game Solver: Hybrid Neural Networks + Alpha-Beta

This project implements a full AI game-solving pipeline for:
- Tic-Tac-Toe
- Connect Four
- Simplified Chess (using python-chess)

Core idea: combine classical alpha-beta search with two learned models:
1. Heuristic evaluator network (position scoring in [-1, 1])
2. Move-ordering network (rank legal moves for stronger pruning)

## Project Layout

- `game_solver/core`: game interface + alpha-beta engine
- `game_solver/games`: Tic-Tac-Toe, Connect Four, simplified Chess states
- `game_solver/data`: feature extraction + data collection/import
- `game_solver/ml`: model definitions, training, transfer-learning experiment
- `game_solver/engine`: hybrid neural + alpha-beta integration
- `game_solver/benchmark`: configuration ablation and metrics
- `game_solver/ui`: CLI and Tkinter explainability GUI
- `scripts`: script entry points
- `artifacts`: generated datasets, models, benchmark outputs

## Setup

Use your existing virtual environment (`.venv`) and install dependencies:

```powershell
& ".venv/Scripts/python.exe" -m pip install -r requirements.txt
```

## How to Run

### Baseline Minimax + Alpha-Beta CLI

```powershell
& ".venv/Scripts/python.exe" scripts/run_cli.py --game tic_tac_toe --depth 5 --human-player 1
& ".venv/Scripts/python.exe" scripts/run_cli.py --game connect_four --depth 4 --human-player 1
& ".venv/Scripts/python.exe" scripts/run_cli.py --game chess --depth 3 --human-player 1
```

### Data Collection + Feature Engineering

Self-play generation:

```powershell
& ".venv/Scripts/python.exe" scripts/collect_data.py --games tic_tac_toe connect_four --num-games 1000 --search-depth 2 --output-dir artifacts/datasets
```

Chess PGN import from a full Lichess PGN/ZST (all game lengths by default):

```powershell
& ".venv/Scripts/python.exe" scripts/collect_data.py --games chess --chess-pgn path/to/lichess_games.pgn.zst --output-dir artifacts/datasets
```

Optional caps if you want a subset:

```powershell
& ".venv/Scripts/python.exe" scripts/collect_data.py --games chess --chess-pgn path/to/lichess_games.pgn.zst --chess-max-games 50000 --chess-max-moves 25 --output-dir artifacts/datasets
```

### Train Two Networks Per Game

```powershell
& ".venv/Scripts/python.exe" scripts/train_models.py --games tic_tac_toe connect_four chess --dataset-dir artifacts/datasets --model-dir artifacts/models
```

By default, chess is refit on the full chess dataset before saving the final models.

### Hybrid Integration Demo

```powershell
& ".venv/Scripts/python.exe" scripts/hybrid_demo.py --game connect_four --depth 4 --model-dir artifacts/models --nn-eval --nn-order
```

### Benchmarking and Comparative Analysis

```powershell
& ".venv/Scripts/python.exe" scripts/run_benchmark.py --games tic_tac_toe connect_four chess --depth 4 --positions-per-game 30 --match-games 20 --model-dir artifacts/models --output-dir artifacts/benchmarks
```

Outputs:
- `artifacts/benchmarks/raw_turn_metrics.csv`
- `artifacts/benchmarks/summary_metrics.csv`
- `artifacts/benchmarks/match_metrics.csv`
- per-game plots (`*_nodes.png`, `*_time.png`)

### Explainability GUI (Tkinter)

```powershell
& ".venv/Scripts/python.exe" scripts/run_gui.py --depth 4 --model-dir artifacts/models --nn-eval --nn-order --human-player 1
```

Shows:
- top-3 candidate moves + scores
- nodes explored and pruned
- depth used
- dominant decision source (heuristic vs ordering)
- unified UI across Tic-Tac-Toe, Connect Four, and Chess with in-game depth selector

### Transfer Learning Experiment

```powershell
& ".venv/Scripts/python.exe" scripts/transfer_learning.py --dataset-dir artifacts/datasets --output-dir artifacts/benchmarks --max-epochs 40
```

Produces transfer-vs-scratch convergence metrics and curve plot.

## Benchmark Configurations

The benchmark runs these four configurations:
- baseline (handcrafted eval + no NN ordering)
- nn_eval (NN eval + no NN ordering)
- nn_order (handcrafted eval + NN ordering)
- full_hybrid (NN eval + NN ordering)

## Notes

- Scripts are executable directly from root (`scripts/...`) and auto-resolve package imports.
- Small datasets are supported with robust split fallbacks in training.
- Chess uses simplified rule handling for consistency with the engine.
