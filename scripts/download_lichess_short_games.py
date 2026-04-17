from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import chess.pgn

DEFAULT_USERS = [
    "DrNykterstein",
    "Hikaru",
    "Firouzja2003",
    "wonderfultime",
    "Jordenvforeest",
    "AnishGiri",
    "Duda",
    "NihalSarin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Lichess games through API and keep only short games (< max full moves)."
    )
    parser.add_argument(
        "--users",
        nargs="+",
        default=DEFAULT_USERS,
        help="Lichess usernames to fetch games from.",
    )
    parser.add_argument(
        "--users-file",
        type=Path,
        default=None,
        help="Optional text file with one Lichess username per line.",
    )
    parser.add_argument(
        "--max-per-user",
        type=int,
        default=400,
        help="Max games to fetch per user from API.",
    )
    parser.add_argument(
        "--target-games",
        type=int,
        default=5000,
        help="Stop once this many filtered games are collected.",
    )
    parser.add_argument(
        "--max-full-moves",
        type=int,
        default=25,
        help="Keep games with full moves strictly below this value.",
    )
    parser.add_argument(
        "--perf-type",
        type=str,
        default="rapid,blitz,classical",
        help="Comma-separated perf types accepted by Lichess API (example: blitz,rapid).",
    )
    parser.add_argument(
        "--rated-only",
        action="store_true",
        help="Fetch rated games only.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries for transient HTTP failures.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.8,
        help="Delay between user requests to be API-friendly.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Lichess API token for higher request limits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "datasets" / "lichess_short_lt25.pgn",
        help="Output PGN path.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("artifacts") / "datasets" / "lichess_short_lt25_stats.json",
        help="Output path for download/filter stats JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    users = dedupe_users(load_users(args.users, args.users_file))
    if not users:
        raise SystemExit("No users provided. Use --users or --users-file.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    kept_games = 0
    fetched_games = 0
    per_user_stats: dict[str, dict[str, int]] = {}
    failures: dict[str, str] = {}

    with args.output.open("w", encoding="utf-8", newline="\n") as out_handle:
        for user in users:
            if kept_games >= args.target_games:
                break

            try:
                pgn_blob = fetch_user_pgn(
                    username=user,
                    max_games=args.max_per_user,
                    perf_type=args.perf_type,
                    rated_only=args.rated_only,
                    token=args.token,
                    max_retries=args.max_retries,
                )
            except Exception as exc:
                failures[user] = str(exc)
                per_user_stats[user] = {"fetched": 0, "kept": 0}
                print(f"[{user}] failed: {exc}")
                time.sleep(max(0.0, args.sleep_seconds))
                continue

            total_for_user = 0
            kept_for_user = 0

            stream = io.StringIO(pgn_blob)
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break

                total_for_user += 1
                fetched_games += 1

                full_moves = full_move_count(game)
                if full_moves >= args.max_full_moves:
                    continue

                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                out_handle.write(game.accept(exporter))
                out_handle.write("\n\n")

                kept_for_user += 1
                kept_games += 1

                if kept_games >= args.target_games:
                    break

            per_user_stats[user] = {
                "fetched": total_for_user,
                "kept": kept_for_user,
            }

            print(f"[{user}] fetched={total_for_user} kept_lt_{args.max_full_moves}={kept_for_user}")

            if kept_games >= args.target_games:
                break

            time.sleep(max(0.0, args.sleep_seconds))

    stats = {
        "users": users,
        "max_per_user": args.max_per_user,
        "target_games": args.target_games,
        "max_full_moves_exclusive": args.max_full_moves,
        "fetched_games": fetched_games,
        "kept_games": kept_games,
        "failures": failures,
        "output_pgn": str(args.output),
        "per_user": per_user_stats,
    }

    with args.stats_output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("\nDone")
    print(f"Output PGN: {args.output}")
    print(f"Stats JSON: {args.stats_output}")
    print(f"Kept {kept_games} short games (full moves < {args.max_full_moves})")
    if failures:
        print(f"Users failed: {len(failures)} (see stats JSON)")
    if kept_games == 0:
        print("No games were collected. Check internet access, usernames, or API limits.")


def fetch_user_pgn(
    username: str,
    max_games: int,
    perf_type: str,
    rated_only: bool,
    token: str | None,
    max_retries: int,
) -> str:
    params = {
        "max": str(max_games),
        "moves": "true",
        "pgnInJson": "false",
        "perfType": perf_type,
        "sort": "dateDesc",
    }
    if rated_only:
        params["rated"] = "true"

    base = f"https://lichess.org/api/games/user/{username}"
    url = f"{base}?{urlencode(params)}"

    headers = {
        "Accept": "application/x-chess-pgn",
        "User-Agent": "ai-game-solver-short-games-downloader/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    retries = 0
    while True:
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=60) as response:
                body = response.read()
            return body.decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 429 and retries < max_retries:
                retries += 1
                wait_s = 60
                print(f"[{username}] rate limited (429), waiting {wait_s}s and retry {retries}/{max_retries}")
                time.sleep(wait_s)
                continue
            if 500 <= exc.code < 600 and retries < max_retries:
                retries += 1
                wait_s = 2 * retries
                print(f"[{username}] server error {exc.code}, retry {retries}/{max_retries} after {wait_s}s")
                time.sleep(wait_s)
                continue
            raise
        except URLError as exc:
            if retries < max_retries:
                retries += 1
                wait_s = 2 * retries
                print(f"[{username}] network error {exc}, retry {retries}/{max_retries} after {wait_s}s")
                time.sleep(wait_s)
                continue
            raise


def full_move_count(game: chess.pgn.Game) -> int:
    plies = 0
    for _move in game.mainline_moves():
        plies += 1
    return (plies + 1) // 2


def load_users(cli_users: list[str], users_file: Path | None) -> list[str]:
    users = list(cli_users)
    if users_file is not None:
        if not users_file.exists():
            raise FileNotFoundError(f"users file not found: {users_file}")
        lines = users_file.read_text(encoding="utf-8").splitlines()
        users.extend(line.strip() for line in lines if line.strip())
    return users


def dedupe_users(users: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for user in users:
        key = user.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(user)
    return out


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
