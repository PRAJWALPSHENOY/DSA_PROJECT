from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("f")]
    cols.sort(key=lambda token: int(token[1:]))
    return cols
