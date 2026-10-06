"""Чтение/запись табличных данных и сохранение картинок."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from .config import timestamped_filename


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает таблицу из xlsx / csv / txt / mat."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".xlsx":
        df = pd.read_excel(p)
    elif suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix == ".txt":
        df = pd.read_csv(p, sep=r"\s+", engine="python")
    elif suffix == ".mat":
        raw = loadmat(p)
        arrays = {k: v for k, v in raw.items() if not k.startswith("__")}
        if not arrays:
            raise ValueError(f"В файле {p} нет массивов данных.")
        first_key = next(iter(arrays))
        df = pd.DataFrame(arrays[first_key])
    else:
        raise ValueError(f"Формат '{suffix}' не поддерживается.")

    # Если у столбцов числовые индексы — значит, заголовков не было.
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def write_table(df: pd.DataFrame, base_name: str) -> dict[str, str]:
    """Сохраняет DataFrame в xlsx / csv / txt / mat с меткой времени."""
    saved: dict[str, str] = {}

    saved["xlsx"] = timestamped_filename(base_name, "xlsx")
    df.to_excel(saved["xlsx"], index=False, engine="openpyxl")

    saved["csv"] = timestamped_filename(base_name, "csv")
    df.to_csv(saved["csv"], index=False)

    saved["txt"] = timestamped_filename(base_name, "txt")
    df.to_csv(saved["txt"], sep="\t", index=False)

    saved["mat"] = timestamped_filename(base_name, "mat")
    savemat(
        saved["mat"],
        {
            "data": df.to_numpy(),
            "columns": np.array(df.columns, dtype=object),
        },
    )
    return saved