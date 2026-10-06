"""Проверка качества данных и просмотр таблицы."""

from __future__ import annotations

import pandas as pd


def check_data_quality(df: pd.DataFrame) -> dict:
    """Ищет пропуски, типы и нечисловые значения в object-столбцах."""
    report = {
        "missing": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "non_numeric": {},
    }
    for col in df.columns:
        if df[col].dtype == object:
            as_num = pd.to_numeric(df[col], errors="coerce")
            bad = df.loc[as_num.isna() & df[col].notna(), col]
            if not bad.empty:
                report["non_numeric"][col] = bad.tolist()[:10]
    return report


def show_table(df: pd.DataFrame, n: int = 10) -> None:
    """Печатает первые n строк и общую информацию о таблице."""
    print(df.head(n))
    print(f"\nShape: {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")


def extract_columns(
    df: pd.DataFrame,
    cols: list[str],
    strict: bool = False,
    ) -> pd.DataFrame:
    """
    Возвращает копию df только с указанными столбцами.

    strict=False (по умолчанию) — отсутствующие столбцы игнорируются
    с предупреждением. strict=True — бросает KeyError.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing and strict:
        raise KeyError(f"Нет столбцов: {missing}")
    if missing:
        print(f"Предупреждение: в таблице нет столбцов {missing} — пропускаем.")

    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def cast_types(df: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
    """Приводит столбцы к заданным типам."""
    out = df.copy()
    for col, dtype in type_map.items():
        out[col] = out[col].astype(dtype)
    return out