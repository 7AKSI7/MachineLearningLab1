"""Предобработка: пропуски, масштабирование, разбиение."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import check_data_quality


# ---------------------------------------------------------- пропуски
def handle_missing(
    df: pd.DataFrame,
    col: str,
    action: str,
    value=None,
) -> pd.DataFrame:
    """
    Обрабатывает пропуски в одном столбце.

    Действия: drop, cast, mean, ffill, bfill, zero, value.
    """
    out = df.copy()

    if action == "drop":
        out = out.dropna(subset=[col])
    elif action == "cast":
        out[col] = pd.to_numeric(out[col], errors="coerce")
    elif action == "mean":
        out[col] = out[col].fillna(out[col].mean())
    elif action == "ffill":
        out[col] = out[col].ffill()
    elif action == "bfill":
        out[col] = out[col].bfill()
    elif action == "zero":
        out[col] = out[col].fillna(0)
    elif action == "value":
        out[col] = out[col].fillna(value)
    else:
        raise ValueError(f"Неизвестное действие: {action}")

    return out


def interactive_menu(df: pd.DataFrame) -> pd.DataFrame:
    """Простенькое консольное меню для ручной чистки данных."""
    current = df
    while True:
        print("\n=== Меню обработки ===")
        print("1. Показать проблемы")
        print("2. Обработать столбец")
        print("3. Выход")
        choice = input("Выбор: ").strip()

        if choice == "1":
            print(check_data_quality(current))
        elif choice == "2":
            col = input("Столбец: ").strip()
            action = input(
                "Действие (drop/cast/mean/ffill/bfill/zero/value): "
            ).strip()
            val = None
            if action == "value":
                val = input("Значение: ").strip()
            current = handle_missing(current, col, action, val)
            print("Готово.")
        elif choice == "3":
            break
    return current


# ------------------------------------------------------------ массивы
def to_numpy(df: pd.DataFrame) -> np.ndarray:
    """DataFrame -> ndarray float64."""
    return df.to_numpy(dtype=np.float64)


# -------------------------------------------------------- масштабирование
class MinMaxScalerCustom:
    """Min-Max масштабирование в диапазон [a, b]."""

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        self.a = float(a)
        self.b = float(b)
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MinMaxScalerCustom":
        x = np.asarray(x, dtype=float)
        self.min_ = np.min(x, axis=0)
        self.max_ = np.max(x, axis=0)
        return self

    def _denom(self) -> np.ndarray:
        span = self.max_ - self.min_
        return np.where(span == 0, 1.0, span)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.a + (x - self.min_) * (self.b - self.a) / self._denom()

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        return (x_new - self.a) * self._denom() / (self.b - self.a) + self.min_


class StandardScalerCustom:
    """Стандартизация: (x - mean) / std."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerCustom":
        x = np.asarray(x, dtype=float)
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0, ddof=0)
        return self

    def _safe_std(self) -> np.ndarray:
        return np.where(self.std_ == 0, 1.0, self.std_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) / self._safe_std()

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        return x_new * self._safe_std() + self.mean_


# ---------------------------------------------------------- разбиение
def split_train_val_test(
    x: np.ndarray,
    ratios,
    percent: bool = False,
):
    """Делит массив на train/val/test без перемешивания (порядок сохраняется)."""
    r = np.asarray(ratios, dtype=float)
    if percent:
        r = r / 100.0
    r = r / r.sum()

    n = len(x)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    n_test = n - n_train - n_val

    return x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]