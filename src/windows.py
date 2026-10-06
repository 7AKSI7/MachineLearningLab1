"""Скользящее окно и скользящее среднее."""

from __future__ import annotations

import numpy as np


def sliding_window(x: np.ndarray, width: int) -> np.ndarray:
    """Матрица окон формы (n - width + 1, width)."""
    x = np.asarray(x)
    n = len(x)
    if width <= 0 or width > n:
        raise ValueError("Некорректная ширина окна")

    idx = np.arange(width)[None, :] + np.arange(n - width + 1)[:, None]
    return x[idx]


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    return sliding_window(x, width).mean(axis=1)