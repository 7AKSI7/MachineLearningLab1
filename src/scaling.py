"""Сравнение собственного масштабирования с sklearn."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .preprocess import MinMaxScalerCustom, StandardScalerCustom


def compare_scalers(x: np.ndarray) -> dict[str, np.ndarray]:
    """Возвращает массивы после 4 вариантов масштабирования."""
    x = np.asarray(x, dtype=float)

    custom_mm = MinMaxScalerCustom().fit(x)
    sk_mm = MinMaxScaler().fit(x)

    custom_std = StandardScalerCustom().fit(x)
    sk_std = StandardScaler().fit(x)

    return {
        "custom_minmax": custom_mm.transform(x),
        "sklearn_minmax": sk_mm.transform(x),
        "custom_standard": custom_std.transform(x),
        "sklearn_standard": sk_std.transform(x),
    }


def check_inverse_transform(x: np.ndarray) -> dict[str, bool]:
    """Проверяет, что inverse_transform возвращает исходные данные."""
    x = np.asarray(x, dtype=float)

    custom_mm = MinMaxScalerCustom().fit(x)
    sk_mm = MinMaxScaler().fit(x)

    x_custom = custom_mm.transform(x)
    x_sk = sk_mm.transform(x)

    custom_ok = np.allclose(x, custom_mm.inverse_transform(x_custom), atol=1e-10)
    sklearn_ok = np.allclose(x, sk_mm.inverse_transform(x_sk), atol=1e-10)

    return {"custom_ok": bool(custom_ok), "sklearn_ok": bool(sklearn_ok)}