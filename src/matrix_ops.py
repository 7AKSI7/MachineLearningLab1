"""Разреженные матрицы, обратная матрица, массив F, PCA-пайплайн."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .preprocess import MinMaxScalerCustom, StandardScalerCustom


def create_sparse_matrix(
    rows: int = 100_000,
    cols: int = 100_000,
    density: float = 1e-5,
    seed: int = 42,
) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    return sparse.random(rows, cols, density=density, format="csr", random_state=rng)


def inverse_matrix(size: int = 100, seed: int = 42) -> dict:
    """Создаёт квадратную матрицу и, если обратима, считает обратную."""
    rng = np.random.default_rng(seed)
    m = rng.random((size, size))
    det = float(np.linalg.det(m))
    invertible = abs(det) > 1e-10

    result = {
        "matrix": m,
        "det": det,
        "invertible": invertible,
    }

    if invertible:
        m_inv = np.linalg.inv(m)
        result["inverse"] = m_inv
        result["check"] = bool(np.allclose(m @ m_inv, np.eye(size), atol=1e-6))

    return result


def build_F(x: np.ndarray) -> np.ndarray:
    """
    Собирает массив F формы (n, m, 3):
    [исходные данные, Min-Max, StandardScaler].
    """
    x = np.asarray(x, dtype=float)
    x_mm = MinMaxScalerCustom().fit_transform(x)
    x_std = StandardScalerCustom().fit_transform(x)
    return np.stack([x, x_mm, x_std], axis=-1)


def pca_pipeline(f: np.ndarray) -> tuple[dict, str]:
    """
    Прогоняет два пайплайна (MinMax+PCA, Standard+PCA) и возвращает
    словарь log-likelihood и имя лучшего варианта.

    Вместо score_samples (который падает на сингулярной ковариации)
    используется аналитическая формула лог-правдоподобия через
    собственные значения PCA.
    """
    flat = f.reshape(len(f), -1)

    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
    }

    scores: dict[str, float] = {}
    for name, scaler in scalers.items():
        pipe = Pipeline([("scaler", scaler), ("pca", PCA())])
        pipe.fit(flat)

        transformed = pipe.named_steps["scaler"].transform(flat)
        pca = pipe.named_steps["pca"]

        # Логарифм правдоподобия для гауссовой модели в PCA-подпространстве:
        # log L = -0.5 * sum_i [ log(2*pi*lambda_i) + z_i^2 / lambda_i ]
        # где lambda_i — собственные значения, z_i — проекции.
        eigenvalues = pca.explained_variance_       # lambda_i
        Z = pipe.named_steps["pca"].transform(transformed)

        eps = 1e-12
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * np.maximum(eigenvalues, eps))
            + (Z ** 2) / np.maximum(eigenvalues, eps)
        )
        scores[name] = float(log_likelihood)

    best = max(scores, key=scores.get)
    return scores, best


def plot_pca_variance(f: np.ndarray) -> plt.Figure:
    """Строит кумулятивную объяснённую дисперсию в лог-масштабе."""
    flat = f.reshape(len(f), -1)
    pca = PCA().fit(flat)

    fig, ax = plt.subplots()
    ax.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Кумулятивная объяснённая дисперсия (log)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig