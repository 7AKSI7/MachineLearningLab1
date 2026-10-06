"""Статистический анализ: графики, корреляции, спектры, интерполяция."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import convolve, periodogram, spectrogram


# --------------------------------------------------------- визуализация
def plot_series(data: np.ndarray, title: str = "Исходные данные") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(data.shape[1]):
        ax.plot(data[:, i], label=f"col_{i}", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Индекс")
    ax.set_ylabel("Значение")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histogram(x: np.ndarray, bins: int = 30) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=bins, density=True, alpha=0.7, edgecolor="black")
    ax.set_title("Нормализованная гистограмма")
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    fig.tight_layout()
    return fig


def plot_ecdf(x: np.ndarray) -> plt.Figure:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots()
    ax.step(xs, ys, where="post")
    ax.set_title("Эмпирическая функция распределения")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    fig.tight_layout()
    return fig


def show_sorted(df: pd.DataFrame) -> None:
    """Печатает отсортированные значения по каждому столбцу."""
    for col in df.columns:
        print(f"\n--- {col} ---")
        print(np.sort(df[col].to_numpy()))


# ---------------------------------------------------------- статистики
def column_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "mode": float(stats.mode(x, keepdims=False).mode),
        "median": float(np.median(x)),
    }


def ci_mean(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(n)
    return stats.t.interval(1 - alpha, df=n - 1, loc=m, scale=se)


def ci_var(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s2 = np.var(x, ddof=1)
    chi2_low = stats.chi2.ppf(alpha / 2, df=n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df=n - 1)
    return (n - 1) * s2 / chi2_high, (n - 1) * s2 / chi2_low


def covariance_correlation(data: np.ndarray):
    """Возвращает (ковариационную, корреляционную) матрицы."""
    data = np.asarray(data, dtype=float)
    return np.cov(data, rowvar=False), np.corrcoef(data, rowvar=False)


def correlation_significance(x: np.ndarray, y: np.ndarray):
    return stats.pearsonr(x, y)


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lags: int = 50):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.arange(-max_lags, max_lags + 1)
    c = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    c = c / (np.std(x) * np.std(y) * len(x))
    mid = len(c) // 2
    return lags, c[mid - max_lags: mid + max_lags + 1]


# ------------------------------------------------------ производные и т.п.
def numerical_gradient(x: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(x, dtype=float))


def convolution(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
    return np.convolve(x, y, mode=mode)


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y))


def cross_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.cross(x[:3], y[:3])


def vector_norms(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "L1": float(np.linalg.norm(x, ord=1)),
        "L2": float(np.linalg.norm(x, ord=2)),
    }


def distribution_tests(x1: np.ndarray, x2: np.ndarray) -> dict:
    """
    Проверяет гипотезы о распределении:
    - x1 против равномерного на [min, max];
    - x2 против нормального с выборочными mean и std;
    - плюс тест Шапиро–Уилка для x2.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # --- Равномерное распределение ---
    lo, hi = float(x1.min()), float(x1.max())
    ks_uniform, p_uniform = stats.kstest(
        x1,
        lambda v: stats.uniform.cdf(v, loc=lo, scale=hi - lo),
    )

    # --- Нормальное распределение ---
    mu, sigma = float(x2.mean()), float(x2.std())
    ks_normal, p_normal = stats.kstest(
        x2,
        lambda v: stats.norm.cdf(v, loc=mu, scale=sigma),
    )

    # --- Шапиро–Уилк ---
    sh_stat, p_shapiro = stats.shapiro(x2)

    return {
        "ks_stat_uniform": float(ks_uniform),
        "p_uniform": float(p_uniform),
        "ks_stat_normal": float(ks_normal),
        "p_normal": float(p_normal),
        "shapiro_stat": float(sh_stat),
        "p_shapiro": float(p_shapiro),
    }


# ------------------------------------------------------------- спектры
def plot_spectrogram(x: np.ndarray) -> plt.Figure:
    f, t, Sxx = spectrogram(x, fs=1.0)
    fig, ax = plt.subplots()
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    ax.set_ylabel("Частота")
    ax.set_xlabel("Время")
    ax.set_title("Спектрограмма")
    fig.tight_layout()
    return fig


def plot_periodogram(x: np.ndarray) -> plt.Figure:
    f, Pxx = periodogram(x, fs=1.0)
    fig, ax = plt.subplots()
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Частота")
    ax.set_ylabel("PSD")
    ax.set_title("Периодограмма")
    fig.tight_layout()
    return fig


def plot_fft_amplitude(x: np.ndarray) -> plt.Figure:
    x = np.asarray(x, dtype=float)
    n = len(x)
    spectrum = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0)
    half = n // 2
    fig, ax = plt.subplots()
    ax.plot(freq[:half], np.abs(spectrum[:half]))
    ax.set_xlabel("Частота")
    ax.set_ylabel("|X(f)|")
    ax.set_title("Амплитудный спектр (FFT)")
    fig.tight_layout()
    return fig


# -------------------------------------------------------- интерполяция
def cubic_interpolation(x: np.ndarray, factor: int = 10):
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    f = interp1d(idx, x, kind="cubic", fill_value="extrapolate")
    x_new = np.linspace(0, len(x) - 1, factor * len(x))
    return x_new, f(x_new)


def spline_interpolation(x: np.ndarray, factor: int = 10):
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    x_new = np.linspace(0, len(x) - 1, factor * len(x))
    spline = UnivariateSpline(idx, x, k=3, s=0)
    return x_new, spline(x_new)


# ------------------------------------------------------------ маски
def binary_masks(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "positive": x > 0,
        "negative": x < 0,
        "zero": x == 0,
        "in_[-1, 1]": (x >= -1) & (x <= 1),
    }