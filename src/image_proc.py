"""
Задание 50. Обработка изображения средствами SciPy.

Каждая операция:
  1. строит сравнительную фигуру (до/после) — task50_<name>_compare.png;
  2. сохраняет обработанное изображение отдельным PNG — task50_<name>.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from skimage.color import rgb2gray
from sklearn.decomposition import PCA

from .config import OUTPUT_DIR, save_figure


# ============================================================
# ВСПОМОГАТЕЛЬНОЕ: сохранение одиночного изображения
# ============================================================

def save_image(array: np.ndarray, name: str) -> str:
    """
    Сохраняет одно изображение (2D или 3D) как PNG без осей.
    Возвращает путь к файлу.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if array.ndim == 2:
        ax.imshow(array, cmap="gray")
    else:
        ax.imshow(array)
    ax.axis("off")
    fig.tight_layout(pad=0)

    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return str(path)


# ============================================================
# 1. РАЗДЕЛЕНИЕ ЦВЕТОВЫХ КАНАЛОВ
# ============================================================

def split_channels(img: np.ndarray) -> dict[str, np.ndarray]:
    """Разделяет цветное изображение на R, G, B."""
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Ожидается цветное изображение (H, W, 3)")

    r, g, b = img[..., 0].copy(), img[..., 1].copy(), img[..., 2].copy()

    # сравнительная фигура
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img);                axes[0].set_title("Исходное")
    axes[1].imshow(r, cmap="Reds");     axes[1].set_title("Красный канал")
    axes[2].imshow(g, cmap="Greens");   axes[2].set_title("Зелёный канал")
    axes[3].imshow(b, cmap="Blues");    axes[3].set_title("Синий канал")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_channels_compare")
    plt.close(fig)

    # отдельные PNG каналов
    save_image(r, "task50_channel_r")
    save_image(g, "task50_channel_g")
    save_image(b, "task50_channel_b")

    return {"r": r, "g": g, "b": b}


# ============================================================
# 2. ПОВОРОТ
# ============================================================

def rotate_image(img: np.ndarray, angle: float = 40.0) -> np.ndarray:
    """Поворот изображения на заданный угол."""
    rotated = rotate(
        img.astype(np.float32),
        angle=angle,
        reshape=True,
        order=3,
        mode="constant",
        cval=0,
    )
    rotated = np.clip(rotated, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);      axes[0].set_title("Исходное")
    axes[1].imshow(rotated);  axes[1].set_title(f"Поворот на {angle}°")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_rotated_compare")
    plt.close(fig)

    save_image(rotated, "task50_rotated")
    return rotated


# ============================================================
# 3. ЧЁРНО-БЕЛОЕ ИЗОБРАЖЕНИЕ
# ============================================================

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Перевод в оттенки серого через skimage."""
    gray = rgb2gray(img)
    gray_u8 = (gray * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);                  axes[0].set_title("Исходное")
    axes[1].imshow(gray_u8, cmap="gray"); axes[1].set_title("Ч/б")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_gray_compare")
    plt.close(fig)

    save_image(gray_u8, "task50_gray")
    return gray_u8


# ============================================================
# 4. ГИСТОГРАММА
# ============================================================

def gray_histogram(gray: np.ndarray, bins: int = 256) -> dict:
    """Гистограмма яркостей ч/б изображения."""
    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=1, color="black", align="edge")
    ax.set_title("Гистограмма ч/б изображения")
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Частота")
    ax.set_xlim(0, 255)
    fig.tight_layout()
    save_figure(fig, "task50_gray_hist")
    plt.close(fig)

    return {"hist": hist, "bin_edges": bin_edges}


# ============================================================
# 5. РАЗБИЕНИЕ НА 3 ЧАСТИ ПО ГИСТОГРАММЕ
# ============================================================

def split_by_histogram(gray: np.ndarray, hist_info: dict):
    """Делит изображение на 3 части по квантилям гистограммы."""
    hist = hist_info["hist"]
    edges = hist_info["bin_edges"]

    cum = np.cumsum(hist)
    total = cum[-1]

    t1 = edges[np.searchsorted(cum, total * 0.33)]
    t2 = edges[np.searchsorted(cum, total * 0.66)]

    part1 = np.where(gray <= t1, gray, 0).astype(np.uint8)
    part2 = np.where((gray > t1) & (gray <= t2), gray, 0).astype(np.uint8)
    part3 = np.where(gray > t2, gray, 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(part1, cmap="gray"); axes[0].set_title(f"Тёмные (≤ {t1})")
    axes[1].imshow(part2, cmap="gray"); axes[1].set_title(f"Средние ({t1}–{t2})")
    axes[2].imshow(part3, cmap="gray"); axes[2].set_title(f"Светлые (> {t2})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_split3_compare")
    plt.close(fig)

    save_image(part1, "task50_split3_dark")
    save_image(part2, "task50_split3_mid")
    save_image(part3, "task50_split3_light")

    return part1, part2, part3


# ============================================================
# 6. КРУГОВАЯ РАМКА
# ============================================================

def add_circular_frame(img: np.ndarray) -> np.ndarray:
    """Зануляет всё, что вне круга, вписанного в изображение."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    radius = min(h, w) / 2.0 - 2

    yy, xx = np.ogrid[:h, :w]
    outside = (xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2

    framed = img.copy()
    if framed.ndim == 2:
        framed[outside] = 0
    else:
        framed[outside, :] = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);    axes[0].set_title("Исходное")
    axes[1].imshow(framed); axes[1].set_title("В круговой рамке")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_frame_compare")
    plt.close(fig)

    save_image(framed, "task50_frame")
    return framed


# ============================================================
# 7. ШУМ
# ============================================================

def add_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Гауссовский шум с нулевым средним."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);   axes[0].set_title("Исходное")
    axes[1].imshow(noisy); axes[1].set_title(f"Шум (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_noisy_compare")
    plt.close(fig)

    save_image(noisy, "task50_noisy")
    return noisy


# ============================================================
# 8. РАЗМЫТИЕ
# ============================================================

def gaussian_blur(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Гауссовское размытие (для цветного — по каждому каналу)."""
    if img.ndim == 3:
        blurred = np.stack(
            [gaussian_filter(img[..., c].astype(np.float32), sigma=sigma)
             for c in range(img.shape[2])],
            axis=-1,
        )
    else:
        blurred = gaussian_filter(img.astype(np.float32), sigma=sigma)

    blurred = np.clip(blurred, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);     axes[0].set_title("Исходное")
    axes[1].imshow(blurred); axes[1].set_title(f"Размытие (σ = {sigma})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_blurred_compare")
    plt.close(fig)

    save_image(blurred, "task50_blurred")
    return blurred


# ============================================================
# 9. SHARPENING
# ============================================================

def sharpen_image(
    img: np.ndarray,
    sigma: float = 2.0,
    amount: float = 1.5,
) -> np.ndarray:
    """Unsharp masking: sharpened = original + amount * (original - blurred)."""
    sigma_arg = (sigma, sigma, 0) if img.ndim == 3 else sigma
    blurred = gaussian_filter(img.astype(np.float32), sigma=sigma_arg)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img);       axes[0].set_title("Исходное")
    axes[1].imshow(sharpened); axes[1].set_title(f"Sharpening (amount={amount})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_sharpened_compare")
    plt.close(fig)

    save_image(sharpened, "task50_sharpened")
    return sharpened


# ============================================================
# 10. PCA
# ============================================================

def _pca_mse_curve(
    flat: np.ndarray,
    max_components: int,
    out_name: str = "task50_pca_mse",
) -> None:
    """Строит кривую MSE восстановления от числа компонент."""
    mse_values = []
    for k in range(1, max_components + 1):
        pca_k = PCA(n_components=k)
        z = pca_k.fit_transform(flat)
        rec = pca_k.inverse_transform(z)
        mse_values.append(float(np.mean((flat - rec) ** 2)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_components + 1), mse_values, marker="o", markersize=3)
    ax.set_xlabel("Число главных компонент")
    ax.set_ylabel("MSE восстановления")
    ax.set_title("Ошибка восстановления vs число компонент")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_name)
    plt.close(fig)


def _pca_components_grid(
    pca: PCA,
    features_per_pixel: int,
    out_name: str = "task50_pca_components",
) -> None:
    """Сохраняет сетку 4×4 из первых 16 главных компонент."""
    n_show = min(16, pca.components_.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        if i < n_show:
            comp = pca.components_[i]
            # Каждая компонента — вектор длины features_per_pixel.
            # Упаковываем её в маленькое изображение.
            if features_per_pixel == 3:
                vec = comp.reshape(1, 1, 3)
                vec = (vec - vec.min()) / max(vec.ptp(), 1e-9)
                ax.imshow(vec)
            else:
                side = int(np.ceil(np.sqrt(len(comp))))
                padded = np.zeros(side * side)
                padded[: len(comp)] = comp
                ax.imshow(padded.reshape(side, side), cmap="gray")
            ax.set_title(f"PC {i + 1}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out_name)
    plt.close(fig)


def apply_pca_to_image(gray_img: np.ndarray, n_components: int = 32) -> np.ndarray:
    """
    PCA к ч/б изображению: каждая строка — отдельный объект (sample).

    Возвращает восстановленное изображение.
    Дополнительно сохраняет:
      - сравнительную фигуру 'до/после' (task50_pca_compare);
      - восстановленное изображение отдельно (task50_pca_restored);
      - кривую MSE от числа компонент (task50_pca_mse);
      - сетку 4×4 из первых 16 главных компонент (task50_pca_components).
    """
    if gray_img.ndim > 2:
        gray_img = gray_img[:, :, 0]
    elif gray_img.ndim == 1:
        side = int(np.sqrt(gray_img.size))
        gray_img = gray_img.reshape(side, side)

    h, w = gray_img.shape
    n_components = max(1, min(n_components, h, w))

    flat = gray_img.astype(np.float32)              # (h, w)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flat)
    projected = pca.inverse_transform(transformed)
    reconstructed = np.clip(projected, 0, 255).astype(np.uint8)

    # сравнительная фигура
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gray_img, cmap="gray");    axes[0].set_title("Исходное Ч/Б")
    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title(f"PCA (компонент: {n_components})")
    for ax in axes:
        ax.axis("off")
    save_figure(fig, "task50_pca_compare")
    plt.close(fig)

    # отдельные PNG
    save_image(gray_img, "task50_pca_input")
    save_image(reconstructed, "task50_pca_restored")

    # кривая MSE (по числу компонент от 1 до min(64, w))
    max_k = min(64, w)
    _pca_mse_curve(flat, max_k, out_name="task50_pca_mse")

    # сетка первых 16 главных компонент
    _pca_components_grid(pca, features_per_pixel=1, out_name="task50_pca_components")

    return reconstructed


# ============================================================
# ОБЩИЙ ЗАПУСК
# ============================================================

def run_task50(image_input) -> dict:
    """
    Выполняет все 10 подзаданий.
    Принимает путь к файлу или массив numpy.
    """
    if isinstance(image_input, np.ndarray):
        img = image_input
    else:
        img = plt.imread(image_input)

    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # исходное изображение — отдельным PNG (пригодится в отчёте)
    save_image(img, "task50_input")

    results: dict = {}
    results["channels"] = split_channels(img)
    results["rotated"] = rotate_image(img, angle=40.0)
    results["gray"] = to_grayscale(img)
    results["hist"] = gray_histogram(results["gray"])
    results["parts"] = split_by_histogram(results["gray"], results["hist"])
    results["framed"] = add_circular_frame(img)
    results["noisy"] = add_noise(img, sigma=25.0)
    results["blurred"] = gaussian_blur(img, sigma=2.0)
    results["sharpened"] = sharpen_image(img, sigma=2.0, amount=1.5)
    results["pca"] = apply_pca_to_image(results["gray"], n_components=32)

    print("Задание 50 выполнено. Все графики и обработанные изображения сохранены.")
    return results