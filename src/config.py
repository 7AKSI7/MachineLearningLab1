"""
Конфигурация проекта лабораторной работы №1.

Содержит:
- фиксацию seed для воспроизводимости;
- пути к данным и результатам;
- утилиты для сохранения графиков и файлов с меткой времени.
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

# ---------------------------------------------------------------- seed
SEED: int = 42


def set_seeds(seed: int = SEED) -> None:
    """Фиксирует состояние всех ГПСЧ, которые используются в проекте."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------- пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = BASE_DIR / "results"

for _d in (DATA_DIR, OUTPUT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset.csv"
IMAGE_PATH = DATA_DIR / "image.jpg"


# ------------------------------------------------------------- утилиты
def timestamped_filename(base: str, ext: str, folder: Path | None = None) -> str:
    """Собирает имя файла вида '<base>_YYYY-mm-dd_HH-MM-SS.<ext>'."""
    target = Path(folder) if folder is not None else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return str(target / f"{base}_{stamp}.{ext}")


def save_figure(fig, name: str, folder: Path | None = None) -> str:
    """Сохраняет matplotlib-фигуру в PNG и возвращает путь."""
    target = Path(folder) if folder is not None else OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path)