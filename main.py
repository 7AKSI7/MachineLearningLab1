"""
Лабораторная работа №1.
Библиотеки Python для машинного обучения.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
import torch
from keras import ops

from src.config import (
    DATASET_PATH,
    IMAGE_PATH,
    OUTPUT_DIR,
    SEED,
    save_figure,
    set_seeds,
)
from src.data_loader import (
    cast_types,
    check_data_quality,
    extract_columns,
    show_table,
)
from src.image_proc import run_task50
from src.io_utils import read_table, write_table
from src.matrix_ops import (
    build_F,
    create_sparse_matrix,
    inverse_matrix,
    pca_pipeline,
    plot_pca_variance,
)
from src.preprocess import (
    MinMaxScalerCustom,
    StandardScalerCustom,
    handle_missing,
    split_train_val_test,
    to_numpy,
)
from src.scaling import check_inverse_transform, compare_scalers
from src.stats_analysis import (
    binary_masks,
    ci_mean,
    ci_var,
    column_stats,
    cross_correlation,
    cubic_interpolation,
    distribution_tests,
    dot_product,
    cross_product,
    numerical_gradient,
    plot_ecdf,
    plot_fft_amplitude,
    plot_histogram,
    plot_periodogram,
    plot_series,
    plot_spectrogram,
    show_sorted,
    spline_interpolation,
    vector_norms,
)
from src.windows import moving_average, sliding_window


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def section(title: str) -> None:
    """Печатает красивый разделитель перед заданием."""
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def make_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит все столбцы к float, заменяя нечисловые значения на NaN."""
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ============================================================
# СТАРТ
# ============================================================

set_seeds(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 62)
print("ЛАБОРАТОРНАЯ РАБОТА №1")
print("Библиотеки Python для машинного обучения")
print("=" * 62)
print(f"\nРезультаты сохраняются в:\n{OUTPUT_DIR}")


# ============================================================
# ЗАДАНИЕ 1. Чтение файлов
# ============================================================

section("ЗАДАНИЕ 1. ЧТЕНИЕ ФАЙЛА")

data = read_table(DATASET_PATH)
print(data.head())
print(f"\nShape: {data.shape}")
print(f"Columns: {list(data.columns)}")


# ============================================================
# ЗАДАНИЕ 2. Запись файлов
# ============================================================

section("ЗАДАНИЕ 2. ЗАПИСЬ ФАЙЛОВ")

saved_paths = write_table(data, "dataset")

print("Созданные файлы:")
for kind, path in saved_paths.items():
    print(f"  {kind}: {path}")


# ============================================================
# ЗАДАНИЕ 3. Сохранение изображения (matplotlib figure)
# ============================================================

section("ЗАДАНИЕ 3. СОХРАНЕНИЕ ГРАФИКА")

plot_df = make_float_frame(data[["x2", "y"]])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(plot_df["x2"], plot_df["y"], ".", markersize=3)
ax.set_xlabel("x2")
ax.set_ylabel("y")
ax.set_title("Зависимость y от x2")
ax.grid(True, alpha=0.3)
fig.tight_layout()

figure_path = save_figure(fig, "task03_scatter")
print("График сохранён:", figure_path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 4. Проверка данных
# ============================================================

section("ЗАДАНИЕ 4. ПРОВЕРКА ДАННЫХ")

report = check_data_quality(data)

print("\nПропущенные значения:")
print(report["missing"])

print("\nТипы данных:")
print(report["dtypes"])

print("\nНекорректные числовые значения:")
print(report["non_numeric"])


# ============================================================
# ЗАДАНИЕ 5. Вывод таблицы
# ============================================================

section("ЗАДАНИЕ 5. ВЫВОД ТАБЛИЦЫ")

show_table(data, n=5)


# ============================================================
# ЗАДАНИЕ 6. Извлечение столбцов
# ============================================================

section("ЗАДАНИЕ 6. ИЗВЛЕЧЕНИЕ СТОЛБЦОВ")

selected_data = extract_columns(data, ["x1", "x2", "y"])
print(selected_data.head())


# ============================================================
# ЗАДАНИЕ 7. Явные типы данных
# ============================================================

section("ЗАДАНИЕ 7. ЯВНЫЕ ТИПЫ ДАННЫХ")

typed_data = cast_types(
    data,
    {"x2": "float64", "y": "float64"},
)

print(typed_data[["x2", "y"]].dtypes)
print(typed_data[["x2", "y"]].head())


# ============================================================
# ЗАДАНИЕ 8. Обработка проблемных данных
# ============================================================

section("ЗАДАНИЕ 8. ОБРАБОТКА ПРОБЛЕМНЫХ ДАННЫХ")

# В столбце x1 встречается строка "ошибка" и есть пустые ячейки.
# Шаг 1: превращаем всё нечисловое в NaN.
data_clean = handle_missing(data, "x1", "cast")

# Шаг 2: заполняем NaN средним.
data_clean = handle_missing(data_clean, "x1", "mean")

# На всякий случай — гарантируем float.
data_clean["x1"] = pd.to_numeric(data_clean["x1"], errors="coerce")
data_clean["x1"] = data_clean["x1"].fillna(data_clean["x1"].mean())

print("x1 после очистки:")
print(data_clean["x1"].head())

print("\nОсталось NaN в x1:", int(data_clean["x1"].isna().sum()))


# ============================================================
# ЗАДАНИЕ 9. DataFrame -> NumPy
# ============================================================

section("ЗАДАНИЕ 9. DATAFRAME -> NUMPY")

# В датасете НЕТ столбца x3. Используем те, что реально есть.
wanted_columns = ["x1", "x2", "x4", "y"]
numeric_df = extract_columns(data_clean, wanted_columns)

# Приводим к числовому типу и выкидываем оставшиеся NaN.
numeric_df = make_float_frame(numeric_df).dropna(axis=0, how="any")

X = to_numpy(numeric_df)

print("Тип:", type(X))
print("Shape:", X.shape)
print(X[:5])


# ============================================================
# ЗАДАНИЕ 10. Масштабирование
# ============================================================

section("ЗАДАНИЕ 10. МАСШТАБИРОВАНИЕ")

mm_scaler = MinMaxScalerCustom()
X_mm = mm_scaler.fit_transform(X)

print("MinMax (первые 5 строк):")
print(X_mm[:5])

std_scaler = StandardScalerCustom()
X_std = std_scaler.fit_transform(X)

print("\nStandardScaler (первые 5 строк):")
print(X_std[:5])


# ============================================================
# ЗАДАНИЕ 11. Восстановление масштаба
# ============================================================

section("ЗАДАНИЕ 11. ВОССТАНОВЛЕНИЕ МАСШТАБА")

X_back = mm_scaler.inverse_transform(X_mm)

print(X_back[:5])
print("\nСовпадает с исходными:", np.allclose(X, X_back))


# ============================================================
# ЗАДАНИЕ 12. Разбиение на 3 части
# ============================================================

section("ЗАДАНИЕ 12. РАЗБИЕНИЕ НА 3 ЧАСТИ")

X_train, X_val, X_test = split_train_val_test(
    X,
    (70, 15, 15),
    percent=True,
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# ЗАДАНИЕ 13. Фиксация seed
# ============================================================

section("ЗАДАНИЕ 13. ФИКСАЦИЯ SEED")

set_seeds(SEED)
sample_a = np.random.rand(5)

set_seeds(SEED)
sample_b = np.random.rand(5)

print("Первый массив:  ", sample_a)
print("Второй массив:  ", sample_b)
print("Одинаковые:     ", np.array_equal(sample_a, sample_b))


# ============================================================
# Подготовка столбцов для заданий 14–49
# ============================================================

col_x1 = X[:, 0]
col_x2 = X[:, 1]
col_x4 = X[:, 2]
col_y = X[:, 3]


# ============================================================
# ЗАДАНИЕ 14. График исходных данных
# ============================================================

section("ЗАДАНИЕ 14. ГРАФИК ИСХОДНЫХ ДАННЫХ")

fig = plot_series(X, title="Исходные данные")
path = save_figure(fig, "task14_series")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 15. Гистограмма
# ============================================================

section("ЗАДАНИЕ 15. ГИСТОГРАММА")

fig = plot_histogram(col_x2, bins=30)
path = save_figure(fig, "task15_histogram")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 16. Сортировка
# ============================================================

section("ЗАДАНИЕ 16. СОРТИРОВКА")

print("Отсортированные значения x2 (первые 15):")
print(np.sort(col_x2)[:15])


# ============================================================
# ЗАДАНИЕ 17. Эмпирическая функция распределения
# ============================================================

section("ЗАДАНИЕ 17. ЭМПИРИЧЕСКАЯ ФУНКЦИЯ")

fig = plot_ecdf(col_x2)
path = save_figure(fig, "task17_ecdf")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 18. Статистики
# ============================================================

section("ЗАДАНИЕ 18. СТАТИСТИКИ")

statistics = column_stats(col_x2)
for name, value in statistics.items():
    print(f"  {name}: {value}")


# ============================================================
# ЗАДАНИЕ 19. Доверительные интервалы
# ============================================================

section("ЗАДАНИЕ 19. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")

mean_interval = ci_mean(col_x2)
var_interval = ci_var(col_x2)

print("Интервал для среднего: ", mean_interval)
print("Интервал для дисперсии:", var_interval)


# ============================================================
# ЗАДАНИЕ 20. Ковариация и корреляция
# ============================================================

section("ЗАДАНИЕ 20. КОВАРИАЦИЯ И КОРРЕЛЯЦИЯ")

cov_matrix = np.cov(X, rowvar=False)
corr_matrix = np.corrcoef(X, rowvar=False)

print("Ковариационная матрица:")
print(cov_matrix)

print("\nКорреляционная матрица:")
print(corr_matrix)


# ============================================================
# ЗАДАНИЕ 21. Значимость корреляции
# ============================================================

section("ЗАДАНИЕ 21. ЗНАЧИМОСТЬ КОРРЕЛЯЦИИ")

r_value, p_value = stats.pearsonr(col_x2, col_y)

print(f"r = {r_value:.6f}")
print(f"p-value = {p_value:.6f}")
print(
    "Корреляция значима." if p_value < 0.05
    else "Корреляция незначима."
)


# ============================================================
# ЗАДАНИЕ 22. Взаимная корреляция
# ============================================================

section("ЗАДАНИЕ 22. ВЗАИМНАЯ КОРРЕЛЯЦИЯ")

lags, cross_vals = cross_correlation(col_x2, col_y, max_lags=50)

print("Лаги (первые 10):     ", lags[:10])
print("Корреляции (первые 10):", np.round(cross_vals[:10], 4))


# ============================================================
# ЗАДАНИЕ 23. Производная и градиент
# ============================================================

section("ЗАДАНИЕ 23. ПРОИЗВОДНАЯ И ГРАДИЕНТ")

dx = numerical_gradient(col_x2)
dx_full = np.gradient(X, axis=0)

print("Производная x2 (первые 10):", dx[:10])
print("Градиент X (первые 3 строки):")
print(dx_full[:3])


# ============================================================
# ЗАДАНИЕ 24. Свёртка
# ============================================================

section("ЗАДАНИЕ 24. СВЁРТКА")

conv_result = np.convolve(col_x2, col_y, mode="full")
print("Свёртка (первые 20 значений):", conv_result[:20])


# ============================================================
# ЗАДАНИЕ 25. Скалярное и векторное произведения
# ============================================================

section("ЗАДАНИЕ 25. ПРОИЗВЕДЕНИЯ")

dot_val = dot_product(col_x2, col_y)
cross_val = cross_product(col_x2, col_y)

print("Скалярное произведение: ", dot_val)
print("Векторное произведение:", cross_val)


# ============================================================
# ЗАДАНИЕ 26. Нормы L1 и L2
# ============================================================

section("ЗАДАНИЕ 26. НОРМЫ L1 И L2")

norms = vector_norms(col_x2)
print("L1:", norms["L1"])
print("L2:", norms["L2"])


# ============================================================
# ЗАДАНИЕ 27. Проверка гипотез
# ============================================================

section("ЗАДАНИЕ 27. ПРОВЕРКА ГИПОТЕЗ")

tests = distribution_tests(col_x1, col_x2)
for name, value in tests.items():
    print(f"  {name}: {value:.6f}")


# ============================================================
# ЗАДАНИЕ 28. Спектрограмма
# ============================================================

section("ЗАДАНИЕ 28. СПЕКТРОГРАММА")

fig = plot_spectrogram(col_x2)
path = save_figure(fig, "task28_spectrogram")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 29. Периодограмма
# ============================================================

section("ЗАДАНИЕ 29. ПЕРИОДОГРАММА")

fig = plot_periodogram(col_x2)
path = save_figure(fig, "task29_periodogram")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 30. АЧХ через FFT
# ============================================================

section("ЗАДАНИЕ 30. АЧХ ЧЕРЕЗ FFT")

fig = plot_fft_amplitude(col_x2)
path = save_figure(fig, "task30_fft")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 31. Кубическая интерполяция
# ============================================================

section("ЗАДАНИЕ 31. КУБИЧЕСКАЯ ИНТЕРПОЛЯЦИЯ")

x_new, y_cubic = cubic_interpolation(col_x2, factor=10)

print("Исходных точек:", len(col_x2))
print("Новых точек:   ", len(y_cubic))


# ============================================================
# ЗАДАНИЕ 32. Интерполяция сплайнами
# ============================================================

section("ЗАДАНИЕ 32. ИНТЕРПОЛЯЦИЯ СПЛАЙНАМИ")

x_new_spl, y_spline = spline_interpolation(col_x2, factor=10)
print("Количество точек:", len(y_spline))


# ============================================================
# ЗАДАНИЕ 33. Бинарные маски
# ============================================================

section("ЗАДАНИЕ 33. БИНАРНЫЕ МАСКИ")

masks = binary_masks(col_x2)
for name, mask in masks.items():
    print(f"  {name}: {int(mask.sum())}")


# ============================================================
# ЗАДАНИЕ 34. Сравнение масштабирования
# ============================================================

section("ЗАДАНИЕ 34. СРАВНЕНИЕ МАСШТАБИРОВАНИЯ")

x_column = X[:, [0]]
scaling_results = compare_scalers(x_column)

print(
    "MinMax custom == sklearn:",
    np.allclose(
        scaling_results["custom_minmax"],
        scaling_results["sklearn_minmax"],
        atol=1e-10,
    ),
)
print(
    "Standard custom == sklearn:",
    np.allclose(
        scaling_results["custom_standard"],
        scaling_results["sklearn_standard"],
        atol=1e-10,
    ),
)

comparison = pd.DataFrame({
    "original": x_column.ravel(),
    "custom": scaling_results["custom_minmax"].ravel(),
    "sklearn": scaling_results["sklearn_minmax"].ravel(),
})
write_table(comparison, "task34_scaling")


# ============================================================
# ЗАДАНИЕ 35. Инверсия
# ============================================================

section("ЗАДАНИЕ 35. ИНВЕРСИЯ")

inverse_results = check_inverse_transform(x_column)

print("Custom восстановление: ", inverse_results["custom_ok"])
print("Sklearn восстановление:", inverse_results["sklearn_ok"])


# ============================================================
# ЗАДАНИЕ 36. Скользящее окно
# ============================================================

section("ЗАДАНИЕ 36. СКОЛЬЗЯЩЕЕ ОКНО")

windows = sliding_window(col_x2, width=5)
print("Shape:", windows.shape)
print(windows[:5])


# ============================================================
# ЗАДАНИЕ 37. Скользящее среднее
# ============================================================

section("ЗАДАНИЕ 37. СКОЛЬЗЯЩЕЕ СРЕДНЕЕ")

moving = moving_average(col_x2, width=5)
print("Shape:", moving.shape)
print(moving[:10])


# ============================================================
# ЗАДАНИЕ 38. Тензор TensorFlow
# ============================================================

section("ЗАДАНИЕ 38. ТЕНЗОР TENSORFLOW")

X_tf = tf.constant(X, dtype=tf.float32)
print(X_tf)
print("Shape:", X_tf.shape)


# ============================================================
# ЗАДАНИЕ 39. Тензор PyTorch
# ============================================================

section("ЗАДАНИЕ 39. ТЕНЗОР PYTORCH")

X_pt = torch.tensor(X, dtype=torch.float32)
print(X_pt)
print("Shape:", X_pt.shape)


# ============================================================
# ЗАДАНИЕ 40. Случайные тензоры
# ============================================================

section("ЗАДАНИЕ 40. СЛУЧАЙНЫЕ ТЕНЗОРЫ")

n, m = X_tf.shape
k, p = 5, 4

A = tf.random.uniform((k, n), minval=0, maxval=10, dtype=tf.int32)
W = tf.random.normal((m, p))
B = tf.random.uniform((k, p))

print("A:", A.shape, "W:", W.shape, "B:", B.shape)


# ============================================================
# ЗАДАНИЕ 41. AXW+B (TensorFlow)
# ============================================================

section("ЗАДАНИЕ 41. AXW+B — TENSORFLOW")

AX = tf.matmul(tf.cast(A, tf.float32), X_tf)
AXW = tf.matmul(AX, W)
result_tf = AXW + B

print(result_tf)
print("Shape:", result_tf.shape)


# ============================================================
# ЗАДАНИЕ 42. AXW+B (PyTorch)
# ============================================================

section("ЗАДАНИЕ 42. AXW+B — PYTORCH")

A_pt = torch.randint(0, 10, (k, n), dtype=torch.float32)
W_pt = torch.randn(m, p)
B_pt = torch.rand(k, p)

result_pt = A_pt @ X_pt @ W_pt + B_pt

print(result_pt)
print("Shape:", result_pt.shape)


# ============================================================
# ЗАДАНИЕ 43. Матрицы T, P, Q
# ============================================================

section("ЗАДАНИЕ 43. МАТРИЦЫ T, P, Q")

T = np.random.rand(3, 10)
P = np.random.rand(3, 10)
Q = np.random.rand(3, 10)

print("T:\n", T)
print("\nP:\n", P)
print("\nQ:\n", Q)


# ============================================================
# ЗАДАНИЕ 44. Операция в Keras
# ============================================================

section("ЗАДАНИЕ 44. ОПЕРАЦИЯ KERAS")

T_k = ops.convert_to_tensor(T)
P_k = ops.convert_to_tensor(P)
Q_k = ops.convert_to_tensor(Q)

V_k = ops.abs(ops.sin(T_k) - ops.exp(P_k) * ops.sqrt(Q_k))

print(V_k)
print("Shape:", V_k.shape)


# ============================================================
# ЗАДАНИЕ 45. Операция в PyTorch
# ============================================================

section("ЗАДАНИЕ 45. ОПЕРАЦИЯ PYTORCH")

T_t = torch.tensor(T)
P_t = torch.tensor(P)
Q_t = torch.tensor(Q)

V_t = torch.abs(torch.sin(T_t) - torch.exp(P_t) * torch.sqrt(Q_t))

print(V_t)
print("Shape:", V_t.shape)


# ============================================================
# ЗАДАНИЕ 46. Разреженная матрица SciPy
# ============================================================

section("ЗАДАНИЕ 46. РАЗРЕЖЕННАЯ МАТРИЦА")

sparse_matrix = create_sparse_matrix(
    rows=100_000,
    cols=100_000,
    density=1e-5,
)

print("Shape:      ", sparse_matrix.shape)
print("NNZ:        ", sparse_matrix.nnz)
print("Dtype:      ", sparse_matrix.dtype)
print("Формат:     ", sparse_matrix.format)

total_cells = sparse_matrix.shape[0] * sparse_matrix.shape[1]
print("Плотность:  ", sparse_matrix.nnz / total_cells)

del sparse_matrix  # освобождаем память


# ============================================================
# ЗАДАНИЕ 47. Обратная матрица
# ============================================================

section("ЗАДАНИЕ 47. ОБРАТНАЯ МАТРИЦА")

inverse_result = inverse_matrix(size=100)

print("Определитель:", inverse_result["det"])
print("Обратима:   ", inverse_result["invertible"])

if inverse_result["invertible"]:
    print("Проверка M @ M^-1 = I:", inverse_result["check"])


# ============================================================
# ЗАДАНИЕ 48. Массив F
# ============================================================

section("ЗАДАНИЕ 48. МАССИВ F")

F = build_F(X)
print("Shape F:", F.shape)

# Для sklearn PCA выпрямляем последние два измерения.
X_F = F.reshape(F.shape[0], -1)
print("Shape X_F (для PCA):", X_F.shape)


# ============================================================
# ЗАДАНИЕ 49. Пайплайн sklearn + PCA
# ============================================================

section("ЗАДАНИЕ 49. ПАЙПЛАЙН SKLEARN + PCA")

# Считаем log-likelihood для двух пайплайнов (minmax / standard).
scores, best = pca_pipeline(F)

print("Log-likelihood по вариантам:")
for name, value in scores.items():
    print(f"  {name}: {value:.4f}")
print("Лучший вариант:", best)

# График кумулятивной объяснённой дисперсии.
fig = plot_pca_variance(F)
path = save_figure(fig, "task49_pca")
print("График сохранён:", path)
plt.close(fig)


# ============================================================
# ЗАДАНИЕ 50. Обработка изображения
# ============================================================

section("ЗАДАНИЕ 50. ОБРАБОТКА ИЗОБРАЖЕНИЯ")

if IMAGE_PATH.exists():
    print(f"Изображение: {IMAGE_PATH}")
    run_task50(str(IMAGE_PATH))
else:
    print(f"Изображение не найдено:\n{IMAGE_PATH}")
    print("Положите файл image.png (или измените IMAGE_PATH в config.py).")


# ============================================================
# ФИНАЛЬНЫЙ ОТЧЁТ
# ============================================================

section("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")

report_file = OUTPUT_DIR / "lab_results.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("ЛАБОРАТОРНАЯ РАБОТА №1\n")
    f.write("Библиотеки Python для машинного обучения\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Датасет:       {DATASET_PATH}\n")
    f.write(f"Изображение:   {IMAGE_PATH}\n")
    f.write(f"Папка вывода:  {OUTPUT_DIR}\n\n")
    f.write("Файлы результатов:\n")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            f.write(f"  - {path.name}\n")

print(f"Общий отчёт: {report_file}")
print("\nВсе 50 заданий выполнены.")