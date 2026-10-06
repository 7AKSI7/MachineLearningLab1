# Лабораторная работа №1. Библиотеки Python для машинного обучения

Учебный проект по дисциплине «Машинное обучение». В работе разобраны основные библиотеки Python для анализа данных, статистики, тензорных вычислений и обработки изображений: NumPy, Pandas, Matplotlib, SciPy, scikit-learn, TensorFlow, Keras, PyTorch.

## Содержание

В проекте последовательно выполнены 50 заданий, сгруппированных по разделам:

| Раздел | Задания | Тема |
|---|---|---|
| 1 | 1–3 | Чтение и запись данных (csv, xlsx, txt, mat), сохранение графиков |
| 2 | 4–7 | Проверка качества, вывод таблицы, извлечение столбцов, типы данных |
| 3 | 8–13 | Предобработка: пропуски, масштабирование, разбиение, seed |
| 4 | 14–33 | Статистика: графики, ECDF, доверительные интервалы, корреляции, спектры, интерполяция, маски |
| 5 | 34–37 | Сравнение масштабирования с sklearn, инверсия, скользящее окно и среднее |
| 6 | 38–45 | Тензоры: TensorFlow, PyTorch, Keras, операции AXW+B и V = |sin(T) − exp(P)·√Q| |
| 7 | 46–49 | Разреженные матрицы, обратная матрица, массив F, пайплайн PCA |
| 8 | 50 | Обработка изображения: каналы, поворот, ч/б, гистограмма, шум, размытие, sharpening, PCA |

## Структура проекта

    lab1/
    ├── main.py                  # точка входа, прогоняет все 50 заданий
    ├── requirements.txt         # зависимости
    ├── README.md                # этот файл
    ├── data/                    # исходные данные
    │   ├── dataset.csv          # таблица с признаками x1, x2, x4, y
    │   └── image.jpg            # цветное изображение для задания 50
    ├── output/                  # все результаты (PNG, таблицы, отчёты)
    └── src/
        ├── __init__.py
        ├── config.py            # seed, пути, save_figure
        ├── io_utils.py          # read_table / write_table
        ├── data_loader.py       # проверка качества, extract_columns, cast_types
        ├── preprocess.py        # handle_missing, MinMaxScalerCustom, StandardScalerCustom, split
        ├── scaling.py           # сравнение собственного и sklearn-масштабирования
        ├── stats_analysis.py    # графики, статистики, корреляции, спектры, интерполяция
        ├── windows.py           # скользящее окно и среднее
        ├── tensors.py           # операции TensorFlow, PyTorch, Keras
        ├── matrix_ops.py        # sparse, inverse, массив F, PCA-пайплайн
        └── image_proc.py        # задание 50

## Установка

Требуется Python 3.10–3.11.

    # 1. Создать виртуальное окружение
    python -m venv lab1_env

    # Windows (PowerShell)
    .\lab1_env\Scripts\Activate.ps1

    # Linux / macOS
    source lab1_env/bin/activate

    # 2. Установить зависимости
    pip install --upgrade pip
    pip install -r requirements.txt

## Запуск

Из корня проекта:

    python main.py

Все результаты складываются в `output/`. Графики сохраняются как PNG, таблицы — в csv, xlsx, txt, mat. Имена файлов соответствуют номерам заданий (`task03_scatter.png`, `task14_series.png`, …, `task50_*`).

## Что появится в `output/`

- `task03_scatter.png` — диаграмма рассеяния x2 vs y.
- `task14_series.png`, `task15_histogram.png`, `task17_ecdf.png` — графики.
- `task28_spectrogram.png`, `task29_periodogram.png`, `task30_fft.png` — спектры.
- `task49_pca.png` — кумулятивная объяснённая дисперсия.
- `task50_input.png`, `task50_channel_r/g/b.png` — исходное изображение и каналы.
- `task50_rotated.png`, `task50_gray.png`, `task50_frame.png` — поворот, ч/б, круглая рамка.
- `task50_noisy.png`, `task50_blurred.png`, `task50_sharpened.png` — шум, размытие, sharpening.
- `task50_pca_restored.png`, `task50_pca_mse.png`, `task50_pca_components.png` — результаты PCA.
- `lab_results.txt` — итоговый отчёт по всем заданиям.

## Технологии

- Python 3.10+
- NumPy, Pandas, Matplotlib
- SciPy, scikit-learn, scikit-image
- TensorFlow, Keras, PyTorch
- openpyxl (для xlsx)

## Примечания

- Все ГПСЧ фиксируются через `set_seeds(SEED)` из `src/config.py`, поэтому результаты воспроизводимы.
- В датасете намеренно есть пропуски и некорректные значения (`ошибка`), чтобы отработать предобработку (задание 8).
- В `dataset.csv` нет столбца `x3` — набор признаков по заданию: `x1, x2, x4, y`.
- Для PCA к изображению используется ч/б версия; число компонент по умолчанию — 32.

## Автор

Студент группы ИПБП41, ДГТУ, кафедра ИПМ «Ростсельмаш», Солодкий Максим Игоревич, вариант 14
