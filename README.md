# ЛР1. Воспроизводимый проект: данные и пайплайн

**Вариант 2 — Wine Quality (UCI).** Регрессия: по 11 физико-химическим показателям
предсказать оценку качества красного вина (целое число 0–10).
В ЛР4 роль потока новых данных будут играть белые вина.

## Структура

```
src/            пакет с кодом: ingest, prepare, featurize, train, evaluate, utils
notebooks/      lr1_data_and_pipeline.ipynb — рабочий ноутбук для Colab
tests/          pytest: юнит-тесты и проверка воспроизводимости
data/raw/       исходный набор (под DVC, в Git только wine.csv.dvc)
data/processed/ data/features/   выходы этапов (под DVC через dvc.yaml)
models/         обученная модель (под DVC)
reports/        metrics.json и данные для графика (в Git)
params.yaml     все параметры пайплайна
dvc.yaml        описание этапов, dvc.lock — их зафиксированное состояние
pyproject.toml  зависимости, uv.lock — точные версии
```

## Развернуть окружение (Colab)

```bash
pip install -q uv
uv export --frozen --no-hashes --no-emit-project -o /content/requirements.lock.txt
pip install -q -r /content/requirements.lock.txt
```

Локально то же самое делает `uv sync`.

## Забрать данные и воспроизвести пайплайн

```bash
git clone https://github.com/St1ck123/mlopslr1.git && cd mlopslr1
dvc remote modify --local origin access_key_id <DAGSHUB_TOKEN>
dvc remote modify --local origin secret_access_key <DAGSHUB_TOKEN>
dvc pull
dvc repro          # Data and pipelines are up to date.
cat reports/metrics.json
```

Метрики лежат в `reports/metrics.json`, их также показывают `dvc metrics show`
и вкладка репозитория на DagsHub.

## Пайплайн

| Этап | Вход | Выход | Что делает |
|---|---|---|---|
| prepare | data/raw/wine.csv | data/processed/*.csv | проверка схемы, удаление дублей, разбиение 80/20 |
| featurize | data/processed/*.csv | data/features/*.csv | доля свободного SO₂, суммарная кислотность |
| train | data/features/train.csv | models/model.joblib | Ridge со стандартизацией (или RandomForest) |
| evaluate | модель и тест | reports/metrics.json | R², MAE, RMSE, проверка порога R² ≥ 0.25 |

```
  +-----------------------+
  | data/raw/wine.csv.dvc |
  +-----------------------+
              *
         +---------+
         | prepare |
         +---------+
              *
        +-----------+
        | featurize |
        +-----------+
         **        **
+-------+            *
| train |          **
+-------+        **
         **    **
        +----------+
        | evaluate |
        +----------+
```

Повторный `dvc repro` ничего не пересчитывает: DVC сравнивает хеши зависимостей
и значения параметров с записанными в `dvc.lock`.

Дубликаты (в красных винах их около 15 %) удаляются **до** разбиения: иначе одна и та
же строка попадает и в обучение, и в тест, и метрика завышается.

## Версии данных

Источник — полный файл `winequality-red.csv` в `data/external/` (вне Git и DVC,
`src.ingest` скачивает его с UCI). Версия данных — первые N строк файла:

| Тег | Команда | Строк | После удаления дублей | Тест, строк | R² | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| `data-v1` | `python -m src.ingest` | 1000 | 851 | 171 | 0.3481 | 0.5695 | 0.7383 |
| `data-v2` | `python -m src.ingest --all` | 1599 | 1359 | 272 | 0.2672 | 0.5279 | 0.6959 |

Смена версии: `ingest --all` → `dvc add data/raw/wine.csv` → `dvc repro` →
`dvc push` → `git commit` → `git tag data-v2` → `git push`.

Возврат к прошлой версии:

```bash
git checkout data-v1
dvc checkout
dvc repro          # ничего не пересчитывает, metrics.json снова с R² = 0.3481
```

На v2 R² ниже, а MAE лучше: тестом становятся другие вина (хвост полного файла),
и разброс оценок в нём меньше (стандартное отклонение 0.81 против 0.92 на v1),
поэтому доля объяснённой дисперсии падает при меньшей абсолютной ошибке.

## Дополнительные задания

Обе вещи переключаются только в `params.yaml` (`data.split`, `train.model`),
код не меняется:

```bash
dvc exp run --temp -S data.split=random
dvc exp run --temp -S train.model=random_forest
dvc exp show
```

| Данные | Модель | ordered, R² | random, R² |
|---|---|---|---|
| v1 (1000) | Ridge | 0.3481 | 0.2663 |
| v1 (1000) | RandomForest 300/12/2 | 0.3415 | 0.2988 |
| v2 (1599) | Ridge | 0.2672 | 0.3329 |
| v2 (1599) | RandomForest 300/12/2 | 0.2932 | 0.3580 |

**Разбиение.** Даты в наборе нет, поэтому «по времени» здесь значит «по порядку
в файле»: тест — последние 20 % строк, как самые новые поступления. Утечки из
будущего, как в данных с датой, здесь нет, поэтому случайное разбиение не обязано
давать оценку выше. Разница в обе стороны (до 0.08) показывает, что хвост файла
по составу отличается от середины, а на тесте из 171–272 вин R² сильно зависит от
того, какие вина в него попали. Разбиение по порядку честнее: оно ближе к тому,
как модель будет работать на новой порции.

**Вторая модель.** RandomForest на v2 лучше Ridge при обоих разбиениях
(0.2932 против 0.2672), на v1 — примерно равен ему. Базовой оставлена Ridge как
более простая и быстрая.

**Воспроизводимость.** `tests/test_reproducibility.py` дважды запускает все
четыре этапа с нуля в чистых каталогах и сравнивает метрики с допуском 1e-9:

```bash
pytest -q          # 3 passed
```

## Проверки качества кода

Ruff (`ruff check`, `ruff format`) и pre-commit с хуками `ruff-check`,
`ruff-format`, `check-added-large-files` (1 МБ), `check-yaml`,
`detect-private-key`:

```bash
pre-commit install
pre-commit run --all-files
```

Токены в репозиторий не попадают: в Colab они читаются из «Секретов»
(`userdata.get`), ключи DVC пишутся только в `.dvc/config.local`.
