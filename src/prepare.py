"""Этап prepare: проверка исходных данных, очистка и разбиение.

В наборе нет даты, поэтому «время» — порядок строк в файле: при split=ordered
в тест уходит последняя часть файла, как будто это самые новые вина.
Дубликаты удаляются до разбиения, иначе одна и та же строка окажется и в
обучении, и в тесте.
"""

from __future__ import annotations

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("prepare")

COLUMNS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol", "quality",
]  # fmt: skip


def check(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Минимальная схема: состав колонок, пропуски, диапазон оценки."""
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"нет колонок: {sorted(missing)}")
    if frame[COLUMNS].isna().any().any():
        raise ValueError("в данных есть пропуски")
    if not frame[target].between(0, 10).all():
        raise ValueError(f"{target} вне диапазона 0–10")
    return frame[COLUMNS]


def split(frame: pd.DataFrame, test_size: float, how: str, seed: int) -> tuple:
    if how == "random":
        frame = frame.sample(frac=1.0, random_state=seed)
    elif how != "ordered":
        raise ValueError(f"неизвестный способ разбиения: {how}")
    edge = int(len(frame) * (1 - test_size))
    return frame.iloc[:edge], frame.iloc[edge:]


def main() -> None:
    params = load_params()
    data = params["data"]

    raw = check(pd.read_csv(data["raw"]), data["target"])
    clean = raw.drop_duplicates().reset_index(drop=True)
    log.info("схема пройдена: %d строк, после удаления дублей %d", len(raw), len(clean))

    train, test = split(
        clean, float(data["test_size"]), data["split"], int(params["train"]["seed"])
    )
    train.to_csv(ensure_parent("data/processed/train.csv"), index=False)
    test.to_csv(ensure_parent("data/processed/test.csv"), index=False)
    log.info("разбиение %s: обучение %d, тест %d", data["split"], len(train), len(test))


if __name__ == "__main__":
    main()
