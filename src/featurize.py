"""Этап featurize: матрица признаков.

Все признаки считаются построчно, без статистик по выборке, поэтому
информация из теста в обучение не попадает. Масштабирование делается
внутри модели (Pipeline) и обучается только на train.
"""

from __future__ import annotations

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("featurize")


def build(frame: pd.DataFrame, target: str, extra_features: bool) -> pd.DataFrame:
    features = frame.copy()
    if extra_features:
        features["free_so2_share"] = (
            features["free_sulfur_dioxide"] / features["total_sulfur_dioxide"]
        )
        features["total_acidity"] = (
            features["fixed_acidity"] + features["volatile_acidity"]
        )
    ordered = [c for c in features.columns if c != target]
    if target in features.columns:
        ordered.append(target)  # при предсказании целевой колонки нет
    return features[ordered]


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    extra = bool(params["featurize"]["extra_features"])

    for part in ("train", "test"):
        frame = pd.read_csv(f"data/processed/{part}.csv")
        built = build(frame, target, extra)
        built.to_csv(ensure_parent(f"data/features/{part}.csv"), index=False)
        log.info("%s: %d строк, признаков %d", part, len(built), built.shape[1] - 1)


if __name__ == "__main__":
    main()
