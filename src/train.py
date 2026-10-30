"""Этап train: обучение модели, выбранной в params.yaml (train.model)."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("train")

MODEL_OUT = "models/model.joblib"


def make_model(p: dict):
    if p["model"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(p["alpha"])))
    if p["model"] == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            random_state=int(p["seed"]),
            n_jobs=1,  # один поток — результат не зависит от числа ядер
        )
    raise ValueError(f"неизвестная модель: {p['model']}")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]

    frame = pd.read_csv("data/features/train.csv")
    x_train, y_train = frame.drop(columns=[target]), frame[target]

    model = make_model(params["train"]).fit(x_train, y_train)
    joblib.dump(
        {"model": model, "features": list(x_train.columns)}, ensure_parent(MODEL_OUT)
    )
    log.info("обучено: %s на %d строках", params["train"]["model"], len(frame))


if __name__ == "__main__":
    main()
