"""Этап train: обучение модели, выбранной в params.yaml (train.model)."""

from __future__ import annotations

import joblib
import json
import time
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import ensure_parent, get_logger, load_params
from src import tracking

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
    if p["model"] == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            learning_rate=float(p.get("learning_rate", 0.05)),
            random_state=int(p["seed"]),
        )
    raise ValueError(f"неизвестная модель: {p['model']}")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]

    frame = pd.read_csv("data/features/train.csv")
    x_train, y_train = frame.drop(columns=[target]), frame[target]

    start = time.perf_counter()
    model = make_model(params["train"]).fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    joblib.dump(
        {"model": model, "features": list(x_train.columns)}, ensure_parent(MODEL_OUT)
    )
    ensure_parent("reports/train_info.json").write_text(
        json.dumps({"train_seconds": round(elapsed, 4), "n_train": len(frame), "model": params["train"]["model"]}, indent=2),
        encoding="utf-8",
    )
    tracking.log_training(params["mlflow"]["experiment"], params["train"], model, x_train.head(50)) if tracking.enabled() else None
    log.info("обучено: %s на %d строках", params["train"]["model"], len(frame))


if __name__ == "__main__":
    main()
