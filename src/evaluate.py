"""Этап evaluate: метрики на тесте, metrics.json и данные для графика.

Если R2 ниже порога evaluate.min_r2, этап завершается с ошибкой.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("evaluate")


def main() -> None:
    params = load_params()
    target = params["data"]["target"]
    min_r2 = float(params["evaluate"]["min_r2"])

    bundle = joblib.load("models/model.joblib")
    frame = pd.read_csv("data/features/test.csv")
    y_test = frame[target]
    y_pred = bundle["model"].predict(frame[bundle["features"]])

    metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
        "n_test": len(frame),
    }
    out = ensure_parent("reports/metrics.json")
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred.round(4)}).to_csv(
        ensure_parent("reports/plots/pred_vs_true.csv"), index=False
    )

    log.info("метрики: %s", metrics)
    if metrics["r2"] < min_r2:
        log.error("R2 = %.4f ниже порога %.2f", metrics["r2"], min_r2)
        sys.exit(1)


if __name__ == "__main__":
    main()
