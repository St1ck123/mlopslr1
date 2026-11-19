from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GRID = [
    {"model": "ridge", "alpha": 1.0, "seed": 42},
    {"model": "ridge", "alpha": 10.0, "seed": 42},
    {"model": "random_forest", "n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2, "seed": 42},
    {"model": "random_forest", "n_estimators": 300, "max_depth": 16, "min_samples_leaf": 1, "seed": 42},
    {"model": "gradient_boosting", "n_estimators": 200, "max_depth": 2, "min_samples_leaf": 2, "learning_rate": 0.05, "seed": 42},
    {"model": "gradient_boosting", "n_estimators": 350, "max_depth": 3, "min_samples_leaf": 2, "learning_rate": 0.05, "seed": 42},
]


def write_yaml(params):
    (ROOT / "params.yaml").write_text(
        yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_results(results):
    directory = ROOT / "results"
    directory.mkdir(exist_ok=True)
    (directory / "experiments.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    keys = [
        "experiment", "model", "alpha", "n_estimators", "max_depth",
        "min_samples_leaf", "learning_rate", "seed", "r2", "mae",
        "rmse", "n_test", "train_seconds", "run_id", "model_uri"
    ]
    with (directory / "experiments.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def run_all():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise EnvironmentError("Требуется MLFLOW_TRACKING_URI для DagsHub")
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    original = json.loads(json.dumps(params))
    results = []
    (ROOT / "reports").mkdir(exist_ok=True)
    try:
        for index, config in enumerate(GRID, 1):
            params["train"] = dict(config)
            params.setdefault("evaluate", {})["min_r2"] = -1_000_000
            write_yaml(params)
            print(f"[{index}/{len(GRID)}] {config}", flush=True)
            subprocess.run(["dvc", "repro", "-f"], cwd=ROOT, check=True)
            metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
            info = json.loads((ROOT / "reports/train_info.json").read_text(encoding="utf-8"))
            record = {
                "experiment": index, "model": config["model"], **config, **metrics,
                "train_seconds": info["train_seconds"],
                "run_id": (ROOT / "reports/run_id.txt").read_text(encoding="utf-8").strip(),
                "model_uri": (ROOT / "reports/model_uri.txt").read_text(encoding="utf-8").strip(),
            }
            results.append(record)
            write_results(results)
    finally:
        if results:
            best = max(results, key=lambda item: item["r2"])
            original["train"] = dict(GRID[best["experiment"] - 1])
        write_yaml(original)
    if results:
        subprocess.run(["dvc", "repro", "-f"], cwd=ROOT, check=True)
    return results


if __name__ == "__main__":
    print("Лучшая конфигурация:", max(run_all(), key=lambda x: x["r2"]))
