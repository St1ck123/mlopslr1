from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def settings():
    return yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["mlflow"]


def metric(client, version, key):
    if version.run_id:
        value = client.get_run(version.run_id).data.metrics.get(key)
        if value is not None:
            return value
    tags = getattr(version, "tags", {}) or {}
    if f"lr2_{key}" in tags:
        return float(tags[f"lr2_{key}"])
    raise KeyError(f"Нет метрики {key} для версии {version.version}")


def register(model_uri=None, run_id=None):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise EnvironmentError("Сначала настройте MLFLOW_TRACKING_URI")
    name = settings()["model_name"]
    if not model_uri:
        model_uri = (ROOT / "reports/model_uri.txt").read_text(encoding="utf-8").strip()
    version = mlflow.register_model(model_uri, name)
    client = MlflowClient()
    if not run_id:
        run_id = (ROOT / "reports/run_id.txt").read_text(encoding="utf-8").strip()
    key = settings().get("primary_metric", "r2")
    score = client.get_run(run_id).data.metrics.get(key)
    if score is None:
        raise ValueError(f"Нет метрики {key} в запуске {run_id}")
    client.set_model_version_tag(name, version.version, "lr2_run_id", run_id)
    client.set_model_version_tag(name, version.version, f"lr2_{key}", str(score))
    client.set_registered_model_alias(name, "challenger", version.version)
    print(f"challenger: {name} v{version.version}, {key}={score:.4f}")
    return version


def promote():
    client = MlflowClient()
    cfg = settings()
    name, key = cfg["model_name"], cfg.get("primary_metric", "r2")
    candidate = client.get_model_version_by_alias(name, "challenger")
    candidate_score = metric(client, candidate, key)
    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except mlflow.exceptions.MlflowException:
        champion = None
    if champion is None:
        client.set_registered_model_alias(name, "champion", candidate.version)
        print(f"champion: v{candidate.version}, {key}={candidate_score:.4f}")
        return True
    current_score = metric(client, champion, key)
    better = candidate_score > current_score
    if better:
        client.set_registered_model_alias(name, "champion", candidate.version)
    print(f"challenger v{candidate.version}: {candidate_score:.4f}; champion v{champion.version}: {current_score:.4f}; {'PROMOTE' if better else 'KEEP'}")
    return better


def status():
    client = MlflowClient()
    cfg = settings()
    name, key = cfg["model_name"], cfg.get("primary_metric", "r2")
    output = {}
    for alias in ("champion", "challenger"):
        try:
            v = client.get_model_version_by_alias(name, alias)
            output[alias] = {
                "version": v.version,
                "run_id": v.run_id,
                key: metric(client, v, key),
            }
        except mlflow.exceptions.MlflowException:
            output[alias] = None
    path = ROOT / "reports/registry_status.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def predict():
    cfg = settings()
    frame = pd.read_csv(ROOT / "data/features/test.csv").head(5)
    x = frame.drop(columns=["quality"])
    model = mlflow.sklearn.load_model(f'models:/{cfg["model_name"]}@champion')
    if hasattr(model, "feature_names_in_"):
        x = x[list(model.feature_names_in_)]
    output = pd.DataFrame({
        "quality_true": frame["quality"],
        "quality_pred": model.predict(x),
    })
    path = ROOT / "reports/registry_predictions.csv"
    path.parent.mkdir(exist_ok=True)
    output.to_csv(path, index=False)
    print(output.to_string(index=False))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "promote", "status", "predict"])
    parser.add_argument("--model-uri")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.action == "register":
        register(args.model_uri, args.run_id)
    elif args.action == "promote":
        promote()
    elif args.action == "status":
        status()
    else:
        predict()


if __name__ == "__main__":
    main()
