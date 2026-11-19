from __future__ import annotations

import os
from pathlib import Path

RUN_ID_FILE = Path("reports/run_id.txt")
MODEL_URI_FILE = Path("reports/model_uri.txt")


def enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def log_training(experiment: str, train_params: dict, model, x_example) -> str | None:
    if not enabled():
        return None
    import cloudpickle
    import mlflow
    import numpy
    import pandas
    import scipy
    import sklearn
    from mlflow.models import infer_signature

    mlflow.set_experiment(experiment)
    requirements = [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pandas.__version__}",
        f"numpy=={numpy.__version__}",
        f"scipy=={scipy.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
    ]
    signature = infer_signature(x_example, model.predict(x_example))
    with mlflow.start_run() as run:
        mlflow.log_params({k: str(v) for k, v in train_params.items()})
        try:
            info = mlflow.sklearn.log_model(
                model, name="model", signature=signature,
                input_example=x_example.head(3),
                serialization_format="cloudpickle",
                pip_requirements=requirements,
            )
        except TypeError:
            info = mlflow.sklearn.log_model(
                model, artifact_path="model", signature=signature,
                input_example=x_example.head(3),
                serialization_format="cloudpickle",
                pip_requirements=requirements,
            )
        RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        RUN_ID_FILE.write_text(run.info.run_id, encoding="utf-8")
        MODEL_URI_FILE.write_text(info.model_uri, encoding="utf-8")
        return run.info.run_id


def log_evaluation(metrics: dict) -> None:
    if not enabled():
        return
    if not RUN_ID_FILE.exists():
        raise RuntimeError("Отсутствует reports/run_id.txt: модель не записана в MLflow")
    import mlflow

    run_id = RUN_ID_FILE.read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            k: float(v) for k, v in metrics.items()
            if k in {"r2", "mae", "rmse", "n_test"}
        })
