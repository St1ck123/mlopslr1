import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.train import make_model


@pytest.mark.parametrize("config", [
    {"model": "ridge", "alpha": 1.0},
    {"model": "random_forest", "n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
    {"model": "gradient_boosting", "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1},
])
def test_models_fit_predict(config):
    features, target = make_regression(n_samples=30, n_features=5, random_state=42)
    frame = pd.DataFrame(features, columns=[f"x{i}" for i in range(5)])
    model = make_model({**config, "seed": 42})
    model.fit(frame, target)
    assert len(model.predict(frame.iloc[:3])) == 3
