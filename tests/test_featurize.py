import pandas as pd

from src.featurize import build
from src.prepare import split


def test_extra_features_and_target_last():
    frame = pd.DataFrame(
        {
            "fixed_acidity": [7.0],
            "volatile_acidity": [0.5],
            "free_sulfur_dioxide": [10.0],
            "total_sulfur_dioxide": [40.0],
            "quality": [5],
        }
    )
    out = build(frame, "quality", extra_features=True)
    assert out.columns[-1] == "quality"
    assert out.loc[0, "free_so2_share"] == 0.25
    assert out.loc[0, "total_acidity"] == 7.5


def test_ordered_split_keeps_tail_in_test():
    frame = pd.DataFrame({"x": range(10)})
    train, test = split(frame, 0.2, "ordered", seed=0)
    assert list(test["x"]) == [8, 9]
    assert len(train) == 8
