"""Два прогона пайплайна с нуля дают одинаковые метрики (с точностью до допуска)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/wine.csv"
STAGES = ["prepare", "featurize", "train", "evaluate"]


def run_from_scratch(workdir: Path) -> dict:
    shutil.copytree(ROOT / "src", workdir / "src")
    shutil.copy(ROOT / "params.yaml", workdir)
    (workdir / "data/raw").mkdir(parents=True)
    shutil.copy(RAW, workdir / "data/raw")
    for stage in STAGES:
        subprocess.run([sys.executable, "-m", f"src.{stage}"], cwd=workdir, check=True)
    return json.loads((workdir / "reports/metrics.json").read_text())


@pytest.mark.skipif(not RAW.exists(), reason="нет данных: выполните dvc pull")
def test_two_runs_give_same_metrics(tmp_path):
    first = run_from_scratch(tmp_path / "run1")
    second = run_from_scratch(tmp_path / "run2")
    assert first.keys() == second.keys()
    for key in first:
        assert first[key] == pytest.approx(second[key], abs=1e-9)
