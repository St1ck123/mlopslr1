"""Общие функции: параметры, журналирование, создание каталогов."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path("params.yaml")


def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    """Читает params.yaml. safe_load не исполняет произвольные теги YAML."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise TypeError(f"{path}: ожидался словарь параметров")
    return params


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(name)-10s %(message)s", force=False
    )
    return logging.getLogger(name)


def ensure_parent(path: Path | str) -> Path:
    """Создаёт каталог для файла, если его нет, и возвращает путь."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
