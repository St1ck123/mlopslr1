"""Имитация выгрузки данных из источника.

Полный файл лежит в data/external/ (вне Git и вне DVC; если его нет, он
скачивается с UCI). Версия данных — это первые n_rows строк: v1 — 1000 вин,
v2 — все 1599, то есть к v1 добавляется новая порция.

    python -m src.ingest           # n_rows из params.yaml (v1)
    python -m src.ingest --all     # весь набор (v2)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from src.utils import ensure_parent, get_logger, load_params

log = get_logger("ingest")


def main() -> None:
    params = load_params()
    ingest = params["ingest"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=ingest["n_rows"])
    parser.add_argument("--all", action="store_true", help="взять весь набор")
    args = parser.parse_args()

    source = Path(ingest["source"])
    if not source.exists():
        urllib.request.urlretrieve(ingest["url"], ensure_parent(source))
        log.info("скачан %s", source)

    frame = pd.read_csv(source, sep=";")
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    if not args.all:
        frame = frame.head(args.n_rows)

    out = ensure_parent(params["data"]["raw"])
    frame.to_csv(out, index=False)
    log.info("выгрузка: %d строк -> %s", len(frame), out)


if __name__ == "__main__":
    main()
