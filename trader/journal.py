"""JSON files under docs/data that the dashboard reads. Committed to git by the workflow."""

from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_DIR


def _path(name: str) -> Path:
    return DATA_DIR / name


def read(name: str, default):
    p = _path(name)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write(name: str, data) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path(name).write_text(json.dumps(data, indent=1, default=str) + "\n")


def upsert_equity(row: dict) -> list[dict]:
    rows = [r for r in read("equity.json", []) if r["date"] != row["date"]]
    rows.append(row)
    rows.sort(key=lambda r: r["date"])
    write("equity.json", rows)
    return rows


def append_decision(record: dict) -> None:
    rows = read("decisions.json", [])
    rows.append(record)
    write("decisions.json", rows)
