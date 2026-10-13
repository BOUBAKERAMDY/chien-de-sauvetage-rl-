"""Lecture des fichiers YAML de `configs/`."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def load_config(name: str = "base") -> dict:
    with open(CONFIGS / f"{name}.yaml") as f:
        return yaml.safe_load(f)
