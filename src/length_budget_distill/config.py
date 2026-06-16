"""Configuration and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


Config = Dict[str, Any]


def load_config(path: str) -> Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(path_value: Optional[str], config: Optional[Config] = None) -> Optional[Path]:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    if config and config.get("_config_path"):
        config_dir_path = Path(config["_config_path"]).resolve().parent / path
        if config_dir_path.exists():
            return config_dir_path

    return cwd_path


def is_real_teacher(config: Config) -> bool:
    backend = config.get("teacher", {}).get("backend", "vllm")
    return backend not in {"local_rule", "fixture"}


def is_real_dataset(config: Config) -> bool:
    source = config.get("dataset", {}).get("source", "local_jsonl")
    return source not in {"local_jsonl", "inline"}


def is_real_student(config: Config) -> bool:
    model_name = config.get("student", {}).get("model_name", "")
    return bool(model_name and model_name != "REQUIRES_USER_APPROVAL")
