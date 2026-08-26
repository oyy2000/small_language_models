#!/usr/bin/env python3
"""Prepare an experiment-scoped BeeGFS result root with a stable project symlink."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument(
        "--beegfs-project-root",
        default="/mnt/beegfs/youyang7/projects/small_language_model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        protocol: Dict[str, Any] = json.load(handle)
    experiment_name = str(protocol["experiment_name"])
    stable_result_root = _resolve(protocol["outputs"]["result_root"])
    expected_stable_root = PROJECT_ROOT / "results" / experiment_name
    if stable_result_root != expected_stable_root:
        raise ValueError(
            f"OPD result root must be the registered stable project path: {expected_stable_root}"
        )
    beegfs_project_root = Path(args.beegfs_project_root).resolve()
    if Path("/mnt/beegfs") not in (beegfs_project_root, *beegfs_project_root.parents):
        raise ValueError("BeeGFS project root must be located under /mnt/beegfs.")
    target_result_root = beegfs_project_root / "results" / experiment_name

    if stable_result_root.is_symlink():
        if stable_result_root.resolve() != target_result_root.resolve():
            raise ValueError(
                f"Existing OPD result symlink targets an unexpected path: {stable_result_root}"
            )
        if not target_result_root.is_dir():
            raise ValueError(f"OPD BeeGFS result target is missing: {target_result_root}")
    elif stable_result_root.exists():
        raise FileExistsError(
            "Refusing to replace an existing non-symlink OPD result root: "
            f"{stable_result_root}"
        )
    else:
        if target_result_root.exists():
            if not target_result_root.is_dir() or any(target_result_root.iterdir()):
                raise FileExistsError(
                    "Refusing to adopt a pre-existing non-empty BeeGFS result target: "
                    f"{target_result_root}"
                )
        else:
            target_result_root.mkdir(parents=True, exist_ok=False)
        stable_result_root.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target_result_root, stable_result_root, target_is_directory=True)

    checkpoint_root = _resolve(protocol["outputs"]["checkpoint_root"]).resolve()
    if Path("/mnt/beegfs") not in (checkpoint_root, *checkpoint_root.parents):
        raise ValueError(f"Registered checkpoint root is not on BeeGFS: {checkpoint_root}")
    print(
        json.dumps(
            {
                "status": "ready",
                "stable_result_root": str(stable_result_root),
                "beegfs_result_root": str(target_result_root),
                "checkpoint_root": str(checkpoint_root),
            },
            indent=2,
        )
    )


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
