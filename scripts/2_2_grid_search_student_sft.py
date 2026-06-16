#!/usr/bin/env python3
"""Launch a grid search over student SFT configs."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config


RunSpec = Tuple[str, Dict[str, Any], Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/student_sft_template.json",
        help="Base student SFT config.",
    )
    parser.add_argument(
        "--grid-config",
        default="configs/student_sft_grid_template.json",
        help="Grid definition JSON.",
    )
    parser.add_argument(
        "--work-dir",
        default="results/student_sft_grid",
        help="Directory for generated run configs, logs, and manifest.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU IDs. Example: 0,1,2,3. If omitted, runs sequentially without setting CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum concurrent runs. Defaults to number of GPU IDs, or 1 if no GPU IDs are provided.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print commands without launching training.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose checkpoint output directory already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base_config = load_config(args.base_config)
    base_config.pop("_config_path", None)
    grid_config = _read_json(Path(args.grid_config))
    runs = _build_runs(base_config, grid_config)
    work_dir = Path(args.work_dir)
    config_dir = work_dir / "configs"
    log_dir = work_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_entries = []
    for run_name, config, overrides in runs:
        config_path = config_dir / f"{run_name}.json"
        output_dir = Path(config["training"]["output_dir"])
        run_entries.append(
            {
                "run_name": run_name,
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "log_path": str(log_dir / f"{run_name}.log"),
                "overrides": overrides,
            }
        )
        _write_json(config_path, config)

    manifest_path = work_dir / "manifest.json"
    _write_json(manifest_path, {"runs": run_entries})
    logging.info("prepared_runs=%d manifest=%s", len(run_entries), manifest_path)

    commands = [
        [
            sys.executable,
            "scripts/2_1_train_student_sft.py",
            "--config",
            entry["config_path"],
        ]
        for entry in run_entries
    ]
    if args.dry_run:
        for entry, command in zip(run_entries, commands):
            print(_format_command(command, entry.get("gpu_id")))
        return

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive.")

    failures = _run_commands(run_entries, commands, gpu_ids, max_parallel, args.skip_existing)
    if failures:
        logging.error("failed_runs=%d", len(failures))
        for entry in failures:
            logging.error("failed run=%s log=%s", entry["run_name"], entry["log_path"])
        raise SystemExit(1)
    logging.info("all grid-search runs finished successfully")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_runs(base_config: Dict[str, Any], grid_config: Dict[str, Any]) -> List[RunSpec]:
    grid = grid_config.get("grid", {})
    if not grid:
        raise ValueError("Grid config must define a non-empty 'grid' object.")

    fixed_overrides = grid_config.get("fixed_overrides", {})
    checkpoint_root = Path(grid_config.get("checkpoint_root", "checkpoints/student_sft_grid"))
    grid_name = _slugify(str(grid_config.get("name", "student_sft_grid")))
    keys = list(grid)
    values = [_as_list(grid[key]) for key in keys]
    runs: List[RunSpec] = []
    for index, combination in enumerate(itertools.product(*values)):
        overrides = dict(zip(keys, combination))
        config = copy.deepcopy(base_config)
        for key, value in fixed_overrides.items():
            _set_path(config, key, value)
        for key, value in overrides.items():
            _set_path(config, key, value)
        run_name = _run_name(grid_name, index, overrides)
        config["experiment_name"] = run_name
        config.setdefault("training", {})["output_dir"] = str(checkpoint_root / run_name)
        runs.append((run_name, config, overrides))
    return runs


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else [value]


def _set_path(config: Dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ValueError(f"Cannot set {dotted_path!r}; {part!r} is not an object.")
    cursor[parts[-1]] = value


def _run_name(grid_name: str, index: int, overrides: Dict[str, Any]) -> str:
    parts = [grid_name, f"{index:03d}"]
    for key, value in overrides.items():
        key_name = key.split(".")[-1]
        parts.append(f"{_slugify(key_name)}-{_slugify(_short_value(value))}")
    return "_".join(parts)


def _short_value(value: Any) -> str:
    if isinstance(value, str):
        path = Path(value)
        if path.suffix:
            return path.stem
        return value
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _slugify(value: str) -> str:
    value = value.replace(".", "p")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return value.lower() or "value"


def _parse_gpu_ids(raw_gpu_ids: str | None) -> List[str]:
    if not raw_gpu_ids:
        return []
    return [item.strip() for item in raw_gpu_ids.split(",") if item.strip()]


def _run_commands(
    run_entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(run_entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)

    while pending or running:
        can_launch = lambda: pending and len(running) < max_parallel and (not gpu_ids or available_gpus)
        while can_launch():
            entry, command = pending.pop(0)
            if skip_existing and Path(entry["output_dir"]).exists():
                logging.info("skip_existing run=%s output_dir=%s", entry["run_name"], entry["output_dir"])
                continue
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                entry["gpu_id"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            logging.info("launch run=%s gpu=%s log=%s", entry["run_name"], gpu_id or "default", entry["log_path"])
            logging.info("command=%s", _format_command(command, gpu_id))
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((process, entry, log_handle))

        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            code = process.poll()
            if code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            gpu_id = entry.get("gpu_id")
            if gpu_id is not None:
                available_gpus.append(gpu_id)
            if code == 0:
                logging.info("finished run=%s", entry["run_name"])
            else:
                entry["returncode"] = code
                failures.append(entry)
                logging.error("failed run=%s returncode=%s", entry["run_name"], code)
        running = still_running
        if running:
            time.sleep(5)

    return failures


def _format_command(command: Iterable[str], gpu_id: str | None = None) -> str:
    rendered = " ".join(command)
    if gpu_id is not None:
        return f"CUDA_VISIBLE_DEVICES={gpu_id} {rendered}"
    return rendered


if __name__ == "__main__":
    main()
