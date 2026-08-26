#!/usr/bin/env python3
"""Launch prepared paired-rewrite SFT runs across disjoint local GPUs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--launcher-shards", type=int, default=1)
    parser.add_argument("--launcher-shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.launcher_shards <= 0 or not 0 <= args.launcher_shard_index < args.launcher_shards:
        raise ValueError("Invalid launcher shard topology")
    prepared_path = Path(args.prepared_manifest)
    prepared = _read_json(prepared_path)
    if prepared.get("status") != "prepared":
        raise ValueError(f"Prepared training manifest has invalid status: {prepared_path}")
    all_runs = list(prepared.get("runs", []))
    runs = [
        dict(run)
        for index, run in enumerate(all_runs)
        if index % args.launcher_shards == args.launcher_shard_index
    ]
    if not runs:
        raise ValueError("No prepared training runs assigned to this launcher shard")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["log_path"] = str(log_dir / f"{run['run_name']}.log")
        run["launch_status"] = "prepared"
    output_manifest = Path(args.output_manifest)
    if output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite launch manifest: {output_manifest}")
    launch_manifest: Dict[str, Any] = {
        "status": "prepared" if args.dry_run else "running",
        "stage": prepared["stage"],
        "config_hash": prepared["config_hash"],
        "prepared_manifest": str(prepared_path),
        "launcher_shards": args.launcher_shards,
        "launcher_shard_index": args.launcher_shard_index,
        "runs": runs,
    }
    _write_json(output_manifest, launch_manifest)
    commands = [
        [
            sys.executable,
            "scripts/14_4_train_paired_rewrite_student.py",
            "--config",
            str(run["config_path"]),
            *( ["--skip-complete"] if args.skip_complete else [] ),
        ]
        for run in runs
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    failures = _run_commands(runs, commands, gpu_ids, max_parallel)
    launch_manifest["status"] = "failed" if failures else "complete"
    launch_manifest["runs"] = runs
    _write_json(output_manifest, launch_manifest)
    if failures:
        raise SystemExit(f"Paired-rewrite training failures={len(failures)}")


def _run_commands(
    runs: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
) -> List[Dict[str, Any]]:
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    runtime_root_value = os.environ.get("LBD_RUNTIME_CHECKPOINT_ROOT")
    if not runtime_root_value:
        raise ValueError("LBD_RUNTIME_CHECKPOINT_ROOT is required and must be node-local")
    runtime_root = Path(runtime_root_value)
    runtime_root.mkdir(parents=True, exist_ok=True)
    pending = list(zip(runs, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)
    while pending or running:
        while pending and len(running) < max_parallel and (not gpu_ids or available_gpus):
            run, command = pending.pop(0)
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            runtime_dir = runtime_root / str(run["run_name"])
            if runtime_dir.exists() and any(runtime_dir.iterdir()):
                raise FileExistsError(f"Node-local run directory is non-empty: {runtime_dir}")
            env = os.environ.copy()
            env["LBD_RUNTIME_OUTPUT_DIR"] = str(runtime_dir)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                run["gpu_id"] = gpu_id
            log_handle = Path(run["log_path"]).open("w", encoding="utf-8")
            run["runtime_output_dir"] = str(runtime_dir)
            run["launch_status"] = "running"
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append((process, run, log_handle))
        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, run, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, run, log_handle))
                continue
            log_handle.close()
            if run.get("gpu_id") is not None:
                available_gpus.append(str(run["gpu_id"]))
            manifest_path = Path(run["output_dir"]) / "training_manifest.json"
            valid = return_code == 0 and _complete_training_manifest(manifest_path, run)
            run["launch_status"] = "complete" if valid else "failed"
            run["returncode"] = return_code
            run["training_manifest"] = str(manifest_path)
            if not valid:
                failures.append(run)
            logging.info("paired_training_finished run=%s status=%s", run["run_name"], run["launch_status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _complete_training_manifest(path: Path, run: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    manifest = _read_json(path)
    return (
        manifest.get("status") == "complete"
        and manifest.get("run_name") == run["run_name"]
        and manifest.get("train_sha256") == run["train_sha256"]
        and len(manifest.get("snapshots", [])) == len(run["snapshot_epochs"])
    )


def _parse_gpu_ids(raw: str | None) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()] if raw else []


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
