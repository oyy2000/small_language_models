#!/usr/bin/env python3
"""Launch a teacher-filtered shard of frozen multi-teacher logit-KD runs."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    file_sha256,
    read_json,
    resolve_project_path,
    runtime_metadata,
    validated_training_marker,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--teacher-names", required=True, help="Comma-separated generator names.")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol_path = resolve_project_path(args.protocol)
    frozen = read_json(protocol_path)
    teacher_names = {
        value.strip()
        for value in args.teacher_names.replace(":", ",").split(",")
        if value.strip()
    }
    registered_names = {str(item["generator_name"]) for item in frozen["conditions"]}
    if not teacher_names or not teacher_names <= registered_names:
        raise ValueError(
            f"Invalid teacher filter={sorted(teacher_names)} registered={sorted(registered_names)}"
        )
    tasks = [
        {
            "condition_id": str(condition["condition_id"]),
            "generator_name": str(condition["generator_name"]),
            "budget_name": str(condition["budget_name"]),
        }
        for condition in frozen["conditions"]
        if str(condition["generator_name"]) in teacher_names
    ]
    if not tasks:
        raise ValueError("No KD training conditions matched the teacher filter")
    checkpoint_root = resolve_project_path(frozen["outputs"]["checkpoint_root"])
    for task in tasks:
        task["output_dir"] = str(
            checkpoint_root / "logit_kd" / f"{task['condition_id']}__seed_17"
        )
        task["run_name"] = f"logit_kd__{task['condition_id']}__seed_17"
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    max_parallel = args.max_parallel or len(gpu_ids)
    if not 0 < max_parallel <= len(gpu_ids):
        raise ValueError("max_parallel must be in 1..len(gpu_ids)")

    result_root = resolve_project_path(frozen["outputs"]["result_root"])
    work_root = result_root / "pilot" / "kd_training"
    teacher_slug = "_".join(sorted(teacher_names))
    log_root = work_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest_path = work_root / f"launcher_{teacher_slug}.json"
    if manifest_path.exists() and not args.skip_complete:
        raise FileExistsError(f"Refusing to overwrite launcher manifest: {manifest_path}")
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "frozen_protocol_path": str(protocol_path),
        "frozen_protocol_sha256": file_sha256(protocol_path),
        "teacher_names": sorted(teacher_names),
        "runtime": runtime_metadata(),
        "runs": tasks,
    }
    write_json(manifest_path, manifest)
    commands = [
        [
            sys.executable,
            "scripts/13_4_train_multiteacher_logit_kd.py",
            "--protocol",
            str(protocol_path),
            "--condition-id",
            task["condition_id"],
            "--publish-dir",
            task["output_dir"],
            "--skip-complete",
        ]
        for task in tasks
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    failures = _run(tasks, commands, gpu_ids, max_parallel, log_root, args.skip_complete)
    manifest["status"] = "failed" if failures else "complete"
    manifest["runs"] = tasks
    write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Multi-teacher KD failures={len(failures)} manifest={manifest_path}")
    logging.info("multiteacher_kd_launcher_complete teachers=%s runs=%d", teacher_slug, len(tasks))


def _run(
    tasks: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    log_root: Path,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(tasks, commands))
    available = list(gpu_ids)
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
    failures: List[Dict[str, Any]] = []
    while pending or running:
        while pending and available and len(running) < max_parallel:
            task, command = pending.pop(0)
            marker = validated_training_marker(task["output_dir"])
            if marker is not None and skip_complete:
                task["status"] = "skipped_complete"
                task["adapter_model_sha256"] = marker["adapter_model_sha256"]
                continue
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_path = log_root / f"{task['run_name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            task.update({"status": "running", "gpu_id": gpu_id, "log_path": str(log_path)})
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((process, task, handle, gpu_id))
            logging.info("launched_multiteacher_kd run=%s gpu=%s", task["run_name"], gpu_id)
        still_running = []
        for process, task, handle, gpu_id in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, task, handle, gpu_id))
                continue
            handle.close()
            available.append(gpu_id)
            marker = validated_training_marker(task["output_dir"])
            if return_code == 0 and marker is not None:
                task["status"] = "complete"
                task["adapter_model_sha256"] = marker["adapter_model_sha256"]
            else:
                task.update({"status": "failed", "returncode": return_code})
                failures.append(dict(task))
            logging.info("finished_multiteacher_kd run=%s status=%s", task["run_name"], task["status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


if __name__ == "__main__":
    main()
