#!/usr/bin/env python3
"""Launch one deterministic shard of validation-grid or formal KD training runs."""

from __future__ import annotations

import argparse
import json
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
    kd_run_name,
    load_protocol,
    protocol_hash,
    read_json,
    resolve_project_path,
    runtime_metadata,
    validated_training_marker,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--stage", choices=["validation", "formal"], required=True)
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated local GPU IDs.")
    parser.add_argument("--launcher-shards", type=int, default=1)
    parser.add_argument("--launcher-shard-index", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _tasks(protocol: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    tasks = []
    if stage == "validation":
        parameters = [
            (float(alpha), float(temperature))
            for alpha in protocol["kd"]["alpha_grid"]
            for temperature in protocol["kd"]["temperature_grid"]
        ]
    else:
        selection_path = resolve_project_path(protocol["outputs"]["result_root"]) / "validation" / "selection.json"
        selection = read_json(selection_path)
        if selection.get("status") != "complete":
            raise ValueError("Formal launch requires a complete validation selection.")
        parameters = [(float(selection["selected_alpha"]), float(selection["selected_temperature"]))]
    for alpha, temperature in parameters:
        for budget_name in protocol["budgets"]:
            run_name = kd_run_name(budget_name, alpha, temperature)
            if stage == "formal":
                output_dir = resolve_project_path(protocol["outputs"]["checkpoint_root"]) / "formal" / f"{budget_name}__seed_17"
            else:
                output_dir = resolve_project_path(protocol["outputs"]["checkpoint_root"]) / stage / run_name
            tasks.append(
                {
                    "run_name": run_name,
                    "budget_name": budget_name,
                    "alpha": alpha,
                    "temperature": temperature,
                    "output_dir": str(output_dir),
                }
            )
    return tasks


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.launcher_shards <= 0 or not 0 <= args.launcher_shard_index < args.launcher_shards:
        raise ValueError("Invalid launcher shard topology.")
    protocol = load_protocol(args.config)
    all_tasks = _tasks(protocol, args.stage)
    assigned = [task for index, task in enumerate(all_tasks) if index % args.launcher_shards == args.launcher_shard_index]
    if not assigned:
        raise ValueError("No KD training tasks assigned to launcher shard.")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    max_parallel = args.max_parallel or len(gpu_ids)
    if max_parallel <= 0 or max_parallel > len(gpu_ids):
        raise ValueError("max_parallel must be between one and the number of GPUs.")
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    work_dir = result_root / args.stage / "training"
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / f"launcher_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "stage": args.stage,
        "protocol_hash": protocol_hash(protocol),
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "runtime": runtime_metadata(),
        "runs": assigned,
    }
    write_json(manifest_path, manifest)
    commands: List[List[str]] = []
    for task in assigned:
        commands.append(
            [
                sys.executable,
                "scripts/9_1_train_logit_kd.py",
                "--config",
                args.config,
                "--stage",
                args.stage,
                "--budget",
                task["budget_name"],
                "--alpha",
                str(task["alpha"]),
                "--temperature",
                str(task["temperature"]),
                "--skip-complete",
            ]
        )
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    pending = list(zip(assigned, commands))
    available = list(gpu_ids)
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
    failures = []
    while pending or running:
        while pending and available and len(running) < max_parallel:
            task, command = pending.pop(0)
            marker = validated_training_marker(task["output_dir"])
            if marker is not None and args.skip_complete:
                task["status"] = "skipped_complete"
                task["adapter_model_sha256"] = marker["adapter_model_sha256"]
                continue
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_path = log_dir / f"{task['run_name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            task.update({"status": "running", "gpu_id": gpu_id, "log_path": str(log_path)})
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((process, task, handle, gpu_id))
            logging.info("launched_kd_training run=%s gpu=%s", task["run_name"], gpu_id)
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
            logging.info("finished_kd_training run=%s status=%s", task["run_name"], task["status"])
        running = still_running
        manifest["runs"] = assigned
        write_json(manifest_path, manifest)
        if running:
            time.sleep(5)
    manifest["status"] = "failed" if failures else "complete"
    write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"KD training failures={len(failures)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
