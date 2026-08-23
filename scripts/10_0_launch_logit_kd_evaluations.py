#!/usr/bin/env python3
"""Launch one deterministic shard of validation or formal logit-KD evaluations."""

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
    kd_run_name,
    load_protocol,
    protocol_hash,
    read_json,
    resolve_project_path,
    runtime_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--stage", choices=["validation", "formal"], required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--launcher-shards", type=int, default=1)
    parser.add_argument("--launcher-shard-index", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _tasks(protocol: Dict[str, Any], stage: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    if stage == "validation":
        tasks.append({"method": "base", "budget": "short_128", "run_name": "base_qwen2p5_1p5b_instruct"})
        for budget in protocol["budgets"]:
            tasks.append({"method": "sft", "budget": budget, "run_name": f"sft__{budget}__seed_17"})
        parameters = [
            (float(alpha), float(temperature))
            for alpha in protocol["kd"]["alpha_grid"]
            for temperature in protocol["kd"]["temperature_grid"]
        ]
    else:
        selection = read_json(resolve_project_path(protocol["outputs"]["result_root"]) / "validation" / "selection.json")
        if selection.get("status") != "complete":
            raise ValueError("Formal evaluation requires a complete validation selection.")
        parameters = [(float(selection["selected_alpha"]), float(selection["selected_temperature"]))]
    for alpha, temperature in parameters:
        for budget in protocol["budgets"]:
            tasks.append(
                {
                    "method": "kd",
                    "budget": budget,
                    "alpha": alpha,
                    "temperature": temperature,
                    "run_name": f"kd__{kd_run_name(budget, alpha, temperature)}",
                }
            )
    return tasks


def _valid_marker(marker_path: Path) -> Dict[str, Any] | None:
    if not marker_path.is_file():
        return None
    marker = read_json(marker_path)
    prediction = Path(str(marker.get("prediction_path")))
    summary = Path(str(marker.get("summary_path")))
    if (
        marker.get("status") == "complete"
        and prediction.is_file()
        and summary.is_file()
        and marker.get("prediction_sha256") == file_sha256(prediction)
        and marker.get("summary_sha256") == file_sha256(summary)
    ):
        return marker
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.launcher_shards <= 0 or not 0 <= args.launcher_shard_index < args.launcher_shards:
        raise ValueError("Invalid launcher shard topology.")
    protocol = load_protocol(args.config)
    all_tasks = _tasks(protocol, args.stage)
    assigned = [task for index, task in enumerate(all_tasks) if index % args.launcher_shards == args.launcher_shard_index]
    if not assigned:
        raise ValueError("No evaluation tasks assigned to launcher shard.")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    max_parallel = args.max_parallel or len(gpu_ids)
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    eval_root = result_root / args.stage / "eval"
    manifest_path = eval_root / f"launcher_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
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
    commands = []
    for task in assigned:
        command = [
            sys.executable,
            "scripts/10_1_eval_logit_kd.py",
            "--config",
            args.config,
            "--stage",
            args.stage,
            "--method",
            task["method"],
            "--budget",
            task["budget"],
            "--skip-complete",
        ]
        if task["method"] == "kd":
            command.extend(["--alpha", str(task["alpha"]), "--temperature", str(task["temperature"])])
        commands.append(command)
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
            marker_path = eval_root / "markers" / f"{task['run_name']}.json"
            marker = _valid_marker(marker_path)
            if marker is not None and args.skip_complete:
                task.update({"status": "skipped_complete", "prediction_sha256": marker["prediction_sha256"]})
                continue
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            launcher_log = eval_root / "launcher_logs" / f"{task['run_name']}.log"
            launcher_log.parent.mkdir(parents=True, exist_ok=True)
            handle = launcher_log.open("w", encoding="utf-8")
            task.update({"status": "running", "gpu_id": gpu_id, "launcher_log": str(launcher_log)})
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((process, task, handle, gpu_id))
            logging.info("launched_evaluation run=%s gpu=%s", task["run_name"], gpu_id)
        still_running = []
        for process, task, handle, gpu_id in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, task, handle, gpu_id))
                continue
            handle.close()
            available.append(gpu_id)
            marker = _valid_marker(eval_root / "markers" / f"{task['run_name']}.json")
            if return_code == 0 and marker is not None:
                task.update({"status": "complete", "prediction_sha256": marker["prediction_sha256"]})
            else:
                task.update({"status": "failed", "returncode": return_code})
                failures.append(dict(task))
            logging.info("finished_evaluation run=%s status=%s", task["run_name"], task["status"])
        running = still_running
        manifest["runs"] = assigned
        write_json(manifest_path, manifest)
        if running:
            time.sleep(5)
    manifest["status"] = "failed" if failures else "complete"
    write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Evaluation failures={len(failures)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
