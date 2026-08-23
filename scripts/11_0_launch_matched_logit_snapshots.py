#!/usr/bin/env python3
"""Launch matched-logit extraction tasks across stable idle GPUs."""

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

from length_budget_distill.logit_kd import load_protocol, protocol_hash, read_json, resolve_project_path, runtime_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--launcher-shards", type=int, default=1)
    parser.add_argument("--launcher-shard-index", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _task_complete(output_dir: Path, num_shards: int) -> bool:
    markers = sorted(output_dir.glob("shard_*_of_*.complete.json"))
    if len(markers) != num_shards:
        return False
    observed = set()
    for path in markers:
        marker = read_json(path)
        if marker.get("status") != "complete" or int(marker.get("num_shards", -1)) != num_shards:
            return False
        observed.add(int(marker["shard_index"]))
    return observed == set(range(num_shards))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.launcher_shards <= 0 or not 0 <= args.launcher_shard_index < args.launcher_shards:
        raise ValueError("Invalid launcher shard topology.")
    protocol = load_protocol(args.config)
    num_shards = int(protocol["outputs"]["logit_shards_per_snapshot"])
    all_tasks = [
        {"budget": budget, "method": method, "run_name": f"{budget}__{method}"}
        for budget in protocol["budgets"]
        for method in ("teacher", "base", "sft", "kd")
    ]
    assigned = [task for index, task in enumerate(all_tasks) if index % args.launcher_shards == args.launcher_shard_index]
    if not assigned:
        raise ValueError("No logit tasks assigned to launcher shard.")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    max_parallel = args.max_parallel or len(gpu_ids)
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    logits_root = result_root / "formal" / "logits"
    manifest_path = logits_root / f"launcher_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "protocol_hash": protocol_hash(protocol),
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "runtime": runtime_metadata(),
        "runs": assigned,
    }
    write_json(manifest_path, manifest)
    commands = [
        [
            sys.executable,
            "scripts/11_1_extract_matched_logit_snapshots.py",
            "--config",
            args.config,
            "--budget",
            task["budget"],
            "--method",
            task["method"],
            "--num-shards",
            str(num_shards),
            "--shard-index",
            "-1",
            "--skip-complete",
        ]
        for task in assigned
    ]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    pending = list(zip(assigned, commands))
    available = list(gpu_ids)
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any, str]] = []
    failures = []
    log_dir = logits_root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and available and len(running) < max_parallel:
            task, command = pending.pop(0)
            output_dir = logits_root / task["budget"] / task["method"]
            if _task_complete(output_dir, num_shards) and args.skip_complete:
                task["status"] = "skipped_complete"
                continue
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_path = log_dir / f"{task['run_name']}.log"
            handle = log_path.open("w", encoding="utf-8")
            task.update({"status": "running", "gpu_id": gpu_id, "log_path": str(log_path)})
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((process, task, handle, gpu_id))
            logging.info("launched_logit_snapshot run=%s gpu=%s", task["run_name"], gpu_id)
        still_running = []
        for process, task, handle, gpu_id in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, task, handle, gpu_id))
                continue
            handle.close()
            available.append(gpu_id)
            complete = _task_complete(logits_root / task["budget"] / task["method"], num_shards)
            if return_code == 0 and complete:
                task["status"] = "complete"
            else:
                task.update({"status": "failed", "returncode": return_code})
                failures.append(dict(task))
            logging.info("finished_logit_snapshot run=%s status=%s", task["run_name"], task["status"])
        running = still_running
        manifest["runs"] = assigned
        write_json(manifest_path, manifest)
        if running:
            time.sleep(5)
    manifest["status"] = "failed" if failures else "complete"
    write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Logit snapshot failures={len(failures)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
