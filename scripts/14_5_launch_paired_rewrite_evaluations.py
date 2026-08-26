#!/usr/bin/env python3
"""Launch selection or confirmatory paired-rewrite evaluations across GPUs."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_paired_rewrite_7b_pilot_v1.json")
    parser.add_argument("--stage", choices=["selection", "confirmatory"], required=True)
    parser.add_argument("--training-manifest-glob", required=True)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--output-dir", required=True)
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
    config = load_config(args.config)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    training_manifests = _load_training_manifests(args.training_manifest_glob)
    if args.stage == "selection":
        models = _selection_models(training_manifests)
        budgets = [int(config["evaluation"]["primary_max_new_tokens"])]
        decoding_modes = [("greedy", 0.0, 1.0, 1)]
        split = str(config["selection"]["split"])
        start_index = int(config["selection"]["start_index"])
        limit = int(config["selection"]["limit"])
    else:
        if not args.selection_json:
            raise ValueError("--selection-json is required for confirmatory evaluation")
        selection = _read_json(Path(args.selection_json))
        models = _confirmatory_models(training_manifests, selection)
        budgets = [int(value) for value in config["evaluation"]["budget_sweep"]]
        decoding_modes = [("greedy", 0.0, 1.0, 1)]
        decoding_modes.append(
            (
                "sample",
                float(config["evaluation"]["sampling_temperature"]),
                float(config["evaluation"]["sampling_top_p"]),
                int(config["evaluation"]["sampling_candidates"]),
            )
        )
        split = str(config["evaluation"]["confirmatory_split"])
        start_index = int(config["evaluation"]["confirmatory_start_index"])
        limit = int(config["evaluation"]["confirmatory_limit"])

    all_tasks: List[Dict[str, Any]] = []
    for model in models:
        for budget in budgets:
            for mode, temperature, top_p, samples in decoding_modes:
                if mode == "sample" and budget != max(budgets):
                    continue
                task_id = f"{model['model_id']}__cap_{budget}__{mode}"
                all_tasks.append(
                    {
                        **model,
                        "task_id": task_id,
                        "max_new_tokens": budget,
                        "decoding_mode": mode,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_samples": samples,
                    }
                )
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % args.launcher_shards == args.launcher_shard_index
    ]
    if not tasks:
        raise ValueError("No evaluation tasks assigned to this launcher shard")
    output_dir = Path(args.output_dir)
    for directory in (output_dir / "predictions", output_dir / "summaries", output_dir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task["prediction_path"] = str(output_dir / "predictions" / f"{task['task_id']}.jsonl")
        task["summary_path"] = str(output_dir / "summaries" / f"{task['task_id']}.json")
        task["log_path"] = str(output_dir / "logs" / f"{task['task_id']}.log")
        task["eval_status"] = "prepared"
    manifest_path = output_dir / (
        f"eval_manifest_{args.stage}_shard_{args.launcher_shard_index:02d}_"
        f"of_{args.launcher_shards:02d}.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation manifest: {manifest_path}")
    manifest: Dict[str, Any] = {
        "status": "prepared" if args.dry_run else "running",
        "stage": args.stage,
        "evidence_level": "exploratory_single_seed_pilot",
        "config_hash": config_hash,
        "split": split,
        "start_index": start_index,
        "limit": limit,
        "launcher_shards": args.launcher_shards,
        "launcher_shard_index": args.launcher_shard_index,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    _write_json(manifest_path, manifest)
    commands = [_eval_command(config, args.config, task, split, start_index, limit) for task in tasks]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    failures = _run_commands(tasks, commands, gpu_ids, max_parallel, limit, args.skip_complete)
    manifest["status"] = "failed" if failures else "complete"
    manifest["tasks"] = tasks
    _write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Paired-rewrite evaluation failures={len(failures)}")


def _load_training_manifests(pattern: str) -> List[Dict[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No training manifests matched {pattern!r}")
    manifests = [_read_json(path) for path in paths]
    for path, manifest in zip(paths, manifests):
        if manifest.get("status") != "complete":
            raise ValueError(f"Incomplete training manifest: {path}")
        manifest["_manifest_path"] = str(path)
    return manifests


def _selection_models(manifests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for manifest in manifests:
        if manifest.get("stage") != "grid":
            continue
        for snapshot in manifest["snapshots"]:
            models.append(
                {
                    "model_id": f"{manifest['run_name']}__{_epoch_name(float(snapshot['target_epoch']))}",
                    "run_name": manifest["run_name"],
                    "condition": manifest["condition"],
                    "schedule": manifest.get("schedule", "base_single_stage"),
                    "learning_rate": float(manifest["learning_rate"]),
                    "epoch": float(snapshot["target_epoch"]),
                    "adapter_path": str(snapshot["path"]),
                    "adapter_model_sha256": snapshot["adapter_model_sha256"],
                }
            )
    if not models:
        raise ValueError("No grid model snapshots found")
    return models


def _confirmatory_models(
    manifests: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if selection.get("status") != "complete":
        raise ValueError("Incomplete recipe selection")
    models: List[Dict[str, Any]] = []
    for condition, path_value in selection["selected_snapshots"].items():
        path = Path(str(path_value))
        _require_snapshot(path)
        models.append(
            {
                "model_id": f"selected__{condition}",
                "run_name": f"selected__{condition}",
                "condition": condition,
                "schedule": "base_single_stage",
                "learning_rate": float(selection["selected_learning_rate"]),
                "epoch": float(selection["selected_epoch"]),
                "adapter_path": str(path),
                "adapter_model_sha256": file_sha256(path / "adapter_model.safetensors"),
            }
        )
    for manifest in manifests:
        if manifest.get("stage") != "final":
            continue
        snapshot = manifest["snapshots"][-1]
        models.append(
            {
                "model_id": str(manifest["run_name"]),
                "run_name": manifest["run_name"],
                "condition": manifest["condition"],
                "schedule": manifest.get("schedule"),
                "learning_rate": float(manifest["learning_rate"]),
                "epoch": float(snapshot["target_epoch"]),
                "adapter_path": str(snapshot["path"]),
                "adapter_model_sha256": snapshot["adapter_model_sha256"],
            }
        )
    if len(models) < 4:
        raise ValueError("Confirmatory registry is missing selected or final-stage models")
    seen = set()
    unique = []
    for model in models:
        if model["model_id"] in seen:
            raise ValueError(f"Duplicate confirmatory model ID: {model['model_id']}")
        seen.add(model["model_id"])
        unique.append(model)
    return unique


def _eval_command(
    config: Mapping[str, Any],
    config_path: str,
    task: Mapping[str, Any],
    split: str,
    start_index: int,
    limit: int,
) -> List[str]:
    return [
        sys.executable,
        "scripts/14_5_eval_paired_rewrite_model.py",
        "--config", config_path,
        "--model-id", str(task["model_id"]),
        "--adapter-path", str(task["adapter_path"]),
        "--split", split,
        "--start-index", str(start_index),
        "--limit", str(limit),
        "--max-new-tokens", str(task["max_new_tokens"]),
        "--temperature", str(task["temperature"]),
        "--top-p", str(task["top_p"]),
        "--num-samples", str(task["num_samples"]),
        "--seed", str(config["training"]["seed"]),
        "--batch-size", str(config["evaluation"]["batch_size"]),
        "--prediction-path", str(task["prediction_path"]),
        "--summary-path", str(task["summary_path"]),
    ]


def _run_commands(
    tasks: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    expected_n: int,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    pending = list(zip(tasks, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)
    while pending or running:
        while pending and len(running) < max_parallel and (not gpu_ids or available_gpus):
            task, command = pending.pop(0)
            if _complete_eval(task, expected_n):
                if not skip_complete:
                    raise FileExistsError(f"Complete evaluation exists: {task['summary_path']}")
                task["eval_status"] = "skipped_complete"
                continue
            existing = [Path(task[key]) for key in ("prediction_path", "summary_path") if Path(task[key]).exists()]
            if existing:
                raise FileExistsError(f"Incomplete evaluation artifacts exist: {existing}")
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                task["gpu_id"] = gpu_id
            log_handle = Path(task["log_path"]).open("w", encoding="utf-8")
            task["eval_status"] = "running"
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((process, task, log_handle))
        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, task, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, task, log_handle))
                continue
            log_handle.close()
            if task.get("gpu_id") is not None:
                available_gpus.append(str(task["gpu_id"]))
            valid = return_code == 0 and _complete_eval(task, expected_n)
            task["eval_status"] = "complete" if valid else "failed"
            task["returncode"] = return_code
            if valid:
                task["prediction_sha256"] = file_sha256(task["prediction_path"])
                task["summary_sha256"] = file_sha256(task["summary_path"])
            else:
                failures.append(task)
            logging.info("paired_eval_finished task=%s status=%s", task["task_id"], task["eval_status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _complete_eval(task: Mapping[str, Any], expected_n: int) -> bool:
    summary_path = Path(str(task["summary_path"]))
    prediction_path = Path(str(task["prediction_path"]))
    if not summary_path.is_file() or not prediction_path.is_file():
        return False
    summary = _read_json(summary_path)
    return (
        summary.get("status") == "complete"
        and int(summary.get("n", -1)) == expected_n
        and int(summary.get("prediction_count", -1)) == expected_n * int(task["num_samples"])
        and summary.get("prediction_sha256") == file_sha256(prediction_path)
    )


def _require_snapshot(path: Path) -> None:
    if not (path / "SNAPSHOT_COMPLETE").is_file():
        raise FileNotFoundError(f"Snapshot completion marker is missing: {path}")


def _epoch_name(epoch: float) -> str:
    return f"epoch_{epoch:g}".replace(".", "p")


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
