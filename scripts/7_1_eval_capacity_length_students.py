#!/usr/bin/env python3
"""Evaluate formal capacity-length adapters and the unfine-tuned base student."""

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
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    runtime_metadata,
    validated_adapter_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument(
        "--evaluation-config",
        default="configs/capacity_length_factorial_eval_v1.json",
        help="Stage-specific evaluation overrides bound to the parent protocol hash.",
    )
    parser.add_argument("--training-manifest-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
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
        raise ValueError("Invalid launcher shard topology.")
    config = load_config(args.config)
    config.pop("_config_path", None)
    config_hash = canonical_sha256(config)
    evaluation_overlay = _read_json(Path(args.evaluation_config))
    if evaluation_overlay.get("parent_config_sha256") != config_hash:
        raise ValueError(
            "Evaluation overlay parent hash mismatch: "
            f"overlay={evaluation_overlay.get('parent_config_sha256')} config={config_hash}"
        )
    evaluation = dict(config["evaluation"])
    evaluation.update(dict(evaluation_overlay.get("evaluation_overrides", {})))
    evaluation_overlay_sha256 = file_sha256(args.evaluation_config)
    start_index = int(evaluation[f"{args.stage}_start_index"])
    limit = int(evaluation[f"{args.stage}_limit"])
    expected_n = limit

    manifest_paths = [Path(path) for path in sorted(glob.glob(args.training_manifest_glob))]
    if not manifest_paths:
        raise FileNotFoundError(f"No training manifests matched {args.training_manifest_glob!r}")
    by_run: Dict[str, Dict[str, Any]] = {}
    for path in manifest_paths:
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            raise ValueError(f"Training manifest is not complete: {path} status={manifest.get('status')}")
        if manifest.get("config_hash") != config_hash:
            raise ValueError(f"Training/config hash mismatch in {path}")
        for run in manifest.get("runs", []):
            run_name = str(run["run_name"])
            if run_name in by_run:
                raise ValueError(f"Duplicate training run across manifests: {run_name}")
            if run.get("status") not in {"complete", "skipped_complete"}:
                raise ValueError(f"Training run is incomplete: {run_name} status={run.get('status')}")
            adapter_path = Path(run["output_dir"])
            adapter_evidence = validated_adapter_evidence(adapter_path)
            if adapter_evidence is None:
                raise FileNotFoundError(f"Training adapter lacks completion evidence: {adapter_path}")
            for evidence_name in (
                "train_sha256",
                "run_config_sha256",
                "training_source_sha256",
                "launcher_source_sha256",
                "adapter_config_sha256",
                "adapter_model_sha256",
            ):
                if run.get(evidence_name) != adapter_evidence[evidence_name]:
                    raise ValueError(
                        f"Training manifest adapter evidence mismatch for {run_name}: {evidence_name}"
                    )
            by_run[run_name] = run

    all_runs = [by_run[key] for key in sorted(by_run)]
    all_runs.append(
        {
            "run_name": "base_qwen2p5_1p5b_instruct",
            "output_dir": None,
            "status": "base_model",
            "mode": "base",
            "seed": None,
            "generator_name": None,
            "budget_name": None,
            "baseline_name": "base_model",
            "overrides": {"mode": "base", "baseline_name": "base_model"},
        }
    )
    assigned_runs = [
        run
        for index, run in enumerate(all_runs)
        if index % args.launcher_shards == args.launcher_shard_index
    ]
    if not assigned_runs:
        raise ValueError("No evaluation runs were assigned to this launcher shard.")

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    summary_dir = output_dir / "summaries"
    log_dir = output_dir / "logs"
    for path in (prediction_dir, summary_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for run in assigned_runs:
        run_name = str(run["run_name"])
        entry = {
            **run,
            "prediction_path": str(prediction_dir / f"{run_name}.jsonl"),
            "summary_path": str(summary_dir / f"{run_name}.json"),
            "log_path": str(log_dir / f"{run_name}.log"),
            "eval_status": "prepared",
        }
        entries.append(entry)
        command = [
            sys.executable,
            "scripts/4_1_eval_model.py",
            "--config",
            args.config,
            "--model-name",
            config["student"]["model_name"],
            "--split",
            evaluation["dataset_split"],
            "--start-index",
            str(start_index),
            "--limit",
            str(limit),
            "--output-jsonl",
            entry["prediction_path"],
            "--summary-json",
            entry["summary_path"],
            "--max-new-tokens",
            str(evaluation["max_new_tokens"]),
            "--temperature",
            str(evaluation["temperature"]),
            "--top-p",
            str(evaluation["top_p"]),
            "--batch-size",
            str(evaluation["batch_size"]),
            "--torch-dtype",
            "bfloat16",
        ]
        if run.get("output_dir"):
            command.extend(["--adapter-path", str(run["output_dir"])])
        commands.append(command)

    manifest_path = output_dir / (
        f"eval_manifest_{args.stage}_shard_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    )
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "stage": args.stage,
        "config_hash": config_hash,
        "evaluation_overlay": args.evaluation_config,
        "evaluation_overlay_sha256": evaluation_overlay_sha256,
        "start_index": start_index,
        "limit": limit,
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "run_count": len(entries),
        "runtime": runtime_metadata(),
        "runs": entries,
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    failures = _run_commands(entries, commands, gpu_ids, max_parallel, expected_n, args.skip_complete)
    manifest["status"] = "failed" if failures else "complete"
    manifest["runs"] = entries
    _write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Evaluation failures={len(failures)}; manifest={manifest_path}")
    logging.info("evaluation_shard_complete runs=%d manifest=%s", len(entries), manifest_path)


def _run_commands(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    expected_n: int,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive.")
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)
    while pending or running:
        while pending and len(running) < max_parallel and (not gpu_ids or available_gpus):
            entry, command = pending.pop(0)
            if _complete_eval(entry, expected_n):
                if skip_complete:
                    entry["prediction_sha256"] = file_sha256(entry["prediction_path"])
                    entry["summary_sha256"] = file_sha256(entry["summary_path"])
                    entry["eval_status"] = "skipped_complete"
                    continue
                raise FileExistsError(f"Complete evaluation already exists: {entry['summary_path']}")
            existing = [path for path in (Path(entry["prediction_path"]), Path(entry["summary_path"])) if path.exists()]
            if existing:
                raise FileExistsError(f"Incomplete evaluation artifacts exist for {entry['run_name']}: {existing}")
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                entry["gpu_id"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            entry["eval_status"] = "running"
            logging.info("launch_eval run=%s gpu=%s", entry["run_name"], gpu_id or "default")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((process, entry, log_handle))
        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            gpu_id = entry.get("gpu_id")
            if gpu_id is not None:
                available_gpus.append(str(gpu_id))
            if return_code == 0 and _complete_eval(entry, expected_n):
                entry["prediction_sha256"] = file_sha256(entry["prediction_path"])
                entry["summary_sha256"] = file_sha256(entry["summary_path"])
                entry["eval_status"] = "complete"
            else:
                entry["eval_status"] = "failed"
                entry["returncode"] = return_code
                failures.append(entry)
            logging.info("finished_eval run=%s status=%s", entry["run_name"], entry["eval_status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _complete_eval(entry: Dict[str, Any], expected_n: int) -> bool:
    prediction_path = Path(entry["prediction_path"])
    summary_path = Path(entry["summary_path"])
    if not prediction_path.is_file() or not summary_path.is_file():
        return False
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary.get("n", -1)) != expected_n:
        return False
    with prediction_path.open("r", encoding="utf-8") as handle:
        if sum(1 for line in handle if line.strip()) != expected_n:
            return False
    expected_prediction_hash = entry.get("prediction_sha256")
    expected_summary_hash = entry.get("summary_sha256")
    if expected_prediction_hash is not None and expected_prediction_hash != file_sha256(prediction_path):
        return False
    if expected_summary_hash is not None and expected_summary_hash != file_sha256(summary_path):
        return False
    return True


def _parse_gpu_ids(raw: str | None) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
