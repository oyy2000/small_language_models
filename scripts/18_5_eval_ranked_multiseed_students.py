#!/usr/bin/env python3
"""Evaluate the base model and nine seed-17/42/73 ranked-length adapters."""

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

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256, runtime_metadata
from length_budget_distill.ranked_evaluation import completed_evaluation_evidence
from length_budget_distill.ranked_multiseed_evaluation import (
    validate_all_parent_trainings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    training_runs = validate_all_parent_trainings(config, PROJECT_ROOT)
    config_hash = canonical_sha256(
        {key: value for key, value in config.items() if key != "_config_path"}
    )
    evaluation = dict(config["evaluation"])
    output_dir = _resolve(args.output_dir)
    prediction_dir = output_dir / "predictions"
    summary_dir = output_dir / "summaries"
    log_dir = output_dir / "logs"
    for path in (prediction_dir, summary_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, Any]] = [
        {
            "model_id": str(run["model_id"]),
            "run_name": str(run["run_name"]),
            "mode": "equal_example",
            "generator_name": str(run["generator_name"]),
            "budget_name": str(run["budget_name"]),
            "seed": int(run["seed"]),
            "adapter_path": str(run["adapter_path"]),
            "adapter_config_sha256": str(run["adapter_config_sha256"]),
            "adapter_model_sha256": str(run["adapter_model_sha256"]),
            "training_data_sha256": str(run["train_sha256"]),
            "training_examples": int(run["n"]),
            "supervised_tokens": int(run["supervised_tokens"]),
        }
        for run in training_runs
    ]
    if evaluation["include_base_model"]:
        tasks.append(
            {
                "model_id": "base",
                "run_name": "base_qwen2p5_1p5b_instruct",
                "mode": "base",
                "generator_name": None,
                "budget_name": None,
                "seed": None,
                "adapter_path": None,
            }
        )
    if len(tasks) != int(evaluation["expected_run_count"]):
        raise ValueError(f"Evaluation task count mismatch: {len(tasks)}")

    entries: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for task in tasks:
        model_id = str(task["model_id"])
        entry = {
            **task,
            "prediction_path": str(prediction_dir / f"{model_id}.jsonl"),
            "summary_path": str(summary_dir / f"{model_id}.json"),
            "log_path": str(log_dir / f"{model_id}.log"),
            "eval_status": "prepared",
        }
        entries.append(entry)
        command = [
            sys.executable,
            "scripts/4_1_eval_model.py",
            "--config",
            str(config_path),
            "--model-name",
            str(config["student"]["model_name"]),
            "--split",
            str(evaluation["dataset_split"]),
            "--start-index",
            str(evaluation["start_index"]),
            "--limit",
            str(evaluation["limit"]),
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
            str(config["student"]["torch_dtype"]),
        ]
        if entry["adapter_path"]:
            command.extend(["--adapter-path", str(entry["adapter_path"])])
        commands.append(command)

    manifest_path = output_dir / "eval_manifest_formal_shard_00_of_01.json"
    if manifest_path.exists() and not args.skip_complete:
        raise FileExistsError(f"Refusing to overwrite evaluation manifest: {manifest_path}")
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "stage": "comparative_multiseed",
        "protocol_variant": config["protocol_variant"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "start_index": int(evaluation["start_index"]),
        "limit": int(evaluation["limit"]),
        "decoding": {
            key: evaluation[key]
            for key in ("max_new_tokens", "temperature", "top_p", "batch_size")
        },
        "evaluation_source_sha256": file_sha256(PROJECT_ROOT / "scripts/4_1_eval_model.py"),
        "launcher_source_sha256": file_sha256(Path(__file__).resolve()),
        "validation_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/ranked_multiseed_evaluation.py"
        ),
        "run_count": len(entries),
        "runtime": runtime_metadata(),
        "runs": entries,
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU ID.")
    max_parallel = args.max_parallel or int(evaluation["max_parallel"])
    if max_parallel <= 0 or max_parallel > len(gpu_ids):
        raise ValueError("--max-parallel must be positive and no larger than GPU count.")
    failures = _run_tasks(
        entries,
        commands,
        gpu_ids,
        max_parallel,
        evaluation,
        manifest,
        manifest_path,
        args.skip_complete,
    )
    manifest["status"] = "failed" if failures else "complete"
    _write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Evaluation failures={len(failures)}; manifest={manifest_path}")
    logging.info("ranked_multiseed_evaluation_complete manifest=%s", manifest_path)


def _run_tasks(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    evaluation: Dict[str, Any],
    manifest: Dict[str, Any],
    manifest_path: Path,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    available_gpus = list(gpu_ids)
    failures: List[Dict[str, Any]] = []
    expected_support: List[str] | None = None
    while pending or running:
        while pending and len(running) < max_parallel and available_gpus:
            entry, command = pending.pop(0)
            evidence = _completed(entry, evaluation)
            if evidence is not None:
                if not skip_complete:
                    raise FileExistsError(f"Complete evaluation already exists: {entry['model_id']}")
                expected_support = _check_support(
                    expected_support, evidence["problem_ids"], entry["model_id"]
                )
                entry.update({key: value for key, value in evidence.items() if key != "problem_ids"})
                entry["eval_status"] = "skipped_complete"
                _write_json(manifest_path, manifest)
                continue
            existing = [
                path
                for path in (Path(entry["prediction_path"]), Path(entry["summary_path"]))
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    f"Incomplete evaluation artifacts exist for {entry['model_id']}: {existing}"
                )
            gpu_id = available_gpus.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            entry["gpu_id"] = gpu_id
            entry["eval_status"] = "running"
            log_handle = Path(entry["log_path"]).open("x", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            entry["pid"] = process.pid
            running.append((process, entry, log_handle))
            _write_json(manifest_path, manifest)
            logging.info("launch_eval model=%s gpu=%s pid=%d", entry["model_id"], gpu_id, process.pid)

        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            available_gpus.append(str(entry["gpu_id"]))
            evidence = _completed(entry, evaluation) if return_code == 0 else None
            if evidence is None:
                entry["eval_status"] = "failed"
                entry["returncode"] = return_code
                failures.append(entry)
            else:
                expected_support = _check_support(
                    expected_support, evidence["problem_ids"], entry["model_id"]
                )
                entry.update({key: value for key, value in evidence.items() if key != "problem_ids"})
                entry["eval_status"] = "complete"
            logging.info("finished_eval model=%s status=%s", entry["model_id"], entry["eval_status"])
            _write_json(manifest_path, manifest)
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _completed(entry: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any] | None:
    return completed_evaluation_evidence(
        entry["prediction_path"],
        entry["summary_path"],
        expected_n=int(evaluation["limit"]),
        expected_start_index=int(evaluation["start_index"]),
        expected_split=str(evaluation["dataset_split"]),
    )


def _check_support(
    expected: List[str] | None,
    observed: List[str],
    model_id_value: Any,
) -> List[str]:
    if expected is not None and observed != expected:
        raise ValueError(f"Evaluation problem support mismatch for {model_id_value}")
    return observed if expected is None else expected


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
