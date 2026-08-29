#!/usr/bin/env python3
"""Evaluate a disjoint tail of the main matrix without mutating its manifest."""

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
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256, runtime_metadata
from length_budget_distill.ranked_evaluation import completed_evaluation_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--task-start-index", type=int, required=True)
    parser.add_argument("--task-end-index", type=int, required=True, help="Exclusive task index.")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--helper-manifest", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    evaluation = dict(config["evaluation"])
    authoritative_path = _resolve(args.eval_manifest)
    authoritative = _read_json(authoritative_path)
    if authoritative.get("status") not in {"running", "complete"}:
        raise ValueError("Authoritative evaluation manifest is not active or complete.")
    if authoritative.get("config_hash") != config_hash:
        raise ValueError("Authoritative evaluation/config hash mismatch.")
    all_entries = list(authoritative.get("runs", []))
    if len(all_entries) != int(evaluation["expected_run_count"]):
        raise ValueError("Authoritative evaluation task count mismatch.")
    if not 0 <= args.task_start_index < args.task_end_index <= len(all_entries):
        raise ValueError("Invalid helper task interval.")
    selected = [dict(row) for row in all_entries[args.task_start_index : args.task_end_index]]
    selected_ids = [str(row.get("model_id", "")) for row in selected]
    if not all(selected_ids) or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Helper task identities are missing or duplicated.")
    active_ids = {
        str(row.get("model_id", ""))
        for row in all_entries
        if row.get("eval_status") == "running"
    }
    overlap = active_ids.intersection(selected_ids)
    if overlap:
        raise ValueError(f"Helper interval overlaps authoritative running tasks: {sorted(overlap)}")

    helper_path = _resolve(args.helper_manifest)
    if helper_path.exists():
        raise FileExistsError(f"Refusing to overwrite helper manifest: {helper_path}")
    helper_log_dir = helper_path.parent / "logs"
    helper_log_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for source in selected:
        entry = dict(source)
        entry["helper_log_path"] = str(helper_log_dir / f"{entry['model_id']}.log")
        entry["helper_status"] = "prepared"
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
            str(entry["prediction_path"]),
            "--summary-json",
            str(entry["summary_path"]),
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
        if entry.get("adapter_path"):
            command.extend(["--adapter-path", str(entry["adapter_path"])])
        commands.append(command)

    helper = {
        "status": "prepared" if args.dry_run else "running",
        "artifact_type": "non_authoritative_disjoint_evaluation_helper",
        "config_path": str(config_path),
        "config_hash": config_hash,
        "authoritative_manifest": str(authoritative_path),
        "task_interval": [args.task_start_index, args.task_end_index],
        "task_count": len(entries),
        "selected_model_ids": selected_ids,
        "evaluation_source_sha256": file_sha256(PROJECT_ROOT / "scripts/4_1_eval_model.py"),
        "helper_source_sha256": file_sha256(Path(__file__).resolve()),
        "runtime": runtime_metadata(),
        "runs": entries,
    }
    _write_json(helper_path, helper)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    max_parallel = args.max_parallel or len(gpu_ids)
    if not gpu_ids or max_parallel <= 0 or max_parallel > len(gpu_ids):
        raise ValueError("Invalid helper GPU IDs or --max-parallel.")
    failures = _run(entries, commands, gpu_ids, max_parallel, evaluation, helper, helper_path)
    helper["status"] = "failed" if failures else "complete"
    _write_json(helper_path, helper)
    if failures:
        raise SystemExit(f"Helper evaluation failures={len(failures)}")


def _run(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    evaluation: Mapping[str, Any],
    helper: Dict[str, Any],
    helper_path: Path,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    available = list(gpu_ids)
    failures: List[Dict[str, Any]] = []
    support: List[str] | None = None
    while pending or running:
        while pending and len(running) < max_parallel and available:
            entry, command = pending.pop(0)
            evidence = _completed(entry, evaluation)
            if evidence is not None:
                support = _check_support(support, evidence["problem_ids"], entry["model_id"])
                entry.update({key: value for key, value in evidence.items() if key != "problem_ids"})
                entry["helper_status"] = "skipped_complete"
                _write_json(helper_path, helper)
                continue
            existing = [
                str(path)
                for path in (Path(entry["prediction_path"]), Path(entry["summary_path"]))
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    f"Incomplete helper evaluation artifacts for {entry['model_id']}: {existing}"
                )
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            entry["gpu_id"] = gpu_id
            entry["helper_status"] = "running"
            log_handle = Path(entry["helper_log_path"]).open("x", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            entry["pid"] = process.pid
            running.append((process, entry, log_handle))
            _write_json(helper_path, helper)
        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            available.append(str(entry["gpu_id"]))
            evidence = _completed(entry, evaluation) if return_code == 0 else None
            if evidence is None:
                entry["helper_status"] = "failed"
                entry["returncode"] = return_code
                failures.append(entry)
            else:
                support = _check_support(support, evidence["problem_ids"], entry["model_id"])
                entry.update({key: value for key, value in evidence.items() if key != "problem_ids"})
                entry["helper_status"] = "complete"
            _write_json(helper_path, helper)
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _completed(entry: Mapping[str, Any], evaluation: Mapping[str, Any]) -> Dict[str, Any] | None:
    return completed_evaluation_evidence(
        entry["prediction_path"],
        entry["summary_path"],
        expected_n=int(evaluation["limit"]),
        expected_start_index=int(evaluation["start_index"]),
        expected_split=str(evaluation["dataset_split"]),
    )


def _check_support(expected: List[str] | None, observed: List[str], model_id: Any) -> List[str]:
    if expected is not None and expected != observed:
        raise ValueError(f"Helper evaluation support mismatch for {model_id}")
    return observed if expected is None else expected


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
