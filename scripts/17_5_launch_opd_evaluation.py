#!/usr/bin/env python3
"""Launch base and OPD-adapter evaluation on identical standard-prompt support."""

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
from length_budget_distill.factorial import file_sha256, runtime_metadata
from length_budget_distill.opd import (
    OPD_ARMS,
    protocol_hash,
    validate_opd_protocol,
    validated_opd_adapter,
)
from length_budget_distill.opd_analysis import completed_opd_evaluation


MODEL_IDS = ("base",) + OPD_ARMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument(
        "--split-name",
        choices=["primary_evaluation", "secondary_evaluation"],
        required=True,
    )
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    split = protocol["splits"][args.split_name]
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU ID.")
    max_parallel = args.max_parallel or min(len(gpu_ids), len(MODEL_IDS))
    if max_parallel <= 0 or max_parallel > len(gpu_ids):
        raise ValueError("--max-parallel must be positive and no larger than the GPU count.")

    output_dir = _resolve(args.output_dir)
    prediction_dir = output_dir / "predictions"
    summary_dir = output_dir / "summaries"
    log_dir = output_dir / "logs"
    for path in (prediction_dir, summary_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / (
        "evaluation_launcher_manifest_dry_run.json"
        if args.dry_run
        else "evaluation_launcher_manifest.json"
    )
    if manifest_path.exists() and not args.skip_complete:
        raise FileExistsError(f"Refusing to overwrite evaluation manifest: {manifest_path}")

    checkpoint_root = _resolve(protocol["outputs"]["checkpoint_root"])
    tasks: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for model_id in MODEL_IDS:
        adapter_path = None if model_id == "base" else checkpoint_root / "pilot" / model_id
        adapter_evidence = None
        if adapter_path is not None and not args.dry_run:
            adapter_evidence = validated_opd_adapter(
                protocol,
                arm=model_id,
                adapter_dir=adapter_path,
                stage="pilot",
            )
            if adapter_evidence is None:
                raise ValueError(f"Invalid or incomplete OPD adapter: {adapter_path}")
        entry = {
            "model_id": model_id,
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
            "adapter_train_manifest_sha256": (
                adapter_evidence.get("train_manifest_sha256") if adapter_evidence else None
            ),
            "prediction_path": str(prediction_dir / f"{model_id}.jsonl"),
            "summary_path": str(summary_dir / f"{model_id}.json"),
            "log_path": str(log_dir / f"{model_id}.log"),
            "eval_status": "prepared",
        }
        command = [
            sys.executable,
            "scripts/17_5_eval_opd_model.py",
            "--config",
            str(config_path),
            "--split-name",
            args.split_name,
            "--model-id",
            model_id,
            "--output-jsonl",
            entry["prediction_path"],
            "--summary-json",
            entry["summary_path"],
        ]
        if adapter_path is not None:
            command.extend(["--adapter-path", str(adapter_path)])
        tasks.append(entry)
        commands.append(command)

    manifest: Dict[str, Any] = {
        "status": "prepared" if args.dry_run else "running",
        "stage": "pilot",
        "split_name": args.split_name,
        "dataset_split": split["dataset_split"],
        "start_index": int(split["start_index"]),
        "limit": int(split["limit"]),
        "prompt_mode": "common_standard_prompt",
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "evaluation_source_sha256": file_sha256(
            PROJECT_ROOT / "scripts/17_5_eval_opd_model.py"
        ),
        "launcher_source_sha256": file_sha256(Path(__file__).resolve()),
        "analysis_library_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/opd_analysis.py"
        ),
        "run_count": len(tasks),
        "runtime": runtime_metadata(),
        "runs": tasks,
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    failures = _run_tasks(
        tasks,
        commands,
        gpu_ids,
        max_parallel,
        protocol,
        args.split_name,
        manifest,
        manifest_path,
        args.skip_complete,
    )
    manifest["status"] = "failed" if failures else "complete"
    _write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"OPD evaluation failures={len(failures)}; manifest={manifest_path}")
    logging.info("opd_evaluation_launcher_complete split=%s manifest=%s", args.split_name, manifest_path)


def _run_tasks(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    protocol: Dict[str, Any],
    split_name: str,
    manifest: Dict[str, Any],
    manifest_path: Path,
    skip_complete: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    available = list(gpu_ids)
    failures: List[Dict[str, Any]] = []
    support: List[str] | None = None
    while pending or running:
        while pending and len(running) < max_parallel and available:
            entry, command = pending.pop(0)
            evidence = _completed(entry, protocol, split_name)
            if evidence is not None:
                if not skip_complete:
                    raise FileExistsError(f"Complete evaluation exists: {entry['model_id']}")
                support = _check_support(support, evidence.pop("problem_ids"), entry["model_id"])
                entry.update(evidence)
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
            gpu_id = available.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_handle = Path(entry["log_path"]).open("x", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            entry.update({"eval_status": "running", "gpu_id": gpu_id, "pid": process.pid})
            running.append((process, entry, log_handle))
            _write_json(manifest_path, manifest)
            logging.info("launched_opd_eval model=%s gpu=%s pid=%d", entry["model_id"], gpu_id, process.pid)

        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            available.append(str(entry["gpu_id"]))
            evidence = _completed(entry, protocol, split_name) if return_code == 0 else None
            if evidence is None:
                entry.update({"eval_status": "failed", "returncode": return_code})
                failures.append(dict(entry))
            else:
                support = _check_support(support, evidence.pop("problem_ids"), entry["model_id"])
                entry.update(evidence)
                entry.update({"eval_status": "complete", "returncode": return_code})
            _write_json(manifest_path, manifest)
            logging.info("finished_opd_eval model=%s status=%s", entry["model_id"], entry["eval_status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _completed(
    entry: Dict[str, Any],
    protocol: Dict[str, Any],
    split_name: str,
) -> Dict[str, Any] | None:
    return completed_opd_evaluation(
        protocol,
        split_name=split_name,
        model_id=str(entry["model_id"]),
        prediction_path=entry["prediction_path"],
        summary_path=entry["summary_path"],
    )


def _check_support(expected: List[str] | None, observed: List[str], model_id: Any) -> List[str]:
    if expected is not None and observed != expected:
        raise ValueError(f"Evaluation problem support mismatch for {model_id}")
    return observed if expected is None else expected


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
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
