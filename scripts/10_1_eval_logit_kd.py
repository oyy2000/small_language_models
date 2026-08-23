#!/usr/bin/env python3
"""Run and register one base, SFT, or logit-KD GSM8K evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import validated_adapter_evidence
from length_budget_distill.logit_kd import (
    copy_with_retries,
    file_sha256,
    kd_run_name,
    load_protocol,
    protocol_hash,
    read_jsonl,
    resolve_project_path,
    runtime_metadata,
    validated_training_marker,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--stage", choices=["validation", "formal"], required=True)
    parser.add_argument("--method", choices=["base", "sft", "kd"], required=True)
    parser.add_argument("--budget", choices=["short_128", "medium_256", "long_512"], default="short_128")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def _adapter_and_name(protocol: Dict[str, Any], args: argparse.Namespace) -> tuple[Path | None, str]:
    if args.method == "base":
        return None, "base_qwen2p5_1p5b_instruct"
    if args.method == "sft":
        path = resolve_project_path(protocol["budgets"][args.budget]["baseline_adapter"])
        if validated_adapter_evidence(path) is None:
            raise ValueError(f"Parent SFT adapter is incomplete: {path}")
        return path, f"sft__{args.budget}__seed_17"
    if args.alpha is None or args.temperature is None:
        raise ValueError("KD evaluation requires --alpha and --temperature.")
    run_name = kd_run_name(args.budget, args.alpha, args.temperature)
    checkpoint_root = resolve_project_path(protocol["outputs"]["checkpoint_root"])
    if args.stage == "formal":
        path = checkpoint_root / "formal" / f"{args.budget}__seed_17"
    else:
        path = checkpoint_root / args.stage / run_name
    marker = validated_training_marker(path)
    if marker is None:
        raise ValueError(f"KD adapter is incomplete: {path}")
    if float(marker["alpha"]) != args.alpha or float(marker["temperature"]) != args.temperature:
        raise ValueError(f"KD adapter parameter mismatch: {path}")
    return path, f"kd__{run_name}"


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load_protocol(args.config)
    adapter_path, run_name = _adapter_and_name(protocol, args)
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    eval_root = result_root / args.stage / "eval"
    prediction_path = eval_root / "predictions" / f"{run_name}.jsonl"
    summary_path = eval_root / "summaries" / f"{run_name}.json"
    log_path = eval_root / "logs" / f"{run_name}.log"
    marker_path = eval_root / "markers" / f"{run_name}.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        valid = (
            prediction_path.is_file()
            and summary_path.is_file()
            and marker.get("prediction_sha256") == file_sha256(prediction_path)
            and marker.get("summary_sha256") == file_sha256(summary_path)
        )
        if valid and args.skip_complete:
            logging.info("evaluation_already_complete run=%s", run_name)
            return
        if valid:
            raise FileExistsError(f"Evaluation is already complete: {marker_path}")
        raise ValueError(f"Existing evaluation marker is invalid: {marker_path}")
    existing = [path for path in (prediction_path, summary_path, log_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Incomplete evaluation artifacts exist: {existing}")

    stage_config = protocol[args.stage]
    runtime_root = Path(os.environ.get("LBD_RUNTIME_EVAL_ROOT", tempfile.gettempdir()))
    runtime_dir = runtime_root / protocol["experiment_name"] / args.stage / run_name
    if runtime_dir.exists():
        raise FileExistsError(f"Runtime evaluation directory exists: {runtime_dir}")
    runtime_dir.mkdir(parents=True)
    runtime_prediction = runtime_dir / "predictions.jsonl"
    runtime_summary = runtime_dir / "summary.json"
    runtime_log = runtime_dir / "eval.log"
    command = [
        sys.executable,
        "scripts/4_1_eval_model.py",
        "--config",
        protocol["parent"]["config_path"],
        "--model-name",
        protocol["models"]["student"]["model_name"],
        "--split",
        stage_config["dataset_split"],
        "--start-index",
        str(stage_config["start_index"]),
        "--limit",
        str(stage_config["limit"]),
        "--output-jsonl",
        str(runtime_prediction),
        "--summary-json",
        str(runtime_summary),
        "--max-new-tokens",
        str(stage_config["max_new_tokens"]),
        "--temperature",
        str(stage_config["temperature"]),
        "--top-p",
        str(stage_config["top_p"]),
        "--batch-size",
        str(stage_config["batch_size"]),
        "--torch-dtype",
        "bfloat16",
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    with runtime_log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise SystemExit(f"Evaluation failed returncode={completed.returncode} log={runtime_log}")
    rows = read_jsonl(runtime_prediction)
    if len(rows) != int(stage_config["limit"]):
        raise ValueError(f"Evaluation prediction count mismatch: expected={stage_config['limit']} actual={len(rows)}")
    problem_ids = [str(row.get("problem_id")) for row in rows]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError(f"Evaluation contains duplicate problem IDs: {run_name}")
    summary = json.loads(runtime_summary.read_text(encoding="utf-8"))
    if int(summary.get("n", -1)) != int(stage_config["limit"]):
        raise ValueError(f"Evaluation summary count mismatch: {run_name}")
    for source, destination in (
        (runtime_prediction, prediction_path),
        (runtime_summary, summary_path),
        (runtime_log, log_path),
    ):
        copy_with_retries(source, destination)
    marker = {
        "status": "complete",
        "stage": args.stage,
        "method": args.method,
        "budget_name": args.budget if args.method != "base" else None,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "run_name": run_name,
        "protocol_hash": protocol_hash(protocol),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_sha256(prediction_path),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "log_path": str(log_path),
        "log_sha256": file_sha256(log_path),
        "source_sha256": {
            "scripts/4_1_eval_model.py": file_sha256(PROJECT_ROOT / "scripts/4_1_eval_model.py"),
            "scripts/10_1_eval_logit_kd.py": file_sha256(Path(__file__).resolve()),
        },
        "records": len(rows),
        "unique_problem_ids": len(set(problem_ids)),
        "runtime": runtime_metadata(),
    }
    write_json(marker_path, marker)
    logging.info("evaluation_complete run=%s prediction=%s", run_name, prediction_path)


if __name__ == "__main__":
    main()
