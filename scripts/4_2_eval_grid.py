#!/usr/bin/env python3
"""Evaluate every checkpoint listed in a student SFT grid manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="results/student_sft_grid/manifest.json",
        help="Manifest produced by 2_2_grid_search_student_sft.py.",
    )
    parser.add_argument(
        "--config",
        default="configs/real_length_budget_template.json",
        help="Dataset/eval config.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name used with each LoRA adapter.",
    )
    parser.add_argument("--split", default="test", help="Eval split.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based first eval example.")
    parser.add_argument("--limit", type=int, default=50, help="Number of eval examples per run.")
    parser.add_argument(
        "--output-dir",
        default="results/student_sft_grid/eval",
        help="Directory for per-run predictions, summaries, and aggregate reports.",
    )
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU IDs, e.g. 0,1,2,3.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum concurrent eval jobs. Defaults to number of GPU IDs, or 1 if no GPU IDs are provided.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation token limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation top-p.")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Model dtype passed to 4_1_eval_model.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print eval commands without launching.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with an existing summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    entries = _read_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    summary_dir = output_dir / "summaries"
    log_dir = output_dir / "logs"
    for path in (pred_dir, summary_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    eval_entries = []
    commands = []
    for entry in entries:
        run_name = entry["run_name"]
        eval_entry = {
            "run_name": run_name,
            "adapter_path": entry["output_dir"],
            "summary_json": str(summary_dir / f"{run_name}.json"),
            "output_jsonl": str(pred_dir / f"{run_name}.jsonl"),
            "log_path": str(log_dir / f"{run_name}.log"),
            "overrides": entry.get("overrides", {}),
        }
        eval_entries.append(eval_entry)
        commands.append(_eval_command(args, eval_entry))

    if args.dry_run:
        for entry, command in zip(eval_entries, commands):
            print(_format_command(command, entry.get("gpu_id")))
        return

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    failures = _run_commands(eval_entries, commands, gpu_ids, max_parallel, args.skip_existing)
    if failures:
        logging.error("failed_eval_runs=%d", len(failures))
        for entry in failures:
            logging.error("failed run=%s log=%s", entry["run_name"], entry["log_path"])
        raise SystemExit(1)

    aggregate = _collect_summaries(eval_entries)
    aggregate_json = output_dir / "grid_eval_summary.json"
    aggregate_csv = output_dir / "grid_eval_summary.csv"
    _write_json(aggregate_json, {"runs": aggregate})
    _write_csv(aggregate_csv, aggregate)
    logging.info("wrote_aggregate_json=%s", aggregate_json)
    logging.info("wrote_aggregate_csv=%s", aggregate_csv)


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"Manifest has no runs: {path}")
    return runs


def _eval_command(args: argparse.Namespace, entry: Dict[str, Any]) -> List[str]:
    return [
        sys.executable,
        "scripts/4_1_eval_model.py",
        "--config",
        args.config,
        "--model-name",
        args.model_name,
        "--adapter-path",
        entry["adapter_path"],
        "--split",
        args.split,
        "--start-index",
        str(args.start_index),
        "--limit",
        str(args.limit),
        "--output-jsonl",
        entry["output_jsonl"],
        "--summary-json",
        entry["summary_json"],
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--torch-dtype",
        args.torch_dtype,
    ]


def _parse_gpu_ids(raw_gpu_ids: str | None) -> List[str]:
    if not raw_gpu_ids:
        return []
    return [item.strip() for item in raw_gpu_ids.split(",") if item.strip()]


def _run_commands(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    skip_existing: bool,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)

    while pending or running:
        can_launch = lambda: pending and len(running) < max_parallel and (not gpu_ids or available_gpus)
        while can_launch():
            entry, command = pending.pop(0)
            if skip_existing and Path(entry["summary_json"]).exists():
                logging.info("skip_existing run=%s summary=%s", entry["run_name"], entry["summary_json"])
                continue
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                entry["gpu_id"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            logging.info("launch eval run=%s gpu=%s log=%s", entry["run_name"], gpu_id or "default", entry["log_path"])
            logging.info("command=%s", _format_command(command, gpu_id))
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((process, entry, log_handle))

        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            code = process.poll()
            if code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            gpu_id = entry.get("gpu_id")
            if gpu_id is not None:
                available_gpus.append(gpu_id)
            if code == 0:
                logging.info("finished eval run=%s", entry["run_name"])
            else:
                entry["returncode"] = code
                failures.append(entry)
                logging.error("failed eval run=%s returncode=%s", entry["run_name"], code)
        running = still_running
        if running:
            time.sleep(5)

    return failures


def _collect_summaries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for entry in entries:
        summary_path = Path(entry["summary_json"])
        if not summary_path.exists():
            logging.warning("missing_summary run=%s path=%s", entry["run_name"], summary_path)
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row = {
            "run_name": entry["run_name"],
            "accuracy": summary.get("accuracy"),
            "correct": summary.get("correct"),
            "n": summary.get("n"),
            "adapter_path": entry["adapter_path"],
            "summary_json": entry["summary_json"],
            "output_jsonl": entry["output_jsonl"],
        }
        row.update({f"override.{key}": value for key, value in entry.get("overrides", {}).items()})
        rows.append(row)
    return sorted(rows, key=lambda item: (item.get("accuracy") is None, -(item.get("accuracy") or 0.0)))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_command(command: Iterable[str], gpu_id: str | None = None) -> str:
    rendered = " ".join(command)
    if gpu_id is not None:
        return f"CUDA_VISIBLE_DEVICES={gpu_id} {rendered}"
    return rendered


if __name__ == "__main__":
    main()
