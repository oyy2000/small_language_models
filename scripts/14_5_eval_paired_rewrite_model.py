#!/usr/bin/env python3
"""Evaluate one paired-rewrite adapter under one decoding budget."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.factorial import file_sha256
from length_budget_distill.paired_rewrite_evaluation import (
    evaluate_problems,
    load_model_bundle,
    summarize_predictions,
)
from length_budget_distill.records import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--summary-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.start_index < 0 or args.limit <= 0:
        raise ValueError("Evaluation slice must be non-empty and non-negative")
    prediction_path = Path(args.prediction_path)
    summary_path = Path(args.summary_path)
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite paired-rewrite evaluation artifacts")
    config = load_config(args.config)
    config["dataset"] = dict(config["dataset"])
    config["dataset"]["split"] = args.split
    config["dataset"]["max_examples"] = args.start_index + args.limit
    problems = load_problem_records(config)[args.start_index : args.start_index + args.limit]
    if len(problems) != args.limit:
        raise ValueError(f"Expected {args.limit} evaluation rows, got {len(problems)}")
    adapter = str(_resolve(args.adapter_path)) if args.adapter_path else None
    if adapter:
        _require_snapshot(Path(adapter))
    bundle = load_model_bundle(config["student"]["model_name"], adapter)
    rows = evaluate_problems(
        problems,
        bundle,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    write_jsonl(prediction_path, rows)
    summary = summarize_predictions(
        rows,
        model_id=args.model_id,
        model_name=config["student"]["model_name"],
        adapter_path=adapter,
        split=args.split,
        start_index=args.start_index,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    summary["prediction_path"] = str(prediction_path)
    summary["prediction_sha256"] = file_sha256(prediction_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    logging.info("paired_eval_complete model=%s accuracy=%.4f", args.model_id, summary["greedy_accuracy"])


def _require_snapshot(path: Path) -> None:
    required = ["adapter_config.json", "adapter_model.safetensors", "SNAPSHOT_COMPLETE"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete adapter snapshot {path}; missing={missing}")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
