#!/usr/bin/env python3
"""Generate teacher traces under short, medium, and long length budgets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.backends import make_teacher_backend
from length_budget_distill.bucketing import get_length_budgets, iter_shard
from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.prompts import build_teacher_prompt, get_prompt_strategy
from length_budget_distill.records import TraceRecord, trace_to_dict, write_jsonl
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.tokenization import make_token_counter
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    parser.add_argument("--output-dir", required=True, help="Directory for raw traces and SFT records.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of independent data shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after dataset loading.")
    parser.add_argument("--log-every", type=int, default=10, help="Log shard progress every N generated traces.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = load_problem_records(config)
    if args.limit is not None:
        problems = problems[: args.limit]

    budgets = get_length_budgets(config)
    teacher = make_teacher_backend(config)
    token_counter = make_token_counter(config)
    prompt_strategy = get_prompt_strategy(config)

    logging.info("experiment=%s", config.get("experiment_name", "unnamed"))
    logging.info("prompt_strategy=%s", prompt_strategy)
    logging.info("teacher_backend=%s teacher_model=%s", teacher.backend_name, teacher.model_name)
    shard_problems = list(iter_shard(problems, args.num_shards, args.shard_index))
    total_traces = len(shard_problems) * len(budgets)

    logging.info("num_problems=%d num_budgets=%d", len(problems), len(budgets))
    logging.info("num_shard_problems=%d total_shard_traces=%d", len(shard_problems), total_traces)
    logging.info("shard_index=%d num_shards=%d", args.shard_index, args.num_shards)

    traces: List[TraceRecord] = []
    start_time = time.monotonic()
    completed = 0
    correct = 0
    log_every = max(1, int(args.log_every))

    for _, problem in shard_problems:
        for budget in budgets:
            prompt = build_teacher_prompt(problem, budget, config)
            solution = teacher.generate(problem, budget, prompt)
            predicted = extract_final_answer(solution)
            is_correct = verify_answer(predicted, problem.answer)
            correct += int(is_correct)
            token_count = token_counter.count(solution)
            trace_id = f"{problem.problem_id}:{budget['name']}"
            traces.append(
                TraceRecord(
                    trace_id=trace_id,
                    problem_id=problem.problem_id,
                    question=problem.question,
                    answer=problem.answer,
                    budget_name=budget["name"],
                    max_solution_tokens=int(budget["max_solution_tokens"]),
                    teacher_backend=teacher.backend_name,
                    teacher_model=teacher.model_name,
                    prompt=prompt if config.get("output", {}).get("include_prompt_in_trace", True) else "",
                    solution=solution,
                    predicted_answer=predicted,
                    is_correct=is_correct,
                    solution_token_count=token_count,
                    metadata={
                        "problem_metadata": problem.metadata,
                        "prompt_strategy": prompt_strategy,
                    },
                )
            )
            completed += 1
            if completed == 1 or completed % log_every == 0 or completed == total_traces:
                elapsed = time.monotonic() - start_time
                traces_per_sec = completed / elapsed if elapsed > 0 else 0.0
                remaining = total_traces - completed
                eta_sec = remaining / traces_per_sec if traces_per_sec > 0 else 0.0
                percent = (completed / total_traces * 100.0) if total_traces else 100.0
                logging.info(
                    "progress shard=%d percent=%.1f%% processed=%d/%d correct=%d elapsed_min=%.1f eta_min=%.1f traces_per_sec=%.3f",
                    args.shard_index,
                    percent,
                    completed,
                    total_traces,
                    correct,
                    elapsed / 60.0,
                    eta_sec / 60.0,
                    traces_per_sec,
                )

    shard_suffix = f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}"
    raw_path = output_dir / f"{shard_suffix}.jsonl"
    sft_path = output_dir / f"sft_{shard_suffix}.jsonl"
    config_snapshot_path = output_dir / "config_snapshot.json"

    raw_count = write_jsonl(raw_path, (trace_to_dict(trace) for trace in traces))
    keep_only_correct = bool(config.get("output", {}).get("keep_only_correct_for_sft", True))
    sft_records = [
        trace_to_sft_record(trace)
        for trace in traces
        if trace.is_correct or not keep_only_correct
    ]
    sft_count = write_jsonl(sft_path, sft_records)

    with config_snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    logging.info("wrote_raw=%d path=%s", raw_count, raw_path)
    logging.info("wrote_sft=%d path=%s", sft_count, sft_path)


if __name__ == "__main__":
    main()
