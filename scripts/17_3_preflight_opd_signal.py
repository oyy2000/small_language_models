#!/usr/bin/env python3
"""Generate one shard of the frozen dual-arm OPD signal preflight."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.factorial import canonical_sha256, file_sha256, runtime_metadata
from length_budget_distill.logit_kd import load_teacher_and_student
from length_budget_distill.opd import (
    OPD_ARMS,
    build_bounded_concise_prompt,
    collect_scored_rollouts,
    generate_greedy_completion_ids,
    preflight_summary,
    protocol_hash,
    read_json,
    reference_length_bounds,
    validate_opd_protocol,
    write_gzip_jsonl,
    write_json,
)
from length_budget_distill.student_prompts import build_student_math_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    configured_shards = int(protocol["preflight"]["num_shards"])
    num_shards = args.num_shards or configured_shards
    if num_shards != configured_shards or not 0 <= args.shard_index < num_shards:
        raise ValueError("Preflight shard topology does not match the registered protocol.")
    output_dir = _resolve(args.output_dir)
    suffix = f"shard_{args.shard_index:05d}_of_{num_shards:05d}"
    rows_path = output_dir / "shards" / f"{suffix}.jsonl.gz"
    manifest_path = output_dir / "manifests" / f"{suffix}.json"
    expected_prompt_count = sum(
        offset % num_shards == args.shard_index
        for offset in range(int(protocol["preflight"]["prompt_count"]))
    )
    expected_rollout_count = (
        expected_prompt_count * len(OPD_ARMS) * int(protocol["training"]["rollouts_per_prompt"])
    )
    if rows_path.is_file() and manifest_path.is_file() and args.skip_complete:
        manifest = read_json(manifest_path)
        expected = {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "shard_index": args.shard_index,
            "num_shards": num_shards,
            "prompt_count": expected_prompt_count,
            "rollout_count": expected_rollout_count,
            "rollout_path": str(rows_path),
            "rollout_sha256": file_sha256(rows_path),
            "source_sha256": file_sha256(Path(__file__).resolve()),
        }
        if all(manifest.get(key) == value for key, value in expected.items()):
            logging.info("opd_preflight_shard_already_complete shard=%d", args.shard_index)
            return
        raise ValueError(f"Existing OPD preflight shard failed validation: {suffix}")
    if rows_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite OPD preflight shard: {suffix}")

    split = protocol["splits"]["calibration"]
    dataset_config = dict(protocol)
    dataset_config["dataset"] = dict(protocol["dataset"])
    dataset_config["dataset"]["split"] = split["dataset_split"]
    dataset_config["dataset"]["max_examples"] = int(split["start_index"]) + int(split["limit"])
    problems = load_problem_records(dataset_config)
    start = int(split["start_index"])
    count = int(protocol["preflight"]["prompt_count"])
    problems = problems[start : start + count]
    if len(problems) != count:
        raise ValueError(f"Preflight problem count mismatch: expected={count} actual={len(problems)}")
    assigned = [
        (offset, problem)
        for offset, problem in enumerate(problems)
        if offset % num_shards == args.shard_index
    ]
    if len(assigned) != expected_prompt_count:
        raise ValueError("Preflight shard prompt count mismatch.")

    teacher, student, tokenizer, valid_vocab_size, model_evidence = load_teacher_and_student(
        protocol,
        train_student=True,
    )
    references: List[Dict[str, Any]] = []
    for offset, problem in assigned:
        standard_prompt = build_student_math_prompt(problem.question)
        generated = generate_greedy_completion_ids(
            student,
            tokenizer,
            standard_prompt,
            max_new_tokens=int(protocol["reference_generation"]["max_new_tokens"]),
            valid_vocab_size=valid_vocab_size,
        )
        completion = list(generated["completion_token_ids"])
        if tokenizer.eos_token_id is not None and completion[-1] == tokenizer.eos_token_id:
            completion = completion[:-1]
        reference_tokens = len(completion)
        if reference_tokens <= 0:
            raise RuntimeError(
                f"Frozen student produced an empty preflight reference: {problem.problem_id}"
            )
        lower, upper = reference_length_bounds(
            reference_tokens,
            lower_ratio=float(protocol["concise_prompt"]["lower_ratio"]),
            upper_ratio=float(protocol["concise_prompt"]["upper_ratio"]),
            minimum_tokens=int(protocol["concise_prompt"]["minimum_tokens"]),
            maximum_tokens=int(protocol["concise_prompt"]["maximum_tokens"]),
        )
        references.append(
            {
                "problem_id": problem.problem_id,
                "source_index": start + offset,
                "question": problem.question,
                "gold_answer": problem.answer,
                "standard_prompt": standard_prompt,
                "reference_output_tokens": reference_tokens,
                "concise_lower_tokens": lower,
                "concise_upper_tokens": upper,
                "concise_prompt": build_bounded_concise_prompt(problem.question, lower, upper),
            }
        )
        if len(references) % 10 == 0:
            logging.info(
                "preflight_reference_progress shard=%d records=%d/%d",
                args.shard_index,
                len(references),
                len(assigned),
            )

    rows: List[Dict[str, Any]] = []
    for arm in OPD_ARMS:
        arm_rows = collect_scored_rollouts(
            protocol,
            arm=arm,
            references=references,
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            diagnostic_top_k=int(protocol["diagnostics"]["top_k"]),
            diagnostic_rollouts=(
                int(protocol["diagnostics"]["top_k_rollouts"])
                if args.shard_index == 0
                else 0
            ),
        )
        rows.extend(arm_rows)
        logging.info("preflight_arm_complete arm=%s rollouts=%d", arm, len(arm_rows))
    summary = preflight_summary(rows)
    write_gzip_jsonl(rows_path, rows)
    manifest = {
        "status": "complete",
        "stage": "preflight_shard",
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "shard_index": args.shard_index,
        "num_shards": num_shards,
        "prompt_count": len(references),
        "problem_ids_sha256": canonical_sha256([row["problem_id"] for row in references]),
        "rollout_path": str(rows_path),
        "rollout_sha256": file_sha256(rows_path),
        "rollout_count": len(rows),
        "local_summary": summary,
        "model_evidence": model_evidence,
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "runtime": runtime_metadata(),
    }
    write_json(manifest_path, manifest)
    logging.info(
        "opd_preflight_shard_complete shard=%d prompts=%d rollouts=%d",
        args.shard_index,
        len(references),
        len(rows),
    )


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
