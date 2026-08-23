#!/usr/bin/env python3
"""Build equal-example, equal-token, and calibration SFT datasets."""

from __future__ import annotations

import argparse
import json
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
from length_budget_distill.factorial import (
    canonical_sha256,
    common_problem_ids,
    deterministic_equal_token_subset,
    expected_conditions,
    file_sha256,
    selected_by_condition,
)
from length_budget_distill.records import read_jsonl, trace_from_dict, write_jsonl
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.student_prompts import build_student_math_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional seed-subset overlay bound to the immutable parent protocol hash.",
    )
    parser.add_argument("--selected-traces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config_hash = canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})
    configured_seeds = [int(seed) for seed in config.get("balancing", {}).get("training_seeds", [17, 42, 73])]
    seeds, run_config_evidence, protocol_variant = _resolve_training_seeds(
        args.run_config,
        config_hash=config_hash,
        configured_seeds=configured_seeds,
    )
    selected = [trace_from_dict(row) for row in read_jsonl(Path(args.selected_traces))]
    selected_traces_sha256 = file_sha256(args.selected_traces)
    conditions = expected_conditions(config)
    common_ids = common_problem_ids(selected, conditions)
    minimum_common = int(config["balancing"][f"{args.stage}_min_common_problems"])
    if len(common_ids) < minimum_common:
        raise ValueError(
            f"Common-problem gate failed before data construction: actual={len(common_ids)} required={minimum_common}"
        )
    grouped = selected_by_condition(selected, common_ids)
    missing = [condition for condition in conditions if len(grouped.get(condition, [])) != len(common_ids)]
    if missing:
        raise ValueError(f"Selected trace matrix is incomplete for conditions={missing}")

    output_dir = Path(args.output_dir)
    if (output_dir / "DATASETS_COMPLETE").exists():
        raise FileExistsError(f"Dataset output is already complete: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_entries: List[Dict[str, Any]] = []

    totals = {condition: sum(trace.solution_token_count for trace in grouped[condition]) for condition in conditions}
    equal_token_target = min(totals.values())
    max_gap = int(config.get("balancing", {}).get("max_equal_token_gap", 512))

    for generator_name, budget_name in conditions:
        condition = (generator_name, budget_name)
        condition_slug = f"{generator_name}__{budget_name}"
        equal_example_path = output_dir / "equal_example" / f"{condition_slug}.jsonl"
        equal_example_records = [trace_to_sft_record(trace) for trace in grouped[condition]]
        write_jsonl(equal_example_path, equal_example_records)
        for seed in seeds:
            run_entries.append(
                _run_entry(
                    mode="equal_example",
                    generator_name=generator_name,
                    budget_name=budget_name,
                    seed=seed,
                    path=equal_example_path,
                    n=len(grouped[condition]),
                    supervised_tokens=totals[condition],
                    train_sha256=file_sha256(equal_example_path),
                )
            )

        for seed in seeds:
            subset_seed = int(canonical_sha256([config_hash, condition_slug, seed])[:8], 16)
            subset, token_total = deterministic_equal_token_subset(
                grouped[condition],
                target_tokens=equal_token_target,
                seed=subset_seed,
            )
            gap = equal_token_target - token_total
            if gap < 0 or gap > max_gap:
                raise ValueError(
                    f"Equal-token subset is outside tolerance for {condition_slug}, seed={seed}: "
                    f"target={equal_token_target} actual={token_total} gap={gap} tolerance={max_gap}"
                )
            equal_token_path = output_dir / "equal_token" / f"{condition_slug}__seed_{seed}.jsonl"
            write_jsonl(equal_token_path, (trace_to_sft_record(trace) for trace in subset))
            run_entries.append(
                _run_entry(
                    mode="equal_token",
                    generator_name=generator_name,
                    budget_name=budget_name,
                    seed=seed,
                    path=equal_token_path,
                    n=len(subset),
                    supervised_tokens=token_total,
                    token_target=equal_token_target,
                    token_gap=gap,
                    train_sha256=file_sha256(equal_token_path),
                )
            )

    problems = {problem.problem_id: problem for problem in load_problem_records(config)}
    missing_problem_ids = [problem_id for problem_id in common_ids if problem_id not in problems]
    if missing_problem_ids:
        raise ValueError(f"Could not reload common source problems: {missing_problem_ids[:10]}")
    gold_records = []
    answer_only_records = []
    for problem_id in common_ids:
        problem = problems[problem_id]
        raw_answer = str(problem.metadata.get("raw_answer", problem.answer))
        if "####" in raw_answer:
            rationale = raw_answer.rsplit("####", 1)[0].strip()
            gold_completion = f"{rationale}\nAnswer: {problem.answer}" if rationale else f"Answer: {problem.answer}"
        else:
            gold_completion = f"Answer: {problem.answer}"
        student_prompt = build_student_math_prompt(problem.question)
        base_metadata = {"problem_id": problem_id, "source": "gsm8k_gold", "config_hash": config_hash}
        gold_records.append(
            {
                "id": f"{problem_id}:gold_rationale",
                "prompt": student_prompt,
                "completion": gold_completion,
                "messages": [
                    {"role": "user", "content": student_prompt},
                    {"role": "assistant", "content": gold_completion},
                ],
                "metadata": base_metadata,
            }
        )
        answer_only_records.append(
            {
                "id": f"{problem_id}:answer_only",
                "prompt": student_prompt,
                "completion": f"Answer: {problem.answer}",
                "messages": [
                    {"role": "user", "content": student_prompt},
                    {"role": "assistant", "content": f"Answer: {problem.answer}"},
                ],
                "metadata": {**base_metadata, "source": "gsm8k_answer_only"},
            }
        )
    calibration = {
        "gold_rationale": (output_dir / "calibration" / "gold_rationale.jsonl", gold_records),
        "answer_only": (output_dir / "calibration" / "answer_only.jsonl", answer_only_records),
    }
    for baseline_name, (path, records) in calibration.items():
        write_jsonl(path, records)
        for seed in seeds:
            run_entries.append(
                {
                    "run_name": f"{baseline_name}__seed_{seed}",
                    "mode": "calibration",
                    "baseline_name": baseline_name,
                    "generator_name": None,
                    "budget_name": None,
                    "seed": seed,
                    "train_path": str(path),
                    "train_sha256": file_sha256(path),
                    "n": len(records),
                    "supervised_tokens": None,
                }
            )

    expected_runs = len(conditions) * len(seeds) * 2 + len(calibration) * len(seeds)
    if len(run_entries) != expected_runs:
        raise RuntimeError(f"Run-manifest size mismatch: expected={expected_runs} actual={len(run_entries)}")
    manifest = {
        "status": "complete",
        "stage": args.stage,
        "config_hash": config_hash,
        "selected_traces_path": args.selected_traces,
        "selected_traces_sha256": selected_traces_sha256,
        "common_problem_count": len(common_ids),
        "common_problem_ids": common_ids,
        "equal_token_target": equal_token_target,
        "configured_training_seeds": configured_seeds,
        "training_seeds": seeds,
        "protocol_variant": protocol_variant,
        "run_config": run_config_evidence,
        "expected_run_count": expected_runs,
        "runs": run_entries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    (output_dir / "DATASETS_COMPLETE").write_text(
        f"config_hash={config_hash}\nrun_count={expected_runs}\n"
        f"common_problem_count={len(common_ids)}\nmanifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )
    logging.info(
        "datasets_complete common=%d runs=%d seeds=%s variant=%s equal_token_target=%d output=%s",
        len(common_ids),
        expected_runs,
        seeds,
        protocol_variant,
        equal_token_target,
        output_dir,
    )


def _run_entry(
    mode: str,
    generator_name: str,
    budget_name: str,
    seed: int,
    path: Path,
    n: int,
    supervised_tokens: int,
    train_sha256: str,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "run_name": f"{mode}__{generator_name}__{budget_name}__seed_{seed}",
        "mode": mode,
        "generator_name": generator_name,
        "budget_name": budget_name,
        "seed": seed,
        "train_path": str(path),
        "train_sha256": train_sha256,
        "n": n,
        "supervised_tokens": supervised_tokens,
        **extra,
    }


def _resolve_training_seeds(
    run_config_path: str | None,
    *,
    config_hash: str,
    configured_seeds: List[int],
) -> tuple[List[int], Dict[str, Any] | None, str]:
    if not configured_seeds or len(configured_seeds) != len(set(configured_seeds)):
        raise ValueError("balancing.training_seeds must contain unique seeds.")
    if run_config_path is None:
        return configured_seeds, None, "registered_parent_protocol"

    path = Path(run_config_path)
    with path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    if not isinstance(run_config, dict):
        raise ValueError(f"Run config must be a JSON object: {path}")
    if run_config.get("parent_config_sha256") != config_hash:
        raise ValueError(
            "Run-config parent hash mismatch: "
            f"overlay={run_config.get('parent_config_sha256')} config={config_hash}"
        )
    seeds = [int(seed) for seed in run_config.get("training_seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Run config must define a non-empty list of unique training_seeds.")
    unknown = sorted(set(seeds) - set(configured_seeds))
    if unknown:
        raise ValueError(f"Run config requests seeds outside the parent protocol: {unknown}")
    protocol_variant = str(run_config.get("protocol_variant", "training_seed_subset"))
    evidence = {
        "path": str(path),
        "sha256": file_sha256(path),
        "parent_config_sha256": config_hash,
    }
    return seeds, evidence, protocol_variant


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
