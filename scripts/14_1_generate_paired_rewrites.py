#!/usr/bin/env python3
"""Generate sharded paired rewrites of the verified 7B standard rationales."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.backends import GenerationRequest, make_teacher_backend
from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256, runtime_metadata
from length_budget_distill.paired_rewrite import (
    assess_rewrite_candidate,
    build_rewrite_prompt,
    essential_step_values,
    minimum_target_token_count,
    source_problem_id,
    source_token_count,
    stable_rewrite_seed,
    target_token_count,
)
from length_budget_distill.records import ProblemRecord, read_jsonl, write_jsonl
from length_budget_distill.tokenization import make_token_counter
from length_budget_distill.verifiers import extract_final_answer, verify_answer


Task = Tuple[Mapping[str, Any], str, float, int, int, List[str], GenerationRequest]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_paired_rewrite_7b_pilot_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard topology")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    config = load_config(args.config)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    paired = dict(config["paired_rewrite"])
    source_path = _resolve_project_path(str(paired["source_standard_path"]))
    expected_source_hash = str(paired["source_standard_sha256"])
    actual_source_hash = file_sha256(source_path)
    if actual_source_hash != expected_source_hash:
        raise ValueError(
            f"Source standard hash mismatch: expected={expected_source_hash} actual={actual_source_hash}"
        )
    rows = list(read_jsonl(source_path))
    expected_records = int(paired["expected_records"])
    if len(rows) != expected_records:
        raise ValueError(f"Expected {expected_records} source rows, got {len(rows)}")
    if args.limit is not None:
        rows = rows[: args.limit]
    assigned = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]

    output_dir = Path(args.output_dir)
    suffix = f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    raw_path = output_dir / f"{suffix}.jsonl"
    manifest_path = output_dir / "manifests" / f"{suffix}.json"
    generation_source_sha256 = file_sha256(Path(__file__).resolve())
    rewrite_library_sha256 = file_sha256(PROJECT_ROOT / "src/length_budget_distill/paired_rewrite.py")
    if args.skip_complete and _complete_shard(
        manifest_path,
        raw_path,
        config_hash,
        generation_source_sha256,
        rewrite_library_sha256,
    ):
        logging.info("skip_complete_rewrite_shard=%s", raw_path)
        return
    if raw_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing rewrite shard: {raw_path}")

    token_counter = make_token_counter(config)
    ratios = [(str(item["name"]), float(item["ratio"])) for item in paired["ratios"]]
    num_candidates = int(paired["num_candidates"])
    minimum_target_fraction = float(paired["minimum_target_fraction"])
    base_seed = int(paired["base_seed"])
    tasks: List[Task] = []
    for row in assigned:
        source_id = str(row["id"])
        problem_id = source_problem_id(row)
        source_solution = str(row["completion"])
        gold_answer = _gold_answer(row)
        question = _question(row)
        measured_source_tokens = source_token_count(row, token_counter)
        required = essential_step_values(source_solution, gold_answer)
        for ratio_name, ratio in ratios:
            target = target_token_count(measured_source_tokens, ratio)
            minimum = minimum_target_token_count(target, minimum_target_fraction)
            prompt = build_rewrite_prompt(
                question=question,
                gold_answer=gold_answer,
                source_solution=source_solution,
                source_tokens=measured_source_tokens,
                minimum_tokens=minimum,
                target_tokens=target,
                ratio_name=ratio_name,
                required_step_values=required,
            )
            generation_ceiling = min(
                int(paired.get("absolute_generation_ceiling", 512)),
                measured_source_tokens + int(paired.get("generation_headroom_tokens", 32)),
            )
            budget = {
                "name": ratio_name,
                "max_solution_tokens": target,
                "generation_max_new_tokens": generation_ceiling,
            }
            seed = stable_rewrite_seed(base_seed, source_id, ratio_name)
            problem = ProblemRecord(problem_id=problem_id, question=question, answer=gold_answer)
            tasks.append(
                (
                    row,
                    ratio_name,
                    ratio,
                    minimum,
                    target,
                    required,
                    GenerationRequest(problem=problem, budget=budget, prompt=prompt, seed=seed),
                )
            )

    batch_size = int(args.batch_size or paired.get("batch_size", 16))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_hash": config_hash,
                    "source_rows": len(rows),
                    "assigned_rows": len(assigned),
                    "requests": len(tasks),
                    "expected_candidates": len(tasks) * num_candidates,
                    "batch_size": batch_size,
                    "ratios": [name for name, _ in ratios],
                },
                indent=2,
            )
        )
        return

    runtime_config = dict(config)
    runtime_config["teacher"] = dict(config["rewriter"])
    runtime_config["teacher"]["generation"] = dict(config["generation"])
    backend = make_teacher_backend(runtime_config)
    output_records: List[Dict[str, Any]] = []
    started = time.monotonic()
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        candidates_by_request = backend.generate_batch(
            [task[6] for task in batch],
            num_candidates=num_candidates,
        )
        if len(candidates_by_request) != len(batch):
            raise RuntimeError("Rewrite backend returned the wrong request cardinality")
        for task, solutions in zip(batch, candidates_by_request):
            row, ratio_name, ratio, minimum, target, required, request = task
            if len(solutions) != num_candidates:
                raise RuntimeError("Rewrite backend returned the wrong candidate cardinality")
            source_id = str(row["id"])
            source_solution = str(row["completion"])
            source_tokens = source_token_count(row, token_counter)
            gold_answer = request.problem.answer
            for candidate_index, solution in enumerate(solutions):
                quality = assess_rewrite_candidate(
                    solution,
                    gold_answer=gold_answer,
                    source_tokens=source_tokens,
                    minimum_tokens=minimum,
                    target_tokens=target,
                    required_step_values=required,
                    token_counter=token_counter,
                    candidate_index=candidate_index,
                )
                output_records.append(
                    {
                        "rewrite_id": f"{source_id}:{ratio_name}:candidate_{candidate_index:02d}",
                        "source_trace_id": source_id,
                        "problem_id": source_problem_id(row),
                        "source_file_sha256": actual_source_hash,
                        "source_completion_sha256": canonical_sha256(source_solution),
                        "ratio_name": ratio_name,
                        "ratio": ratio,
                        "minimum_tokens": minimum,
                        "target_tokens": target,
                        "candidate_index": candidate_index,
                        "generation_seed": request.seed,
                        "prompt": request.prompt,
                        "solution": solution.strip(),
                        "quality": quality,
                        "config_hash": config_hash,
                    }
                )
        logging.info("rewrite_progress requests=%d/%d", min(start + len(batch), len(tasks)), len(tasks))

    expected_candidates = len(tasks) * num_candidates
    if len(output_records) != expected_candidates:
        raise RuntimeError(
            f"Rewrite record mismatch: expected={expected_candidates} actual={len(output_records)}"
        )
    write_jsonl(raw_path, output_records)
    manifest = {
        "status": "complete",
        "stage": "paired_rewrite_generation",
        "config_hash": config_hash,
        "source_path": str(source_path),
        "source_sha256": actual_source_hash,
        "source_record_count": len(rows),
        "assigned_source_count": len(assigned),
        "request_count": len(tasks),
        "candidate_count": len(output_records),
        "structurally_valid_count": sum(
            bool(record["quality"]["structurally_valid"]) for record in output_records
        ),
        "within_target_count": sum(bool(record["quality"]["within_target"]) for record in output_records),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "generation_source_sha256": generation_source_sha256,
        "rewrite_library_sha256": rewrite_library_sha256,
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(manifest_path, manifest)
    logging.info("rewrite_shard_complete candidates=%d path=%s", len(output_records), raw_path)


def _gold_answer(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    problem_metadata = metadata.get("problem_metadata", {}) if isinstance(metadata, Mapping) else {}
    raw_answer = problem_metadata.get("raw_answer") if isinstance(problem_metadata, Mapping) else None
    gold = extract_final_answer(str(raw_answer or "")) or extract_final_answer(str(row.get("completion", "")))
    if gold is None or not verify_answer(extract_final_answer(str(row.get("completion", ""))), gold):
        raise ValueError(f"Could not recover a verified gold answer for {source_problem_id(row)}")
    return gold


def _question(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    prefix = "Problem:\n"
    suffix = "\n\nSolve the problem"
    if prefix in prompt and suffix in prompt:
        return prompt.split(prefix, 1)[1].split(suffix, 1)[0].strip()
    messages = row.get("messages", [])
    if messages and isinstance(messages[0], Mapping):
        content = str(messages[0].get("content", ""))
        if prefix in content and suffix in content:
            return content.split(prefix, 1)[1].split(suffix, 1)[0].strip()
    raise ValueError(f"Could not recover question text for {source_problem_id(row)}")


def _complete_shard(
    manifest_path: Path,
    raw_path: Path,
    config_hash: str,
    generation_source_sha256: str,
    rewrite_library_sha256: str,
) -> bool:
    if not manifest_path.is_file() or not raw_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    return (
        manifest.get("status") == "complete"
        and manifest.get("config_hash") == config_hash
        and manifest.get("raw_sha256") == file_sha256(raw_path)
        and manifest.get("generation_source_sha256") == generation_source_sha256
        and manifest.get("rewrite_library_sha256") == rewrite_library_sha256
    )


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
