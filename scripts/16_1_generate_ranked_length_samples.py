#!/usr/bin/env python3
"""Sample one teacher repeatedly and select relative short/medium/long traces."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.backends import GenerationRequest, make_teacher_backend
from length_budget_distill.bucketing import iter_shard
from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    nonempty_line_count,
    runtime_metadata,
    stable_generation_seed,
)
from length_budget_distill.ranked_sampling import (
    LENGTH_LABELS,
    build_length_agnostic_teacher_prompt,
    load_bound_problem_ids,
    require_cohort_problems,
    select_relative_lengths_by_problem,
    unique_correct_candidates,
    validate_ranked_sampling_config,
)
from length_budget_distill.records import ProblemRecord, TraceRecord, trace_to_dict, write_jsonl
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.tokenization import make_token_counter
from length_budget_distill.verifiers import (
    extract_answer_for_verifier,
    verifier_name,
    verifier_version,
    verify_answer_for_verifier,
)


Task = Tuple[int, ProblemRecord, GenerationRequest]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_ranked_sampling_7b_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit the bound cohort before sharding.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the config and cohort without loading the teacher or writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    validate_ranked_sampling_config(config)
    config_hash = canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})
    generation = dict(config["generation"])
    num_shards = int(args.num_shards or generation.get("num_shards", 1))
    if num_shards <= 0 or not 0 <= args.shard_index < num_shards:
        raise ValueError("Invalid shard topology.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    num_candidates = int(generation["num_candidates"])
    batch_size = int(args.batch_size or generation.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    cohort_ids = load_bound_problem_ids(config)
    problems = require_cohort_problems(load_problem_records(config), cohort_ids)
    if args.limit is not None:
        problems = problems[: args.limit]
    source_problem_ids_hash = canonical_sha256([problem.problem_id for problem in problems])
    shard_problems = list(iter_shard(problems, num_shards, args.shard_index))
    expected_records = len(shard_problems) * num_candidates
    logging.info(
        "experiment=%s teacher=%s source_problems=%d shard=%d/%d shard_problems=%d "
        "candidates_per_problem=%d expected_records=%d config_hash=%s",
        config.get("experiment_name"),
        config.get("teacher", {}).get("model_name"),
        len(problems),
        args.shard_index,
        num_shards,
        len(shard_problems),
        num_candidates,
        expected_records,
        config_hash,
    )
    if args.preflight_only:
        return

    output_dir = Path(args.output_dir)
    suffix = f"shard_{args.shard_index:05d}_of_{num_shards:05d}"
    raw_path = output_dir / "raw" / f"{suffix}.jsonl"
    selected_paths = {
        label: output_dir / "selected_shards" / label / f"{suffix}.jsonl"
        for label in LENGTH_LABELS
    }
    sft_paths = {
        label: output_dir / "sft_shards" / label / f"{suffix}.jsonl"
        for label in LENGTH_LABELS
    }
    manifest_path = output_dir / "manifests" / f"{suffix}.json"
    artifact_paths = [raw_path, *selected_paths.values(), *sft_paths.values()]
    if args.skip_existing and _is_complete(manifest_path, config_hash):
        logging.info("skip_complete_shard=%s", manifest_path)
        return
    existing = [path for path in [manifest_path, *artifact_paths] if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing ranked-sampling shard artifacts: "
            f"{existing}"
        )

    teacher_config = dict(config["teacher"])
    teacher_config["generation"] = generation
    runtime_config = dict(config)
    runtime_config["teacher"] = teacher_config
    backend = make_teacher_backend(runtime_config)
    token_counter = make_token_counter(runtime_config)
    teacher_name = str(teacher_config.get("name", teacher_config["model_name"]))
    teacher_size_b = teacher_config.get("size_b")
    max_new_tokens = int(generation["max_new_tokens"])
    pool_budget = {
        "name": "unconstrained_sample_pool",
        "max_solution_tokens": max_new_tokens,
        "generation_max_new_tokens": max_new_tokens,
    }
    base_seed = int(generation.get("base_seed", 0))
    tasks: List[Task] = []
    for source_index, problem in shard_problems:
        prompt = build_length_agnostic_teacher_prompt(problem)
        seed = stable_generation_seed(
            base_seed,
            teacher_name,
            str(pool_budget["name"]),
            problem.problem_id,
        )
        tasks.append(
            (
                source_index,
                problem,
                GenerationRequest(problem=problem, budget=pool_budget, prompt=prompt, seed=seed),
            )
        )

    configured_verifier = verifier_name(config)
    traces: List[TraceRecord] = []
    started = time.monotonic()
    completed = 0
    last_logged = 0
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        candidates_by_request = backend.generate_batch(
            [item[2] for item in batch],
            num_candidates=num_candidates,
        )
        if len(candidates_by_request) != len(batch):
            raise RuntimeError("Generation backend returned the wrong request cardinality.")
        for (source_index, problem, request), candidates in zip(batch, candidates_by_request):
            if len(candidates) != num_candidates:
                raise RuntimeError(
                    f"Expected {num_candidates} candidates for {problem.problem_id}, got {len(candidates)}."
                )
            source_hash = canonical_sha256(
                {"problem_id": problem.problem_id, "question": problem.question, "answer": problem.answer}
            )
            for candidate_index, solution in enumerate(candidates):
                predicted = extract_answer_for_verifier(solution, configured_verifier)
                token_count = token_counter.count(solution)
                traces.append(
                    TraceRecord(
                        trace_id=(
                            f"{problem.problem_id}:{teacher_name}:unconstrained_sample_pool:"
                            f"candidate_{candidate_index:02d}"
                        ),
                        problem_id=problem.problem_id,
                        question=problem.question,
                        answer=problem.answer,
                        budget_name="unconstrained_sample_pool",
                        max_solution_tokens=max_new_tokens,
                        teacher_backend=backend.backend_name,
                        teacher_model=backend.model_name,
                        prompt=request.prompt if config.get("output", {}).get("include_prompt_in_trace", True) else "",
                        solution=solution,
                        predicted_answer=predicted,
                        is_correct=verify_answer_for_verifier(
                            predicted,
                            problem.answer,
                            configured_verifier,
                        ),
                        solution_token_count=token_count,
                        metadata={
                            "problem_metadata": problem.metadata,
                            "prompt_strategy": "length_agnostic_same_prompt_sampling",
                            "source_index": source_index,
                            "num_candidates": num_candidates,
                            "source_cohort_problem_ids_sha256": source_problem_ids_hash,
                        },
                        generator_name=teacher_name,
                        generator_size_b=float(teacher_size_b) if teacher_size_b is not None else None,
                        candidate_index=candidate_index,
                        generation_seed=request.seed,
                        budget_compliant=token_count <= max_new_tokens,
                        selected_for_sft=False,
                        config_hash=config_hash,
                        source_hash=source_hash,
                    )
                )
        completed += len(batch)
        if (
            last_logged == 0
            or completed - last_logged >= max(1, args.log_every)
            or completed == len(tasks)
        ):
            elapsed = max(time.monotonic() - started, 1e-9)
            logging.info(
                "progress requests=%d/%d records=%d rate=%.3f_requests_per_sec",
                completed,
                len(tasks),
                len(traces),
                completed / elapsed,
            )
            last_logged = completed

    if len(traces) != expected_records:
        raise RuntimeError(f"Raw trace cardinality mismatch: expected={expected_records} actual={len(traces)}")
    minimum_unique_correct = int(config["relative_length_selection"].get("minimum_unique_correct", 3))
    selected_by_problem = select_relative_lengths_by_problem(
        traces,
        minimum_unique_correct=minimum_unique_correct,
    )
    selected_by_label = {
        label: [selected_by_problem[problem_id][label] for problem_id in sorted(selected_by_problem)]
        for label in LENGTH_LABELS
    }

    write_jsonl(raw_path, (trace_to_dict(trace) for trace in traces))
    for label in LENGTH_LABELS:
        write_jsonl(selected_paths[label], (trace_to_dict(trace) for trace in selected_by_label[label]))
        write_jsonl(sft_paths[label], (trace_to_sft_record(trace) for trace in selected_by_label[label]))

    dropped_problem_ids = sorted(
        {problem.problem_id for _, problem in shard_problems} - set(selected_by_problem)
    )
    manifest = {
        "status": "complete",
        "experiment_name": config.get("experiment_name"),
        "config_path": args.config,
        "config_hash": config_hash,
        "teacher_name": teacher_name,
        "teacher_model": backend.model_name,
        "teacher_backend": backend.backend_name,
        "source_problem_count": len(problems),
        "source_problem_ids_sha256": source_problem_ids_hash,
        "shard_problem_count": len(shard_problems),
        "shard_index": args.shard_index,
        "num_shards": num_shards,
        "num_candidates": num_candidates,
        "raw_record_count": len(traces),
        "correct_candidate_count": sum(trace.is_correct for trace in traces),
        "unique_correct_candidate_count": sum(
            len(unique_correct_candidates([trace for trace in traces if trace.problem_id == problem.problem_id]))
            for _, problem in shard_problems
        ),
        "eligible_problem_count": len(selected_by_problem),
        "dropped_problem_count": len(dropped_problem_ids),
        "dropped_problem_ids": dropped_problem_ids,
        "minimum_unique_correct": minimum_unique_correct,
        "verifier": configured_verifier,
        "verifier_version": verifier_version(configured_verifier),
        "raw": _artifact_entry(raw_path),
        "selected": {label: _artifact_entry(selected_paths[label]) for label in LENGTH_LABELS},
        "sft": {label: _artifact_entry(sft_paths[label]) for label in LENGTH_LABELS},
        "length_summary": {
            label: _length_summary(selected_by_label[label]) for label in LENGTH_LABELS
        },
        "elapsed_seconds": time.monotonic() - started,
        "runtime": runtime_metadata(),
    }
    _write_json(manifest_path, manifest)
    logging.info(
        "ranked_sampling_complete raw=%d eligible_problems=%d dropped_problems=%d manifest=%s",
        len(traces),
        len(selected_by_problem),
        len(dropped_problem_ids),
        manifest_path,
    )


def _length_summary(traces: Sequence[TraceRecord]) -> Dict[str, Any]:
    values = [int(trace.solution_token_count) for trace in traces]
    if not values:
        return {"count": 0, "mean_tokens": None, "median_tokens": None, "min_tokens": None, "max_tokens": None}
    return {
        "count": len(values),
        "mean_tokens": statistics.fmean(values),
        "median_tokens": statistics.median(values),
        "min_tokens": min(values),
        "max_tokens": max(values),
    }


def _artifact_entry(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "record_count": nonempty_line_count(path),
    }


def _is_complete(manifest_path: Path, config_hash: str) -> bool:
    if not manifest_path.is_file():
        return False
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
        return False
    artifacts = [manifest.get("raw", {})]
    artifacts.extend(manifest.get("selected", {}).values())
    artifacts.extend(manifest.get("sft", {}).values())
    for artifact in artifacts:
        path = Path(str(artifact.get("path", "")))
        if not path.is_file():
            return False
        if file_sha256(path) != artifact.get("sha256"):
            return False
        if nonempty_line_count(path) != int(artifact.get("record_count", -1)):
            return False
    return True


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
