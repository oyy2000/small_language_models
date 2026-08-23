#!/usr/bin/env python3
"""Generate sharded multi-candidate traces for one capacity-length generator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.backends import GenerationRequest, make_teacher_backend
from length_budget_distill.bucketing import get_length_budgets, iter_shard
from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    generator_by_name,
    nonempty_line_count,
    runtime_metadata,
    stable_generation_seed,
)
from length_budget_distill.prompts import build_teacher_prompt, get_prompt_strategy
from length_budget_distill.records import ProblemRecord, TraceRecord, trace_to_dict, write_jsonl
from length_budget_distill.tokenization import make_token_counter
from length_budget_distill.verifiers import (
    extract_answer_for_verifier,
    verifier_name,
    verifier_version,
    verify_answer_for_verifier,
)


Task = Tuple[int, ProblemRecord, Dict[str, Any], GenerationRequest]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument("--generator-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Limit source problems before sharding.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.num_shards <= 0 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("Invalid shard topology.")

    config = load_config(args.config)
    config_for_hash = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(config_for_hash)
    configured_verifier = verifier_name(config)
    generator = generator_by_name(config, args.generator_name)
    generation = dict(config.get("generation", {}))
    num_candidates = int(generation.get("num_candidates", 3))
    if num_candidates <= 0:
        raise ValueError("generation.num_candidates must be positive.")
    batch_size = int(args.batch_size or generation.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    teacher_config = dict(generator)
    resolved_model_path = generator.get("model_path")
    if resolved_model_path:
        model_path = Path(str(resolved_model_path))
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Configured local model snapshot does not exist for {args.generator_name}: {model_path}"
            )
        teacher_config["model_name"] = str(model_path)
        teacher_config["tokenizer_name"] = str(model_path)
    teacher_config["generation"] = generation
    runtime_config = dict(config)
    runtime_config["teacher"] = teacher_config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}"
    raw_path = output_dir / f"{shard_suffix}.jsonl"
    manifest_path = output_dir / "manifests" / f"{shard_suffix}.json"
    if args.skip_existing and _is_complete(manifest_path, raw_path, config_hash):
        logging.info("skip_complete_shard=%s", raw_path)
        return
    if raw_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an incomplete/existing shard: {raw_path}; "
            "remove or move the exact shard after auditing it, or use a new output directory."
        )

    problems = load_problem_records(runtime_config)
    if args.limit is not None:
        problems = problems[: args.limit]
    shard_problems = list(iter_shard(problems, args.num_shards, args.shard_index))
    budgets = get_length_budgets(runtime_config)
    prompt_strategy = get_prompt_strategy(runtime_config)
    base_seed = int(generation.get("base_seed", 0))
    tasks: List[Task] = []
    for source_index, problem in shard_problems:
        for budget in budgets:
            prompt = build_teacher_prompt(problem, budget, runtime_config)
            seed = stable_generation_seed(base_seed, args.generator_name, budget["name"], problem.problem_id)
            tasks.append(
                (
                    source_index,
                    problem,
                    budget,
                    GenerationRequest(problem=problem, budget=budget, prompt=prompt, seed=seed),
                )
            )

    logging.info(
        "experiment=%s generator=%s model=%s shard=%d/%d source_problems=%d shard_problems=%d "
        "requests=%d candidates_per_request=%d expected_records=%d batch_size=%d config_hash=%s",
        config.get("experiment_name"),
        args.generator_name,
        generator["model_name"],
        args.shard_index,
        args.num_shards,
        len(problems),
        len(shard_problems),
        len(tasks),
        num_candidates,
        len(tasks) * num_candidates,
        batch_size,
        config_hash,
    )

    backend = make_teacher_backend(runtime_config)
    token_counter = make_token_counter(runtime_config)
    traces: List[TraceRecord] = []
    started = time.monotonic()
    completed_requests = 0
    last_logged_requests = 0
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        candidates_by_request = backend.generate_batch(
            [item[3] for item in batch],
            num_candidates=num_candidates,
        )
        if len(candidates_by_request) != len(batch):
            raise RuntimeError("Generation backend returned the wrong batch cardinality.")
        for (source_index, problem, budget, request), candidates in zip(batch, candidates_by_request):
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
                trace_id = (
                    f"{problem.problem_id}:{args.generator_name}:{budget['name']}:"
                    f"candidate_{candidate_index:02d}"
                )
                traces.append(
                    TraceRecord(
                        trace_id=trace_id,
                        problem_id=problem.problem_id,
                        question=problem.question,
                        answer=problem.answer,
                        budget_name=str(budget["name"]),
                        max_solution_tokens=int(budget["max_solution_tokens"]),
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
                            "prompt_strategy": prompt_strategy,
                            "source_index": source_index,
                            "num_candidates": num_candidates,
                            "generator_model_id": generator["model_name"],
                            "resolved_model_path": resolved_model_path,
                        },
                        generator_name=args.generator_name,
                        generator_size_b=float(generator["size_b"]),
                        candidate_index=candidate_index,
                        generation_seed=request.seed,
                        budget_compliant=token_count <= int(budget["max_solution_tokens"]),
                        selected_for_sft=False,
                        config_hash=config_hash,
                        source_hash=source_hash,
                    )
                )
        completed_requests += len(batch)
        if (
            last_logged_requests == 0
            or completed_requests - last_logged_requests >= max(1, args.log_every)
            or completed_requests == len(tasks)
        ):
            elapsed = max(time.monotonic() - started, 1e-9)
            logging.info(
                "progress requests=%d/%d records=%d rate=%.3f_requests_per_sec",
                completed_requests,
                len(tasks),
                len(traces),
                completed_requests / elapsed,
            )
            last_logged_requests = completed_requests

    expected_records = len(tasks) * num_candidates
    if len(traces) != expected_records:
        raise RuntimeError(f"Trace cardinality mismatch: expected={expected_records} actual={len(traces)}")
    write_jsonl(raw_path, (trace_to_dict(trace) for trace in traces))
    manifest = {
        "status": "complete",
        "experiment_name": config.get("experiment_name"),
        "generator_name": args.generator_name,
        "generator_model": generator["model_name"],
        "resolved_model_path": resolved_model_path,
        "generator_size_b": generator["size_b"],
        "config_path": args.config,
        "config_hash": config_hash,
        "num_source_problems": len(problems),
        "num_shard_problems": len(shard_problems),
        "num_budgets": len(budgets),
        "num_candidates": num_candidates,
        "record_count": len(traces),
        "correct_candidate_count": sum(trace.is_correct for trace in traces),
        "budget_compliant_count": sum(bool(trace.budget_compliant) for trace in traces),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "raw_path": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "verifier": configured_verifier,
        "verifier_version": verifier_version(configured_verifier),
        "elapsed_seconds": time.monotonic() - started,
        "model_revision": getattr(getattr(backend, "tokenizer", None), "init_kwargs", {}).get("_commit_hash"),
        "runtime": runtime_metadata(),
    }
    _write_json(manifest_path, manifest)
    logging.info("wrote_records=%d raw=%s manifest=%s", len(traces), raw_path, manifest_path)


def _is_complete(manifest_path: Path, raw_path: Path, config_hash: str) -> bool:
    if not manifest_path.is_file() or not raw_path.is_file():
        return False
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
        return False
    count = nonempty_line_count(raw_path)
    if count != int(manifest.get("record_count", -1)):
        return False
    recorded_hash = manifest.get("raw_sha256")
    return recorded_hash is None or recorded_hash == file_sha256(raw_path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
