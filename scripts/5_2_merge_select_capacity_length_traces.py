#!/usr/bin/env python3
"""Audit, merge, and select capacity-length factorial trace shards."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    candidate_audit_rows,
    canonical_sha256,
    common_problem_ids,
    expected_conditions,
    file_sha256,
    nonempty_line_count,
    select_shortest_correct,
)
from length_budget_distill.records import read_jsonl, trace_from_dict, trace_to_dict, write_jsonl
from length_budget_distill.verifiers import VERIFIER_VERSION, extract_final_answer, verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument("--input-glob", required=True, help="Recursive glob for raw shard JSONL files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
    parser.add_argument("--expected-problems", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config_for_hash = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(config_for_hash)
    num_candidates = int(config.get("generation", {}).get("num_candidates", 3))
    conditions = expected_conditions(config)
    if not conditions:
        raise ValueError("No factorial conditions were configured.")

    paths = [Path(path) for path in sorted(glob.glob(args.input_glob, recursive=True))]
    if not paths:
        raise FileNotFoundError(f"No raw shards matched {args.input_glob!r}")
    by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: List[str] = []
    for path in paths:
        for row in read_jsonl(path):
            trace_id = str(row["trace_id"])
            if trace_id in by_id:
                duplicate_ids.append(trace_id)
            else:
                by_id[trace_id] = row
    if duplicate_ids:
        raise ValueError(f"Duplicate trace IDs detected; examples={duplicate_ids[:10]}")

    stored_traces = [trace_from_dict(by_id[key]) for key in sorted(by_id)]
    verification_mismatches: List[Dict[str, Any]] = []
    traces = []
    for trace in stored_traces:
        predicted_answer = extract_final_answer(trace.solution)
        reverified_correct = verify_answer(predicted_answer, trace.answer)
        if reverified_correct != trace.is_correct or predicted_answer != trace.predicted_answer:
            verification_mismatches.append(
                {
                    "trace_id": trace.trace_id,
                    "stored_predicted_answer": trace.predicted_answer,
                    "reverified_predicted_answer": predicted_answer,
                    "stored_is_correct": trace.is_correct,
                    "reverified_is_correct": reverified_correct,
                }
            )
        traces.append(
            replace(
                trace,
                predicted_answer=predicted_answer,
                is_correct=reverified_correct,
            )
        )
    mismatched_hashes = sorted({trace.config_hash for trace in traces if trace.config_hash != config_hash})
    if mismatched_hashes:
        raise ValueError(f"Trace/config hash mismatch: expected={config_hash} observed={mismatched_hashes}")

    audit_rows = candidate_audit_rows(traces, conditions)
    expected_per_condition = args.expected_problems * num_candidates
    cardinality_errors = [
        row
        for row in audit_rows
        if row["candidate_count"] != expected_per_condition or row["problem_count"] != args.expected_problems
    ]
    if cardinality_errors:
        raise ValueError(
            f"Incomplete condition matrix; expected_candidates={expected_per_condition} "
            f"expected_problems={args.expected_problems} errors={cardinality_errors}"
        )

    candidate_sets: Dict[tuple[str, str, str], set[int]] = {}
    for trace in traces:
        generator_name = trace.generator_name or trace.teacher_model
        key = (generator_name, trace.budget_name, trace.problem_id)
        candidate_sets.setdefault(key, set()).add(trace.candidate_index)
    expected_indices = set(range(num_candidates))
    bad_candidate_sets = [key for key, values in candidate_sets.items() if values != expected_indices]
    if bad_candidate_sets:
        raise ValueError(f"Missing or unexpected candidate indices; examples={bad_candidate_sets[:10]}")

    selected = select_shortest_correct(traces)
    common_ids = common_problem_ids(selected, conditions)
    minimum_common = int(
        config.get("balancing", {}).get(
            f"{args.stage}_min_common_problems",
            50 if args.stage == "smoke" else 500,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "traces_merged.jsonl"
    selected_path = output_dir / "selected_traces.jsonl"
    common_path = output_dir / "common_problem_ids.json"
    audit_path = output_dir / "selection_audit.json"
    marker_path = output_dir / "SELECTION_COMPLETE"
    existing = [path for path in (merged_path, selected_path, common_path, audit_path, marker_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing merged artifacts: {existing}")

    write_jsonl(merged_path, (trace_to_dict(trace) for trace in traces))
    write_jsonl(selected_path, (trace_to_dict(trace) for trace in selected))
    _write_json(common_path, {"problem_ids": common_ids, "count": len(common_ids)})
    audit = {
        "status": "passed" if len(common_ids) >= minimum_common else "failed_common_problem_gate",
        "stage": args.stage,
        "config_hash": config_hash,
        "input_shards": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "record_count": nonempty_line_count(path),
            }
            for path in paths
        ],
        "input_shard_count": len(paths),
        "expected_problems": args.expected_problems,
        "num_candidates": num_candidates,
        "expected_total_candidates": len(conditions) * args.expected_problems * num_candidates,
        "actual_total_candidates": len(traces),
        "verifier_version": VERIFIER_VERSION,
        "verification_mismatch_count": len(verification_mismatches),
        "verification_mismatch_examples": verification_mismatches[:50],
        "selected_trace_count": len(selected),
        "common_problem_count": len(common_ids),
        "minimum_common_problem_count": minimum_common,
        "conditions": audit_rows,
        "merged_path": str(merged_path),
        "merged_sha256": file_sha256(merged_path),
        "selected_path": str(selected_path),
        "selected_sha256": file_sha256(selected_path),
        "common_problem_ids_path": str(common_path),
        "common_problem_ids_sha256": file_sha256(common_path),
    }
    _write_json(audit_path, audit)
    if len(common_ids) < minimum_common:
        raise SystemExit(
            f"Common-problem gate failed: actual={len(common_ids)} required={minimum_common}; "
            f"audit={audit_path}"
        )
    marker_path.write_text(f"config_hash={config_hash}\ncommon_problem_count={len(common_ids)}\n", encoding="utf-8")
    logging.info(
        "selection_complete raw=%d selected=%d common=%d output=%s",
        len(traces),
        len(selected),
        len(common_ids),
        output_dir,
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
