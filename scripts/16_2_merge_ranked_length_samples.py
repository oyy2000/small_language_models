#!/usr/bin/env python3
"""Audit ranked-sampling shards and publish three training-ready SFT datasets."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256, nonempty_line_count
from length_budget_distill.ranked_sampling import (
    LENGTH_LABELS,
    load_bound_problem_ids,
    select_relative_lengths_by_problem,
    validate_ranked_sampling_config,
)
from length_budget_distill.records import (
    TraceRecord,
    read_jsonl,
    trace_from_dict,
    trace_to_dict,
    write_jsonl,
)
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.verifiers import (
    extract_answer_for_verifier,
    verifier_name,
    verifier_version,
    verify_answer_for_verifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_ranked_sampling_7b_v1.json",
    )
    parser.add_argument("--input-dir", required=True, help="Root written by phase 16.1 shards.")
    parser.add_argument("--output-dir", required=True, help="New directory for merged datasets and audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    validate_ranked_sampling_config(config)
    config_hash = canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})
    generation = config["generation"]
    num_shards = int(generation["num_shards"])
    num_candidates = int(generation["num_candidates"])
    input_dir = Path(args.input_dir)
    manifests = [_read_manifest(input_dir, index, num_shards) for index in range(num_shards)]
    source_problem_count, source_problem_ids_sha256 = _validate_manifests(
        manifests,
        config_hash=config_hash,
        num_shards=num_shards,
        num_candidates=num_candidates,
    )

    configured_problem_ids = load_bound_problem_ids(config)
    expected_problem_ids = configured_problem_ids[:source_problem_count]
    if canonical_sha256(expected_problem_ids) != source_problem_ids_sha256:
        raise ValueError(
            "Shard source cohort is not the configured cohort prefix: "
            f"source_problem_count={source_problem_count}"
        )

    raw_by_shard: Dict[int, List[TraceRecord]] = {}
    raw_by_id: Dict[str, TraceRecord] = {}
    for manifest in manifests:
        shard_index = int(manifest["shard_index"])
        rows = [trace_from_dict(row) for row in read_jsonl(_artifact_path(manifest["raw"]))]
        raw_by_shard[shard_index] = rows
        for trace in rows:
            if trace.trace_id in raw_by_id:
                raise ValueError(f"Duplicate raw trace ID: {trace.trace_id}")
            raw_by_id[trace.trace_id] = trace
    raw_traces = [raw_by_id[trace_id] for trace_id in sorted(raw_by_id)]
    _audit_raw_matrix(
        raw_by_shard,
        expected_problem_ids=expected_problem_ids,
        num_shards=num_shards,
        num_candidates=num_candidates,
        config_hash=config_hash,
        configured_verifier=verifier_name(config),
    )

    minimum_unique_correct = int(config["relative_length_selection"].get("minimum_unique_correct", 3))
    recomputed = select_relative_lengths_by_problem(
        raw_traces,
        minimum_unique_correct=minimum_unique_correct,
    )
    expected_selected = {
        label: [recomputed[problem_id][label] for problem_id in sorted(recomputed)]
        for label in LENGTH_LABELS
    }
    _audit_selected_and_sft_shards(manifests, expected_selected)

    output_dir = Path(args.output_dir)
    teacher_name = str(config["teacher"].get("name", config["teacher"]["model_name"]))
    selected_path = output_dir / "selected_traces.jsonl"
    dataset_paths = {
        label: output_dir / "sft" / f"{teacher_name}__relative_{label}.jsonl"
        for label in LENGTH_LABELS
    }
    audit_path = output_dir / "generation_audit.json"
    manifest_path = output_dir / "dataset_manifest.json"
    marker_path = output_dir / "GENERATION_COMPLETE"
    output_artifacts = [selected_path, *dataset_paths.values(), audit_path, manifest_path, marker_path]
    existing = [path for path in output_artifacts if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite merged ranked-sampling artifacts: {existing}")

    selected_flat = [
        expected_selected[label][index]
        for index in range(len(recomputed))
        for label in LENGTH_LABELS
    ]
    write_jsonl(selected_path, (trace_to_dict(trace) for trace in selected_flat))
    datasets: List[Dict[str, Any]] = []
    for label in LENGTH_LABELS:
        traces = expected_selected[label]
        path = dataset_paths[label]
        write_jsonl(path, (trace_to_sft_record(trace) for trace in traces))
        token_counts = [int(trace.solution_token_count) for trace in traces]
        datasets.append(
            {
                "label": label,
                "budget_name": f"relative_{label}",
                "train_path": str(path),
                "train_sha256": file_sha256(path),
                "record_count": len(traces),
                "supervised_tokens": sum(token_counts),
                "mean_solution_tokens": statistics.fmean(token_counts) if token_counts else None,
                "median_solution_tokens": statistics.median(token_counts) if token_counts else None,
                "min_solution_tokens": min(token_counts) if token_counts else None,
                "max_solution_tokens": max(token_counts) if token_counts else None,
            }
        )

    eligible_problem_ids = sorted(recomputed)
    dropped_problem_ids = sorted(set(expected_problem_ids) - set(eligible_problem_ids))
    ties = _length_tie_counts(recomputed)
    audit = {
        "status": "passed",
        "experiment_name": config.get("experiment_name"),
        "config_hash": config_hash,
        "source_problem_count": source_problem_count,
        "source_problem_ids_sha256": source_problem_ids_sha256,
        "num_shards": num_shards,
        "num_candidates": num_candidates,
        "expected_raw_record_count": source_problem_count * num_candidates,
        "actual_raw_record_count": len(raw_traces),
        "correct_candidate_count": sum(trace.is_correct for trace in raw_traces),
        "eligible_problem_count": len(eligible_problem_ids),
        "dropped_problem_count": len(dropped_problem_ids),
        "dropped_problem_ids": dropped_problem_ids,
        "minimum_unique_correct": minimum_unique_correct,
        "length_tie_counts": ties,
        "identical_problem_cohort_across_labels": True,
        "verifier": verifier_name(config),
        "verifier_version": verifier_version(verifier_name(config)),
        "input_manifests": [
            {
                "path": str(manifest["_manifest_path"]),
                "sha256": file_sha256(manifest["_manifest_path"]),
                "shard_index": manifest["shard_index"],
            }
            for manifest in manifests
        ],
        "selected_traces": _artifact_entry(selected_path),
        "datasets": datasets,
    }
    _write_json(audit_path, audit)
    manifest = {
        "status": "complete",
        "experiment_name": config.get("experiment_name"),
        "config_path": args.config,
        "config_hash": config_hash,
        "generation_audit_path": str(audit_path),
        "generation_audit_sha256": file_sha256(audit_path),
        "selected_traces_path": str(selected_path),
        "selected_traces_sha256": file_sha256(selected_path),
        "eligible_problem_count": len(eligible_problem_ids),
        "problem_ids": eligible_problem_ids,
        "datasets": datasets,
    }
    _write_json(manifest_path, manifest)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"config_hash={config_hash}\n"
        f"dataset_manifest_sha256={file_sha256(manifest_path)}\n"
        f"generation_audit_sha256={file_sha256(audit_path)}\n"
        f"eligible_problem_count={len(eligible_problem_ids)}\n",
        encoding="utf-8",
    )
    logging.info(
        "generation_merge_complete raw=%d eligible=%d dropped=%d output=%s",
        len(raw_traces),
        len(eligible_problem_ids),
        len(dropped_problem_ids),
        output_dir,
    )


def _read_manifest(input_dir: Path, shard_index: int, num_shards: int) -> Dict[str, Any]:
    suffix = f"shard_{shard_index:05d}_of_{num_shards:05d}"
    path = input_dir / "manifests" / f"{suffix}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ranked-sampling shard manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["_manifest_path"] = str(path)
    return manifest


def _validate_manifests(
    manifests: Sequence[Mapping[str, Any]],
    config_hash: str,
    num_shards: int,
    num_candidates: int,
) -> tuple[int, str]:
    source_counts = {int(manifest.get("source_problem_count", -1)) for manifest in manifests}
    source_hashes = {str(manifest.get("source_problem_ids_sha256")) for manifest in manifests}
    if len(source_counts) != 1 or len(source_hashes) != 1:
        raise ValueError("Shard manifests disagree on the source cohort.")
    if next(iter(source_counts)) < 0 or next(iter(source_hashes)) in {"", "None"}:
        raise ValueError("Shard manifests do not identify a valid source cohort.")
    for expected_index, manifest in enumerate(manifests):
        if manifest.get("status") != "complete":
            raise ValueError(f"Shard manifest is not complete: {manifest.get('_manifest_path')}")
        if manifest.get("config_hash") != config_hash:
            raise ValueError(f"Shard/config hash mismatch: {manifest.get('_manifest_path')}")
        if int(manifest.get("shard_index", -1)) != expected_index:
            raise ValueError(f"Unexpected shard index: {manifest.get('_manifest_path')}")
        if int(manifest.get("num_shards", -1)) != num_shards:
            raise ValueError(f"Unexpected shard topology: {manifest.get('_manifest_path')}")
        if int(manifest.get("num_candidates", -1)) != num_candidates:
            raise ValueError(f"Unexpected candidate count: {manifest.get('_manifest_path')}")
        artifacts = [manifest["raw"]]
        artifacts.extend(manifest["selected"].values())
        artifacts.extend(manifest["sft"].values())
        for artifact in artifacts:
            _validate_artifact(artifact)
    return next(iter(source_counts)), next(iter(source_hashes))


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    path = _artifact_path(artifact)
    if not path.is_file():
        raise FileNotFoundError(f"Missing shard artifact: {path}")
    if file_sha256(path) != artifact.get("sha256"):
        raise ValueError(f"Shard artifact hash mismatch: {path}")
    if nonempty_line_count(path) != int(artifact.get("record_count", -1)):
        raise ValueError(f"Shard artifact cardinality mismatch: {path}")


def _artifact_path(artifact: Mapping[str, Any]) -> Path:
    return Path(str(artifact["path"]))


def _audit_raw_matrix(
    raw_by_shard: Mapping[int, Sequence[TraceRecord]],
    expected_problem_ids: Sequence[str],
    num_shards: int,
    num_candidates: int,
    config_hash: str,
    configured_verifier: str,
) -> None:
    expected_indices = set(range(num_candidates))
    for shard_index in range(num_shards):
        traces = raw_by_shard.get(shard_index, [])
        grouped: Dict[str, List[TraceRecord]] = defaultdict(list)
        for trace in traces:
            grouped[trace.problem_id].append(trace)
        expected_shard_ids = set(expected_problem_ids[shard_index::num_shards])
        if set(grouped) != expected_shard_ids:
            missing = sorted(expected_shard_ids - set(grouped))
            unexpected = sorted(set(grouped) - expected_shard_ids)
            raise ValueError(
                f"Shard problem assignment mismatch for shard={shard_index}: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        for problem_id, candidates in grouped.items():
            indices = {trace.candidate_index for trace in candidates}
            if len(candidates) != num_candidates or indices != expected_indices:
                raise ValueError(
                    f"Candidate matrix mismatch for problem={problem_id}: "
                    f"count={len(candidates)} indices={sorted(indices)}"
                )
            for trace in candidates:
                if trace.config_hash != config_hash:
                    raise ValueError(f"Raw trace/config hash mismatch: {trace.trace_id}")
                predicted = extract_answer_for_verifier(trace.solution, configured_verifier)
                correct = verify_answer_for_verifier(predicted, trace.answer, configured_verifier)
                if predicted != trace.predicted_answer or correct != trace.is_correct:
                    raise ValueError(f"Stored verification mismatch: {trace.trace_id}")


def _audit_selected_and_sft_shards(
    manifests: Sequence[Mapping[str, Any]],
    expected_selected: Mapping[str, Sequence[TraceRecord]],
) -> None:
    expected_by_label = {
        label: {trace.trace_id: trace for trace in traces}
        for label, traces in expected_selected.items()
    }
    observed_by_label: Dict[str, Dict[str, TraceRecord]] = {
        label: {} for label in LENGTH_LABELS
    }
    observed_sft_by_label: Dict[str, Dict[str, Dict[str, Any]]] = {
        label: {} for label in LENGTH_LABELS
    }
    for manifest in manifests:
        for label in LENGTH_LABELS:
            for row in read_jsonl(_artifact_path(manifest["selected"][label])):
                trace = trace_from_dict(row)
                if trace.trace_id in observed_by_label[label]:
                    raise ValueError(f"Duplicate selected trace for label={label}: {trace.trace_id}")
                observed_by_label[label][trace.trace_id] = trace
            for row in read_jsonl(_artifact_path(manifest["sft"][label])):
                record_id = str(row.get("id"))
                if record_id in observed_sft_by_label[label]:
                    raise ValueError(f"Duplicate SFT record for label={label}: {record_id}")
                observed_sft_by_label[label][record_id] = row
    for label in LENGTH_LABELS:
        expected = expected_by_label[label]
        observed = observed_by_label[label]
        if set(observed) != set(expected):
            raise ValueError(f"Selected trace IDs do not match recomputed {label} selection.")
        for trace_id, expected_trace in expected.items():
            if trace_to_dict(observed[trace_id]) != trace_to_dict(expected_trace):
                raise ValueError(f"Selected trace payload mismatch: {trace_id}")
            expected_sft = trace_to_sft_record(expected_trace)
            if observed_sft_by_label[label].get(trace_id) != expected_sft:
                raise ValueError(f"SFT shard payload mismatch: {trace_id}")


def _length_tie_counts(selected: Mapping[str, Mapping[str, TraceRecord]]) -> Dict[str, int]:
    short_medium = 0
    medium_long = 0
    short_long = 0
    for traces in selected.values():
        short_tokens = traces["short"].solution_token_count
        medium_tokens = traces["medium"].solution_token_count
        long_tokens = traces["long"].solution_token_count
        if not short_tokens <= medium_tokens <= long_tokens:
            raise ValueError("Relative-length token ordering invariant failed.")
        short_medium += int(short_tokens == medium_tokens)
        medium_long += int(medium_tokens == long_tokens)
        short_long += int(short_tokens == long_tokens)
    return {
        "short_equals_medium": short_medium,
        "medium_equals_long": medium_long,
        "short_equals_long": short_long,
    }


def _artifact_entry(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "record_count": nonempty_line_count(path),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
