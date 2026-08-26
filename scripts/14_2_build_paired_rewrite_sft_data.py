#!/usr/bin/env python3
"""Audit paired rewrite shards and build matched 881-example SFT datasets."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.paired_rewrite import (
    assess_rewrite_candidate,
    essential_step_values,
    minimum_target_token_count,
    paired_sft_record,
    select_adaptive_rewrite,
    source_problem_id,
    source_token_count,
    target_token_count,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.tokenization import make_token_counter
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_paired_rewrite_7b_pilot_v1.json",
    )
    parser.add_argument("--rewrite-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    paired = dict(config["paired_rewrite"])
    expected_records = int(paired["expected_records"])
    source_path = _resolve(str(paired["source_standard_path"]))
    direct_short_path = _resolve(str(paired["source_direct_short_path"]))
    _require_hash(source_path, str(paired["source_standard_sha256"]))
    _require_hash(direct_short_path, str(paired["source_direct_short_sha256"]))
    source_rows = list(read_jsonl(source_path))
    short_rows = list(read_jsonl(direct_short_path))
    source_by_problem = _index_source(source_rows, expected_records, "standard")
    short_by_problem = _index_source(short_rows, expected_records, "direct short")
    if set(source_by_problem) != set(short_by_problem):
        raise ValueError("Standard and direct-short problem support differ")

    paths = [Path(path) for path in sorted(glob.glob(args.rewrite_glob))]
    if not paths:
        raise FileNotFoundError(f"No rewrite shards matched {args.rewrite_glob!r}")
    expected_generation_sha256 = file_sha256(PROJECT_ROOT / "scripts/14_1_generate_paired_rewrites.py")
    expected_rewrite_library_sha256 = file_sha256(
        PROJECT_ROOT / "src/length_budget_distill/paired_rewrite.py"
    )
    raw_records: Dict[str, Mapping[str, Any]] = {}
    for path in paths:
        shard_manifest_path = path.parent / "manifests" / f"{path.stem}.json"
        shard_manifest = _read_json(shard_manifest_path)
        if shard_manifest.get("status") != "complete":
            raise ValueError(f"Incomplete rewrite shard manifest: {shard_manifest_path}")
        if shard_manifest.get("config_hash") != config_hash:
            raise ValueError(f"Rewrite shard config mismatch: {shard_manifest_path}")
        if shard_manifest.get("raw_sha256") != file_sha256(path):
            raise ValueError(f"Rewrite shard hash mismatch: {path}")
        if shard_manifest.get("generation_source_sha256") != expected_generation_sha256:
            raise ValueError(f"Rewrite generation source mismatch: {shard_manifest_path}")
        if shard_manifest.get("rewrite_library_sha256") != expected_rewrite_library_sha256:
            raise ValueError(f"Rewrite library source mismatch: {shard_manifest_path}")
        for record in read_jsonl(path):
            rewrite_id = str(record["rewrite_id"])
            if rewrite_id in raw_records:
                raise ValueError(f"Duplicate rewrite ID: {rewrite_id}")
            if record.get("config_hash") != config_hash:
                raise ValueError(f"Rewrite/config hash mismatch in {path}: {rewrite_id}")
            raw_records[rewrite_id] = record

    token_counter = make_token_counter(config)
    ratio_specs = {str(item["name"]): dict(item) for item in paired["ratios"]}
    expected_candidates = int(paired["num_candidates"])
    minimum_target_fraction = float(paired["minimum_target_fraction"])
    by_source_ratio: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    recomputed_mismatches: List[str] = []
    for record in raw_records.values():
        source_id = str(record["source_trace_id"])
        problem_id = str(record["problem_id"])
        source_row = source_by_problem.get(problem_id)
        if source_row is None or str(source_row["id"]) != source_id:
            raise ValueError(f"Rewrite is not bound to the expected source row: {record['rewrite_id']}")
        ratio_name = str(record["ratio_name"])
        if ratio_name not in ratio_specs:
            raise ValueError(f"Unexpected rewrite ratio: {ratio_name}")
        source_solution = str(source_row["completion"])
        if record.get("source_file_sha256") != str(paired["source_standard_sha256"]):
            raise ValueError(f"Rewrite source-file hash mismatch: {record['rewrite_id']}")
        if record.get("source_completion_sha256") != canonical_sha256(source_solution):
            raise ValueError(f"Rewrite source-completion hash mismatch: {record['rewrite_id']}")
        gold = _gold_answer(source_row)
        source_tokens = source_token_count(source_row, token_counter)
        required = essential_step_values(source_solution, gold)
        expected_target = target_token_count(source_tokens, float(ratio_specs[ratio_name]["ratio"]))
        expected_minimum = minimum_target_token_count(expected_target, minimum_target_fraction)
        if int(record["target_tokens"]) != expected_target:
            raise ValueError(f"Rewrite target mismatch: {record['rewrite_id']}")
        if int(record["minimum_tokens"]) != expected_minimum:
            raise ValueError(f"Rewrite minimum target mismatch: {record['rewrite_id']}")
        quality = assess_rewrite_candidate(
            str(record["solution"]),
            gold_answer=gold,
            source_tokens=source_tokens,
            minimum_tokens=expected_minimum,
            target_tokens=expected_target,
            required_step_values=required,
            token_counter=token_counter,
            candidate_index=int(record["candidate_index"]),
        )
        if quality != record.get("quality"):
            recomputed_mismatches.append(str(record["rewrite_id"]))
        by_source_ratio[(source_id, ratio_name)].append(quality)
    if recomputed_mismatches:
        raise ValueError(
            f"Stored/recomputed rewrite quality mismatch; examples={recomputed_mismatches[:10]}"
        )

    for source_row in source_rows:
        source_id = str(source_row["id"])
        for ratio_name in ratio_specs:
            candidates = by_source_ratio.get((source_id, ratio_name), [])
            observed_indices = {int(candidate["candidate_index"]) for candidate in candidates}
            if len(candidates) != expected_candidates or observed_indices != set(range(expected_candidates)):
                raise ValueError(
                    f"Incomplete candidate matrix for {source_id}/{ratio_name}: "
                    f"count={len(candidates)} indices={sorted(observed_indices)}"
                )

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "dataset_manifest.json"
    marker_path = output_dir / "DATASETS_COMPLETE"
    if manifest_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite paired SFT data in {output_dir}")

    condition_rows: Dict[str, List[Dict[str, Any]]] = {
        "standard_original": [],
        "direct_short": [],
        "rewrite_80": [],
        "rewrite_65": [],
    }
    selection_audit: List[Dict[str, Any]] = []
    for problem_id in sorted(source_by_problem):
        source_row = source_by_problem[problem_id]
        source_id = str(source_row["id"])
        source_hash = canonical_sha256(str(source_row["completion"]))
        condition_rows["standard_original"].append(
            paired_sft_record(
                source_row,
                condition="standard_original",
                completion=str(source_row["completion"]),
                source_sha256=source_hash,
                selection={"selected_ratio_name": "standard_original", "fallback_level": 0},
            )
        )
        direct_row = short_by_problem[problem_id]
        condition_rows["direct_short"].append(
            paired_sft_record(
                source_row,
                condition="direct_short",
                completion=str(direct_row["completion"]),
                source_sha256=source_hash,
                selection={
                    "selected_ratio_name": "historical_direct_short",
                    "fallback_level": 0,
                    "historical_trace_id": str(direct_row["id"]),
                },
            )
        )
        candidates = {
            ratio_name: by_source_ratio[(source_id, ratio_name)] for ratio_name in ratio_specs
        }
        for condition in ("rewrite_80", "rewrite_65"):
            fallback_ratios = [str(item) for item in ratio_specs[condition]["fallback_ratios"]]
            selected = select_adaptive_rewrite(
                candidates,
                preferred_ratio=condition,
                fallback_ratios=fallback_ratios,
            )
            if selected is None:
                selected = {
                    "requested_ratio_name": condition,
                    "selected_ratio_name": "standard_original",
                    "fallback_level": len(fallback_ratios) + 1,
                    "actual_tokens": source_token_count(source_row, token_counter),
                    "minimum_tokens": minimum_target_token_count(
                        target_token_count(
                            source_token_count(source_row, token_counter),
                            float(ratio_specs[condition]["ratio"]),
                        ),
                        minimum_target_fraction,
                    ),
                    "target_tokens": target_token_count(
                        source_token_count(source_row, token_counter),
                        float(ratio_specs[condition]["ratio"]),
                    ),
                    "within_target": False,
                    "structurally_valid": True,
                    "fallback_reason": "no_structurally_valid_candidate_within_target",
                }
                completion = str(source_row["completion"])
            else:
                completion = str(selected.pop("solution"))
            selection_audit.append(
                {
                    "problem_id": problem_id,
                    "source_trace_id": source_id,
                    "condition": condition,
                    **selected,
                }
            )
            condition_rows[condition].append(
                paired_sft_record(
                    source_row,
                    condition=condition,
                    completion=completion,
                    source_sha256=source_hash,
                    selection=selected,
                )
            )

    run_entries = []
    for condition, rows in condition_rows.items():
        if len(rows) != expected_records or len({source_problem_id(row) for row in rows}) != expected_records:
            raise RuntimeError(f"Incomplete paired dataset for {condition}")
        path = output_dir / condition / f"{condition}.jsonl"
        write_jsonl(path, rows)
        token_counts = [int(token_counter.count(str(row["completion"]))) for row in rows]
        run_entries.append(
            {
                "condition": condition,
                "train_path": str(path),
                "train_sha256": file_sha256(path),
                "n": len(rows),
                "supervised_tokens": sum(token_counts),
                "mean_completion_tokens": statistics.mean(token_counts),
                "median_completion_tokens": statistics.median(token_counts),
                "min_completion_tokens": min(token_counts),
                "max_completion_tokens": max(token_counts),
            }
        )

    audit_path = output_dir / "selection_audit.jsonl"
    write_jsonl(audit_path, selection_audit)
    fallback_counts = Counter(
        (str(row["condition"]), str(row["selected_ratio_name"])) for row in selection_audit
    )
    manifest = {
        "status": "complete",
        "stage": "pilot",
        "evidence_level": "exploratory_single_seed_pilot",
        "config_hash": config_hash,
        "protocol_variant": config["protocol_variant"],
        "problem_count": expected_records,
        "conditions": sorted(condition_rows),
        "inputs": {
            "source_standard": _evidence(source_path),
            "source_direct_short": _evidence(direct_short_path),
            "rewrite_shards": [_evidence(path) for path in paths],
        },
        "selection_audit_path": str(audit_path),
        "selection_audit_sha256": file_sha256(audit_path),
        "fallback_counts": [
            {"condition": key[0], "selected_source": key[1], "count": count}
            for key, count in sorted(fallback_counts.items())
        ],
        "runs": sorted(run_entries, key=lambda row: str(row["condition"])),
    }
    _write_json(manifest_path, manifest)
    marker_path.write_text(
        f"config_hash={config_hash}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        f"problem_count={expected_records}\ncondition_count={len(condition_rows)}\n",
        encoding="utf-8",
    )
    logging.info("paired_datasets_complete conditions=%d output=%s", len(condition_rows), output_dir)


def _index_source(rows: Sequence[Mapping[str, Any]], expected: int, label: str) -> Dict[str, Mapping[str, Any]]:
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {label} rows, got {len(rows)}")
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        problem_id = source_problem_id(row)
        if problem_id in indexed:
            raise ValueError(f"Duplicate {label} problem ID: {problem_id}")
        indexed[problem_id] = row
    return indexed


def _gold_answer(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    problem_metadata = metadata.get("problem_metadata", {}) if isinstance(metadata, Mapping) else {}
    raw = problem_metadata.get("raw_answer") if isinstance(problem_metadata, Mapping) else None
    gold = extract_final_answer(str(raw or "")) or extract_final_answer(str(row["completion"]))
    if gold is None or not verify_answer(extract_final_answer(str(row["completion"])), gold):
        raise ValueError(f"Invalid source gold answer: {source_problem_id(row)}")
    return gold


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_hash(path: Path, expected: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch for {path}: expected={expected} actual={actual}")


def _evidence(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path)}


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
