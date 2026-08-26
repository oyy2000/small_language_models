#!/usr/bin/env python3
"""Build six 7B MATH-mixed equal-example/equal-token pilot datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    canonical_sha256,
    deterministic_equal_token_subset,
    expected_conditions,
    file_sha256,
)
from length_budget_distill.records import read_jsonl, trace_from_dict, write_jsonl
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.math_mix import stable_mixed_sft_order, tag_sft_record


Condition = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_math_mix_pilot_v1.json")
    parser.add_argument("--math-selected-traces", required=True)
    parser.add_argument("--math-selection-audit", required=True)
    parser.add_argument(
        "--gsm-dataset-manifest",
        default=(
            "results/capacity_length_factorial_seed17_v1/formal/"
            "sft_data/dataset_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config_hash = canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})
    seed_values = [int(seed) for seed in config["balancing"]["training_seeds"]]
    if seed_values != [17]:
        raise ValueError(f"Pilot requires exactly training seed 17, got {seed_values}")
    seed = seed_values[0]
    conditions = expected_conditions(config)
    if len(conditions) != 3 or {condition[0] for condition in conditions} != {"qwen2p5_7b"}:
        raise ValueError(f"Pilot must contain the three qwen2p5_7b length conditions, got {conditions}")

    output_dir = Path(args.output_dir)
    marker_path = output_dir / "DATASETS_COMPLETE"
    manifest_path = output_dir / "dataset_manifest.json"
    if marker_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed/existing pilot data in {output_dir}")

    math_audit = _read_json(Path(args.math_selection_audit))
    if math_audit.get("status") != "passed" or math_audit.get("config_hash") != config_hash:
        raise ValueError("MATH selection audit is not passed or is bound to another config")
    minimum_common = int(config["balancing"]["pilot_min_common_problems"])
    common_count = int(math_audit.get("common_problem_count", -1))
    if common_count < minimum_common:
        raise ValueError(
            f"MATH common-problem gate failed: actual={common_count} required={minimum_common}"
        )
    common_path = Path(str(math_audit["common_problem_ids_path"]))
    common_payload = _read_json(common_path)
    common_ids = [str(item) for item in common_payload["problem_ids"]]
    if len(common_ids) != common_count or len(common_ids) != len(set(common_ids)):
        raise ValueError("MATH common-problem IDs are duplicate or inconsistent with the audit")
    common_set = set(common_ids)

    math_traces = [trace_from_dict(row) for row in read_jsonl(Path(args.math_selected_traces))]
    math_by_condition: Dict[Condition, List[Any]] = {condition: [] for condition in conditions}
    for trace in math_traces:
        condition = (str(trace.generator_name), trace.budget_name)
        if condition in math_by_condition and trace.problem_id in common_set:
            math_by_condition[condition].append(trace)
    for condition, traces in math_by_condition.items():
        traces.sort(key=lambda trace: trace.problem_id)
        if len(traces) != common_count or {trace.problem_id for trace in traces} != common_set:
            raise ValueError(f"Incomplete MATH common matrix for {condition}: n={len(traces)}")

    math_token_totals = {
        condition: sum(int(trace.solution_token_count) for trace in traces)
        for condition, traces in math_by_condition.items()
    }
    math_equal_token_target = min(math_token_totals.values())
    max_gap = int(config["balancing"].get("max_equal_token_gap", 512))

    gsm_manifest_path = Path(args.gsm_dataset_manifest)
    gsm_manifest = _read_json(gsm_manifest_path)
    if gsm_manifest.get("status") != "complete":
        raise ValueError(f"GSM8K dataset manifest is incomplete: {gsm_manifest_path}")
    if gsm_manifest.get("training_seeds") != [17] or int(gsm_manifest.get("common_problem_count", -1)) != 881:
        raise ValueError("Expected the completed 881-problem seed-17 GSM8K dataset manifest")

    gsm_runs = _index_gsm_runs(gsm_manifest.get("runs", []))
    run_entries: List[Dict[str, Any]] = []
    for generator_name, budget_name in conditions:
        condition = (generator_name, budget_name)
        math_equal_example_records = [
            tag_sft_record(
                trace_to_sft_record(trace), source="hendrycks_math", id_prefix="math::"
            )
            for trace in math_by_condition[condition]
        ]
        math_subset_seed = int(
            canonical_sha256([config_hash, "math_equal_token", generator_name, budget_name, seed])[:8],
            16,
        )
        math_subset, math_subset_tokens = deterministic_equal_token_subset(
            math_by_condition[condition],
            target_tokens=math_equal_token_target,
            seed=math_subset_seed,
        )
        math_gap = math_equal_token_target - math_subset_tokens
        if math_gap < 0 or math_gap > max_gap:
            raise ValueError(
                f"MATH equal-token gap exceeds tolerance for {condition}: "
                f"target={math_equal_token_target} actual={math_subset_tokens} gap={math_gap}"
            )
        math_equal_token_records = [
            tag_sft_record(
                trace_to_sft_record(trace), source="hendrycks_math", id_prefix="math::"
            )
            for trace in math_subset
        ]

        for mode, math_records, math_tokens in (
            ("equal_example", math_equal_example_records, math_token_totals[condition]),
            ("equal_token", math_equal_token_records, math_subset_tokens),
        ):
            gsm_run = gsm_runs[(mode, generator_name, budget_name, seed)]
            gsm_path = _resolve_project_path(str(gsm_run["train_path"]))
            if file_sha256(gsm_path) != gsm_run.get("train_sha256"):
                raise ValueError(f"GSM8K SFT data hash mismatch: {gsm_path}")
            gsm_records = [
                tag_sft_record(row, source="gsm8k", id_prefix="gsm8k::")
                for row in read_jsonl(gsm_path)
            ]
            if len(gsm_records) != int(gsm_run["n"]):
                raise ValueError(f"GSM8K SFT row-count mismatch: {gsm_path}")
            combined = stable_mixed_sft_order(
                [*gsm_records, *math_records],
                config_hash=config_hash,
                mode=mode,
                budget_name=budget_name,
                seed=seed,
            )
            output_path = output_dir / mode / f"{generator_name}__{budget_name}__seed_{seed}.jsonl"
            write_jsonl(output_path, combined)
            total_tokens = int(gsm_run["supervised_tokens"]) + int(math_tokens)
            run_entries.append(
                {
                    "run_name": f"math_mix__{mode}__{generator_name}__{budget_name}__seed_{seed}",
                    "training_variant": "math_mix",
                    "mode": mode,
                    "generator_name": generator_name,
                    "budget_name": budget_name,
                    "seed": seed,
                    "train_path": str(output_path),
                    "train_sha256": file_sha256(output_path),
                    "n": len(combined),
                    "supervised_tokens": total_tokens,
                    "source_counts": {
                        "gsm8k": len(gsm_records),
                        "hendrycks_math": len(math_records),
                    },
                    "source_supervised_tokens": {
                        "gsm8k": int(gsm_run["supervised_tokens"]),
                        "hendrycks_math": int(math_tokens),
                    },
                    "math_equal_token_target": math_equal_token_target if mode == "equal_token" else None,
                    "math_equal_token_gap": math_gap if mode == "equal_token" else None,
                }
            )

    if len(run_entries) != 6:
        raise RuntimeError(f"Pilot run-manifest size mismatch: expected=6 actual={len(run_entries)}")
    equal_example_supports = {
        int(run["source_counts"]["hendrycks_math"])
        for run in run_entries
        if run["mode"] == "equal_example"
    }
    if equal_example_supports != {common_count}:
        raise RuntimeError(f"Equal-example MATH support is inconsistent: {equal_example_supports}")

    manifest = {
        "status": "complete",
        "stage": "pilot",
        "evidence_level": "exploratory_single_seed_pilot",
        "config_hash": config_hash,
        "protocol_variant": "qwen2p5_7b_math_mix_single_seed_pilot",
        "training_seeds": [seed],
        "expected_run_count": 6,
        "common_problem_count": common_count,
        "math_equal_token_target": math_equal_token_target,
        "math_condition_token_totals": [
            {
                "generator_name": condition[0],
                "budget_name": condition[1],
                "supervised_tokens": math_token_totals[condition],
            }
            for condition in conditions
        ],
        "inputs": {
            "math_selected_traces": _file_evidence(Path(args.math_selected_traces)),
            "math_selection_audit": _file_evidence(Path(args.math_selection_audit)),
            "math_common_problem_ids": _file_evidence(common_path),
            "gsm_dataset_manifest": _file_evidence(gsm_manifest_path),
        },
        "runs": run_entries,
    }
    _write_json(manifest_path, manifest)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"config_hash={config_hash}\nrun_count=6\nmath_common_problem_count={common_count}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )
    logging.info(
        "math_mix_datasets_complete common=%d runs=%d output=%s",
        common_count,
        len(run_entries),
        output_dir,
    )


def _index_gsm_runs(runs: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str, str, int], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, str, str, int], Mapping[str, Any]] = {}
    for run in runs:
        if run.get("generator_name") != "qwen2p5_7b" or run.get("mode") not in {
            "equal_example",
            "equal_token",
        }:
            continue
        key = (
            str(run["mode"]),
            str(run["generator_name"]),
            str(run["budget_name"]),
            int(run["seed"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate GSM8K run identity: {key}")
        indexed[key] = run
    expected = {
        (mode, "qwen2p5_7b", budget, 17)
        for mode in ("equal_example", "equal_token")
        for budget in ("short_128", "medium_256", "long_512")
    }
    if set(indexed) != expected:
        raise ValueError(f"Missing/unexpected GSM8K 7B seed-17 runs: observed={sorted(indexed)}")
    return indexed


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_evidence(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
