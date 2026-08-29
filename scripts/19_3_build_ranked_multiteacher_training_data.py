#!/usr/bin/env python3
"""Audit four teacher datasets and build the 36-run global-support SFT manifest."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    nonempty_line_count,
    read_key_value_marker,
)
from length_budget_distill.ranked_multiteacher import (
    LAUNCHER_ASSIGNMENT_POLICY,
    LAUNCHER_SHARDS,
    RANK_NAMES,
    TEACHER_NAMES,
    ordered_matrix_runs,
    validate_launcher_assignment,
    validate_protocol,
)
from length_budget_distill.ranked_sampling import load_bound_problem_ids
from length_budget_distill.records import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--generation-config-manifest", required=True)
    parser.add_argument(
        "--launcher-plan",
        default=None,
        help="Balanced operational launch plan; defaults beside the frozen protocol.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    materialization_path = _resolve(args.generation_config_manifest)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite main-matrix training data: {output_dir}")
    config = _read_json(config_path)
    validate_protocol(config, require_frozen=True)
    config_hash = canonical_sha256(config)
    launcher_plan_path = (
        _resolve(args.launcher_plan)
        if args.launcher_plan
        else config_path.parent / "launcher_assignment_plan.json"
    )
    launcher_plan = _read_json(launcher_plan_path)
    _require_equal("launcher plan status", launcher_plan.get("status"), "complete")
    _require_equal("launcher plan config hash", launcher_plan.get("config_hash"), config_hash)
    _require_equal(
        "launcher plan config file hash",
        launcher_plan.get("config_file_sha256"),
        file_sha256(config_path),
    )
    launcher_runs = [dict(run) for run in launcher_plan.get("runs", [])]
    validate_launcher_assignment(launcher_runs)
    _require_equal(
        "launcher plan assignment hash",
        launcher_plan.get("assignment_sha256"),
        canonical_sha256(launcher_runs),
    )
    _require_equal(
        "launcher plan registered assignment",
        canonical_sha256(launcher_runs),
        canonical_sha256(ordered_matrix_runs()),
    )
    _require_equal(
        "launcher plan matrix source hash",
        launcher_plan.get("matrix_protocol_source_sha256"),
        file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher.py"),
    )
    _require_equal(
        "launcher plan selector source hash",
        launcher_plan.get("launcher_selection_source_sha256"),
        file_sha256(PROJECT_ROOT / "src/length_budget_distill/factorial.py"),
    )
    materialization = _read_json(materialization_path)
    _require_equal("generation config manifest status", materialization.get("status"), "complete")
    _require_equal(
        "generation config parent file hash",
        materialization.get("parent_config_sha256"),
        file_sha256(config_path),
    )
    _require_equal(
        "generation config parent protocol hash",
        materialization.get("parent_protocol_sha256"),
        config_hash,
    )
    derived_by_teacher = {
        str(row["teacher_name"]): dict(row) for row in materialization.get("teachers", [])
    }
    if set(derived_by_teacher) != set(TEACHER_NAMES) - {"qwen2p5_7b"}:
        raise ValueError("Materialized generation teacher identities mismatch.")

    result_root = _resolve(config["outputs"]["result_root"])
    sources: Dict[str, Dict[str, Any]] = {}
    for teacher_name in TEACHER_NAMES:
        if teacher_name == "qwen2p5_7b":
            sealed = dict(config["sealed_7b_generation"])
            source_config_path = _resolve(sealed["config_path"])
            source_manifest_path = _resolve(sealed["dataset_manifest_path"])
            marker_path = _resolve(sealed["completion_marker_path"])
            source_config_hash = str(sealed["canonical_config_sha256"])
            _require_equal("sealed 7B config file hash", file_sha256(source_config_path), sealed["config_file_sha256"])
            _require_equal("sealed 7B dataset manifest hash", file_sha256(source_manifest_path), sealed["dataset_manifest_sha256"])
            _require_equal("sealed 7B marker hash", file_sha256(marker_path), sealed["completion_marker_sha256"])
        else:
            entry = derived_by_teacher[teacher_name]
            source_config_path = _resolve(entry["config_path"])
            _require_equal(
                f"{teacher_name} generation config file hash",
                file_sha256(source_config_path),
                entry["config_file_sha256"],
            )
            source_config = _read_json(source_config_path)
            source_config_hash = canonical_sha256(source_config)
            _require_equal(
                f"{teacher_name} generation canonical hash",
                source_config_hash,
                entry["canonical_config_sha256"],
            )
            source_manifest_path = result_root / f"formal/teachers/{teacher_name}/datasets/dataset_manifest.json"
            marker_path = result_root / f"formal/teachers/{teacher_name}/datasets/GENERATION_COMPLETE"
        source_manifest = _validate_source_manifest(
            teacher_name,
            source_manifest_path,
            marker_path,
            source_config_hash,
        )
        sources[teacher_name] = {
            "config_path": str(source_config_path),
            "config_sha256": source_config_hash,
            "config_file_sha256": file_sha256(source_config_path),
            "dataset_manifest_path": str(source_manifest_path),
            "dataset_manifest_sha256": file_sha256(source_manifest_path),
            "completion_marker_path": str(marker_path),
            "completion_marker_sha256": file_sha256(marker_path),
            "eligible_problem_count": int(source_manifest["eligible_problem_count"]),
            "problem_ids": [str(value) for value in source_manifest["problem_ids"]],
            "datasets": source_manifest["datasets"],
        }

    source_supports = [set(source["problem_ids"]) for source in sources.values()]
    global_support = set.intersection(*source_supports)
    minimum = int(config["matrix"]["minimum_global_common_problems"])
    if len(global_support) < minimum:
        raise ValueError(
            f"Global teacher/rank support is below the registered gate: "
            f"actual={len(global_support)} minimum={minimum}"
        )
    cohort_order = load_bound_problem_ids(config)
    global_problem_ids = [problem_id for problem_id in cohort_order if problem_id in global_support]
    if len(global_problem_ids) != len(global_support):
        raise ValueError("Global support contains identities outside the registered cohort.")

    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_by_cell: Dict[tuple[str, str], Dict[str, Any]] = {}
    published_datasets: List[Dict[str, Any]] = []
    for teacher_name in TEACHER_NAMES:
        source_by_budget = {
            str(row["budget_name"]): dict(row) for row in sources[teacher_name]["datasets"]
        }
        if set(source_by_budget) != set(RANK_NAMES):
            raise ValueError(f"Rank coverage mismatch for teacher {teacher_name}.")
        for rank_name in RANK_NAMES:
            source_entry = source_by_budget[rank_name]
            source_path = _resolve(source_entry["train_path"])
            rows = list(read_jsonl(source_path))
            by_problem = _unique_sft_rows(rows, teacher_name, rank_name)
            if set(by_problem) != set(sources[teacher_name]["problem_ids"]):
                raise ValueError(f"SFT/source support mismatch for {teacher_name} {rank_name}.")
            selected = [by_problem[problem_id] for problem_id in global_problem_ids]
            destination = output_dir / "sft" / f"{teacher_name}__{rank_name}.jsonl"
            write_jsonl(destination, selected)
            token_counts = [int(row["metadata"]["solution_token_count"]) for row in selected]
            dataset = {
                "generator_name": teacher_name,
                "budget_name": rank_name,
                "train_path": str(destination),
                "train_sha256": file_sha256(destination),
                "record_count": len(selected),
                "supervised_tokens": sum(token_counts),
                "mean_solution_tokens": statistics.fmean(token_counts),
                "median_solution_tokens": statistics.median(token_counts),
                "source_train_path": str(source_path),
                "source_train_sha256": file_sha256(source_path),
            }
            published_datasets.append(dataset)
            dataset_by_cell[(teacher_name, rank_name)] = dataset

    runs = []
    for spec in launcher_runs:
        dataset = dataset_by_cell[(spec["generator_name"], spec["budget_name"])]
        runs.append(
            {
                **spec,
                "train_path": dataset["train_path"],
                "train_sha256": dataset["train_sha256"],
                "n": dataset["record_count"],
                "supervised_tokens": dataset["supervised_tokens"],
            }
        )
    if len(runs) != 36:
        raise ValueError(f"Main-matrix run count mismatch: {len(runs)}")
    manifest_path = output_dir / "dataset_manifest.json"
    manifest = {
        "status": "complete",
        "experiment_name": config["experiment_name"],
        "protocol_variant": config["protocol_variant"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "builder_source_sha256": file_sha256(Path(__file__).resolve()),
        "matrix_protocol_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher.py"
        ),
        "launcher_selection_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/factorial.py"
        ),
        "generation_config_manifest": str(materialization_path),
        "generation_config_manifest_sha256": file_sha256(materialization_path),
        "launcher_plan_path": str(launcher_plan_path),
        "launcher_plan_sha256": file_sha256(launcher_plan_path),
        "launcher_assignment_sha256": canonical_sha256(launcher_runs),
        "support_policy": config["matrix"]["training_support"],
        "source_cohort_count": len(cohort_order),
        "global_common_problem_count": len(global_problem_ids),
        "global_common_problem_ids": global_problem_ids,
        "teacher_retention": [
            {
                "teacher_name": teacher_name,
                "eligible_problem_count": sources[teacher_name]["eligible_problem_count"],
                "global_problem_count": len(global_problem_ids),
                "global_retention_from_source": len(global_problem_ids)
                / sources[teacher_name]["eligible_problem_count"],
            }
            for teacher_name in TEACHER_NAMES
        ],
        "generation_sources": [
            {key: value for key, value in sources[teacher_name].items() if key not in {"problem_ids", "datasets"}}
            for teacher_name in TEACHER_NAMES
        ],
        "dataset_count": len(published_datasets),
        "datasets": published_datasets,
        "run_count": len(runs),
        "launcher_shards": LAUNCHER_SHARDS,
        "launcher_assignment_policy": LAUNCHER_ASSIGNMENT_POLICY,
        "runs": runs,
    }
    _write_json(manifest_path, manifest)
    marker_path = output_dir / "DATA_COMPLETE"
    marker_path.write_text(
        f"status=passed\nconfig_hash={config_hash}\n"
        f"dataset_manifest_sha256={file_sha256(manifest_path)}\n"
        f"launcher_plan_sha256={file_sha256(launcher_plan_path)}\n"
        f"global_common_problem_count={len(global_problem_ids)}\n"
        f"launcher_shards={LAUNCHER_SHARDS}\n"
        f"launcher_assignment_policy={LAUNCHER_ASSIGNMENT_POLICY}\n"
        "teacher_count=4\ndataset_count=12\nrun_count=36\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "global_common_problem_count": len(global_problem_ids),
                "run_count": len(runs),
            },
            indent=2,
        )
    )


def _validate_source_manifest(
    teacher_name: str,
    manifest_path: Path,
    marker_path: Path,
    config_hash: str,
) -> Dict[str, Any]:
    if not manifest_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"Missing generation evidence for {teacher_name}.")
    manifest = _read_json(manifest_path)
    marker = read_key_value_marker(marker_path)
    _require_equal(f"{teacher_name} manifest status", manifest.get("status"), "complete")
    _require_equal(f"{teacher_name} manifest config hash", manifest.get("config_hash"), config_hash)
    _require_equal(f"{teacher_name} marker config hash", marker.get("config_hash"), config_hash)
    _require_equal(
        f"{teacher_name} marker manifest hash",
        marker.get("dataset_manifest_sha256"),
        file_sha256(manifest_path),
    )
    if len(manifest.get("problem_ids", [])) != int(manifest.get("eligible_problem_count", -1)):
        raise ValueError(f"Eligible problem count mismatch for {teacher_name}.")
    if len(set(manifest["problem_ids"])) != len(manifest["problem_ids"]):
        raise ValueError(f"Duplicate eligible problem IDs for {teacher_name}.")
    for dataset in manifest.get("datasets", []):
        path = _resolve(dataset["train_path"])
        _require_equal(f"{teacher_name} dataset hash", file_sha256(path), dataset["train_sha256"])
        _require_equal(
            f"{teacher_name} dataset count",
            nonempty_line_count(path),
            int(dataset["record_count"]),
        )
    return manifest


def _unique_sft_rows(
    rows: List[Dict[str, Any]], teacher_name: str, rank_name: str
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        metadata = dict(row.get("metadata", {}))
        problem_id = str(metadata.get("problem_id", ""))
        if not problem_id or problem_id in result:
            raise ValueError(f"Missing or duplicate SFT problem ID: {teacher_name} {rank_name}")
        _require_equal("SFT generator name", metadata.get("generator_name"), teacher_name)
        _require_equal("SFT rank name", metadata.get("budget_name"), rank_name)
        _require_equal("SFT correctness", metadata.get("is_correct"), True)
        result[problem_id] = dict(row)
    return result


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


if __name__ == "__main__":
    main()
