"""Evidence audit for the capacity-by-CoT-length experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .factorial import (
    canonical_sha256,
    expected_conditions,
    file_sha256,
    nonempty_line_count,
    read_key_value_marker,
    validated_adapter_evidence,
)


def audit_capacity_length_completion(
    config: Mapping[str, Any],
    stage: str,
    stage_root: str | Path,
    project_root: str | Path,
) -> Dict[str, Any]:
    """Validate the immutable evidence chain from raw traces through analysis."""

    if stage not in {"smoke", "formal"}:
        raise ValueError(f"Unsupported stage: {stage}")
    root = Path(stage_root)
    project = Path(project_root)
    normalized_config = dict(config)
    normalized_config.pop("_config_path", None)
    config_hash = canonical_sha256(normalized_config)
    errors: list[str] = []
    counts: Dict[str, int] = {}

    selection_path = root / "selected" / "selection_audit.json"
    selection = _load_json(selection_path, errors)
    expected_problems = 200 if stage == "smoke" else int(normalized_config["dataset"]["max_examples"])
    expected_conditions_count = len(expected_conditions(normalized_config))
    expected_candidates = (
        expected_problems
        * expected_conditions_count
        * int(normalized_config["generation"]["num_candidates"])
    )
    _expect(selection.get("status") == "passed", "selection audit is not passed", errors)
    _expect(selection.get("stage") == stage, "selection stage mismatch", errors)
    _expect(selection.get("config_hash") == config_hash, "selection config hash mismatch", errors)
    _expect(
        int(selection.get("actual_total_candidates", -1)) == expected_candidates,
        "raw candidate cardinality mismatch",
        errors,
    )
    _expect(
        int(selection.get("expected_total_candidates", -1)) == expected_candidates,
        "registered candidate cardinality mismatch",
        errors,
    )
    _expect(
        len(selection.get("conditions", [])) == expected_conditions_count,
        "selection condition matrix mismatch",
        errors,
    )
    minimum_common = int(normalized_config["balancing"][f"{stage}_min_common_problems"])
    _expect(
        int(selection.get("common_problem_count", -1)) >= minimum_common,
        "common-cohort gate failed",
        errors,
    )
    _verify_registered_file(selection, "merged_path", "merged_sha256", project, errors)
    _verify_registered_file(selection, "selected_path", "selected_sha256", project, errors)
    _verify_registered_file(selection, "common_problem_ids_path", "common_problem_ids_sha256", project, errors)
    raw_records = 0
    for shard in selection.get("input_shards", []):
        path = _resolve(project, shard.get("path"))
        if not path.is_file():
            errors.append(f"missing raw shard: {path}")
            continue
        if shard.get("sha256") != file_sha256(path):
            errors.append(f"raw shard hash mismatch: {path}")
        line_count = nonempty_line_count(path)
        if int(shard.get("record_count", -1)) != line_count:
            errors.append(f"raw shard line-count mismatch: {path}")
        raw_records += line_count
    _expect(raw_records == expected_candidates, "raw shard total does not match registered cardinality", errors)
    counts["raw_candidates"] = raw_records
    counts["common_problems"] = int(selection.get("common_problem_count", 0))

    dataset_path = root / "sft_data" / "dataset_manifest.json"
    dataset = _load_json(dataset_path, errors)
    dataset_runs = {str(run.get("run_name")): run for run in dataset.get("runs", [])}
    configured_seeds = [int(seed) for seed in normalized_config["balancing"]["training_seeds"]]
    active_seeds = [int(seed) for seed in dataset.get("training_seeds", [])]
    _expect(bool(active_seeds), "dataset manifest has no active training seeds", errors)
    _expect(len(active_seeds) == len(set(active_seeds)), "dataset training seeds are not unique", errors)
    _expect(set(active_seeds) <= set(configured_seeds), "dataset training seeds exceed parent protocol", errors)
    expected_training_runs = expected_conditions_count * len(active_seeds) * 2 + 2 * len(active_seeds)
    _expect(dataset.get("status") == "complete", "dataset manifest is not complete", errors)
    _expect(dataset.get("stage") == stage, "dataset stage mismatch", errors)
    _expect(dataset.get("config_hash") == config_hash, "dataset config hash mismatch", errors)
    _verify_run_config(dataset, config_hash, active_seeds, project, errors)
    _expect(len(dataset_runs) == expected_training_runs, "dataset run cardinality mismatch", errors)
    _expect(
        int(dataset.get("expected_run_count", -1)) == expected_training_runs,
        "dataset expected-run count mismatch",
        errors,
    )
    for run_name, run in dataset_runs.items():
        path = _resolve(project, run.get("train_path"))
        if not path.is_file() or run.get("train_sha256") != file_sha256(path):
            errors.append(f"SFT data evidence mismatch: {run_name}")
    _verify_marker(root / "sft_data" / "DATASETS_COMPLETE", dataset_path, config_hash, errors)
    counts["training_runs_expected"] = expected_training_runs
    counts["training_seeds"] = len(active_seeds)

    training_runs: Dict[str, Dict[str, Any]] = {}
    training_manifest_paths = sorted((root / "training").glob("training_manifest_shard_*_of_*.json"))
    _expect(len(training_manifest_paths) == 4, "expected exactly four training shard manifests", errors)
    for path in training_manifest_paths:
        manifest = _load_json(path, errors)
        _expect(manifest.get("status") == "complete", f"incomplete training manifest: {path}", errors)
        _expect(manifest.get("config_hash") == config_hash, f"training config mismatch: {path}", errors)
        for run in manifest.get("runs", []):
            run_name = str(run.get("run_name"))
            if run_name in training_runs:
                errors.append(f"duplicate training run: {run_name}")
                continue
            training_runs[run_name] = run
            if run.get("status") not in {"complete", "skipped_complete"}:
                errors.append(f"incomplete training run: {run_name}")
                continue
            evidence = validated_adapter_evidence(_resolve(project, run.get("output_dir")))
            if evidence is None:
                errors.append(f"invalid adapter evidence: {run_name}")
                continue
            for field in (
                "train_sha256",
                "run_config_sha256",
                "training_source_sha256",
                "launcher_source_sha256",
                "adapter_config_sha256",
                "adapter_model_sha256",
            ):
                if run.get(field) != evidence.get(field):
                    errors.append(f"adapter marker mismatch: {run_name} field={field}")
    _expect(set(training_runs) == set(dataset_runs), "training run identities do not match dataset manifest", errors)
    counts["trained_adapters"] = len(training_runs)

    evaluation_runs: Dict[str, Dict[str, Any]] = {}
    eval_manifest_paths = sorted((root / "eval").glob(f"eval_manifest_{stage}_shard_*_of_*.json"))
    _expect(len(eval_manifest_paths) == 4, "expected exactly four evaluation shard manifests", errors)
    expected_eval_examples = int(normalized_config["evaluation"][f"{stage}_limit"])
    for path in eval_manifest_paths:
        manifest = _load_json(path, errors)
        _expect(manifest.get("status") == "complete", f"incomplete evaluation manifest: {path}", errors)
        _expect(manifest.get("stage") == stage, f"evaluation stage mismatch: {path}", errors)
        _expect(manifest.get("config_hash") == config_hash, f"evaluation config mismatch: {path}", errors)
        for run in manifest.get("runs", []):
            run_name = str(run.get("run_name"))
            if run_name in evaluation_runs:
                errors.append(f"duplicate evaluation run: {run_name}")
                continue
            evaluation_runs[run_name] = run
            if run.get("eval_status") not in {"complete", "skipped_complete"}:
                errors.append(f"incomplete evaluation run: {run_name}")
                continue
            prediction_path = _resolve(project, run.get("prediction_path"))
            summary_path = _resolve(project, run.get("summary_path"))
            if not prediction_path.is_file() or run.get("prediction_sha256") != file_sha256(prediction_path):
                errors.append(f"prediction evidence mismatch: {run_name}")
            elif nonempty_line_count(prediction_path) != expected_eval_examples:
                errors.append(f"prediction cardinality mismatch: {run_name}")
            if not summary_path.is_file() or run.get("summary_sha256") != file_sha256(summary_path):
                errors.append(f"evaluation summary evidence mismatch: {run_name}")
    expected_eval_runs = expected_training_runs + 1
    expected_eval_names = set(dataset_runs) | {"base_qwen2p5_1p5b_instruct"}
    _expect(set(evaluation_runs) == expected_eval_names, "evaluation run identities are incomplete", errors)
    _expect(len(evaluation_runs) == expected_eval_runs, "evaluation run cardinality mismatch", errors)
    counts["evaluation_runs"] = len(evaluation_runs)
    counts["predictions"] = len(evaluation_runs) * expected_eval_examples

    analysis_dir = root / "analysis"
    analysis_path = analysis_dir / "capacity_length_analysis.json"
    analysis = _load_json(analysis_path, errors)
    _expect(analysis.get("status") == "complete", "analysis is not complete", errors)
    _expect(analysis.get("stage") == stage, "analysis stage mismatch", errors)
    _expect(analysis.get("config_hash") == config_hash, "analysis config hash mismatch", errors)
    _expect(int(analysis.get("run_count", -1)) == expected_eval_runs, "analysis run count mismatch", errors)
    conclusion = analysis.get("conclusion", {})
    if stage == "smoke":
        expected_level = "pipeline_smoke_only"
    elif active_seeds == configured_seeds:
        expected_level = "registered_formal"
    else:
        expected_level = "revised_formal_single_seed" if len(active_seeds) == 1 else "revised_formal_seed_subset"
    _expect(conclusion.get("evidence_level") == expected_level, "analysis evidence-level mismatch", errors)
    if stage == "smoke":
        _expect(
            conclusion.get("classification") == "smoke_only_no_scientific_conclusion",
            "smoke analysis improperly registers a scientific conclusion",
            errors,
        )
    artifact_manifest_path = analysis_dir / "analysis_artifact_manifest.json"
    artifact_manifest = _load_json(artifact_manifest_path, errors)
    _expect(artifact_manifest.get("status") == "complete", "analysis artifact manifest is incomplete", errors)
    _expect(artifact_manifest.get("config_hash") == config_hash, "analysis artifact config mismatch", errors)
    for artifact in artifact_manifest.get("artifacts", []):
        path = _resolve(project, artifact.get("path"))
        if not path.is_file() or artifact.get("sha256") != file_sha256(path):
            errors.append(f"analysis artifact evidence mismatch: {path}")
    _verify_marker(analysis_dir / "ANALYSIS_COMPLETE", artifact_manifest_path, config_hash, errors)

    return {
        "status": "passed" if not errors else "failed",
        "stage": stage,
        "config_hash": config_hash,
        "counts": counts,
        "errors": errors,
        "evidence": {
            "selection_audit": str(selection_path),
            "dataset_manifest": str(dataset_path),
            "training_manifests": [str(path) for path in training_manifest_paths],
            "evaluation_manifests": [str(path) for path in eval_manifest_paths],
            "analysis": str(analysis_path),
            "analysis_artifact_manifest": str(artifact_manifest_path),
        },
    }


def _resolve(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _load_json(path: Path, errors: list[str]) -> Dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing JSON artifact: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON artifact: {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"JSON artifact must be an object: {path}")
        return {}
    return payload


def _expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _verify_registered_file(
    payload: Mapping[str, Any],
    path_field: str,
    hash_field: str,
    project_root: Path,
    errors: list[str],
) -> None:
    path = _resolve(project_root, payload.get(path_field))
    if not path.is_file() or payload.get(hash_field) != file_sha256(path):
        errors.append(f"registered file evidence mismatch: {path_field}={path}")


def _verify_marker(marker_path: Path, manifest_path: Path, config_hash: str, errors: list[str]) -> None:
    if not marker_path.is_file():
        errors.append(f"missing completion marker: {marker_path}")
        return
    try:
        marker = read_key_value_marker(marker_path)
    except (OSError, ValueError) as exc:
        errors.append(f"invalid completion marker: {marker_path}: {exc}")
        return
    if marker.get("config_hash") != config_hash:
        errors.append(f"completion marker config mismatch: {marker_path}")
    if not manifest_path.is_file():
        errors.append(f"completion marker target is missing: {manifest_path}")
        return
    manifest_field = "manifest_sha256" if "manifest_sha256" in marker else "artifact_manifest_sha256"
    if marker.get(manifest_field) != file_sha256(manifest_path):
        errors.append(f"completion marker manifest hash mismatch: {marker_path}")


def _verify_run_config(
    dataset: Mapping[str, Any],
    config_hash: str,
    active_seeds: list[int],
    project_root: Path,
    errors: list[str],
) -> None:
    evidence = dataset.get("run_config")
    if evidence is None:
        return
    if not isinstance(evidence, Mapping):
        errors.append("dataset run_config evidence is invalid")
        return
    path = _resolve(project_root, evidence.get("path"))
    if not path.is_file() or evidence.get("sha256") != file_sha256(path):
        errors.append(f"run-config evidence mismatch: {path}")
        return
    payload = _load_json(path, errors)
    if payload.get("parent_config_sha256") != config_hash:
        errors.append("run-config parent hash mismatch")
    if [int(seed) for seed in payload.get("training_seeds", [])] != active_seeds:
        errors.append("run-config training seeds mismatch")
