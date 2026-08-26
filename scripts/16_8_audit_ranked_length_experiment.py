#!/usr/bin/env python3
"""Audit the complete ranked-length generation, training, evaluation, and report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
)
from length_budget_distill.ranked_evaluation import (
    completed_evaluation_evidence,
    protocol_hash,
    validate_evaluation_protocol,
    validate_parent_training,
)


EXPECTED_MODELS = {"base", "relative_short", "relative_medium", "relative_long"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_ranked_sampling_7b_eval_seed17_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    validate_evaluation_protocol(config)
    training_runs = validate_parent_training(config, PROJECT_ROOT)
    eval_hash = protocol_hash(config)
    result_root = _resolve(config["outputs"]["result_root"])
    figure_root = _resolve(config["outputs"]["figure_root"])
    audit_path = result_root / "formal/completion_audit.json"
    final_marker_path = result_root / "FORMAL_COMPLETE"
    if audit_path.exists() or final_marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite final experiment evidence: {result_root}")
    errors: List[str] = []

    generation_evidence = _audit_generation(config, errors)
    eval_manifest_path = result_root / "formal/eval/eval_manifest_formal_shard_00_of_01.json"
    eval_manifest = _read_json(eval_manifest_path)
    _expect(eval_manifest.get("status") == "complete", "Evaluation manifest is incomplete.", errors)
    _expect(eval_manifest.get("config_hash") == eval_hash, "Evaluation config hash mismatch.", errors)
    _expect(int(eval_manifest.get("run_count", -1)) == 4, "Evaluation run count mismatch.", errors)
    source_bindings = {
        "evaluation_source_sha256": PROJECT_ROOT / "scripts/4_1_eval_model.py",
        "launcher_source_sha256": PROJECT_ROOT / "scripts/16_6_eval_ranked_length_students.py",
        "validation_source_sha256": PROJECT_ROOT / "src/length_budget_distill/ranked_evaluation.py",
    }
    for field, path in source_bindings.items():
        _expect(eval_manifest.get(field) == file_sha256(path), f"Evaluation source mismatch: {field}", errors)

    evaluation = dict(config["evaluation"])
    evaluated_models: Dict[str, Dict[str, Any]] = {}
    support: List[str] | None = None
    for run in eval_manifest.get("runs", []):
        model_id = str(run.get("model_id", ""))
        if model_id in evaluated_models:
            errors.append(f"Duplicate evaluation model: {model_id}")
            continue
        _expect(
            run.get("eval_status") in {"complete", "skipped_complete"},
            f"Incomplete evaluation model: {model_id}",
            errors,
        )
        evidence = completed_evaluation_evidence(
            run.get("prediction_path", ""),
            run.get("summary_path", ""),
            expected_n=int(evaluation["limit"]),
            expected_start_index=int(evaluation["start_index"]),
            expected_split=str(evaluation["dataset_split"]),
        )
        if evidence is None:
            errors.append(f"Invalid evaluation artifacts: {model_id}")
            continue
        for field in ("prediction_sha256", "summary_sha256"):
            _expect(run.get(field) == evidence[field], f"Evaluation hash mismatch: {model_id} {field}", errors)
        if support is None:
            support = evidence["problem_ids"]
        else:
            _expect(support == evidence["problem_ids"], f"Evaluation support mismatch: {model_id}", errors)
        evaluated_models[model_id] = run
    _expect(set(evaluated_models) == EXPECTED_MODELS, "Evaluation model identities mismatch.", errors)

    analysis_dir = result_root / "formal/analysis"
    analysis_path = analysis_dir / "ranked_length_analysis.json"
    artifact_manifest_path = analysis_dir / "analysis_artifact_manifest.json"
    analysis_marker_path = analysis_dir / "ANALYSIS_COMPLETE"
    analysis = _read_json(analysis_path)
    artifact_manifest = _read_json(artifact_manifest_path)
    marker = read_key_value_marker(analysis_marker_path)
    _expect(analysis.get("status") == "complete", "Analysis is incomplete.", errors)
    _expect(analysis.get("config_hash") == eval_hash, "Analysis config hash mismatch.", errors)
    _expect(int(analysis.get("run_count", -1)) == 4, "Analysis run count mismatch.", errors)
    _expect(int(analysis.get("problem_count", -1)) == 1269, "Analysis problem count mismatch.", errors)
    _expect(
        {str(row.get("model_id")) for row in analysis.get("metrics", [])} == EXPECTED_MODELS,
        "Analysis model identities mismatch.",
        errors,
    )
    _expect(artifact_manifest.get("status") == "complete", "Analysis artifact manifest incomplete.", errors)
    _expect(artifact_manifest.get("config_hash") == eval_hash, "Artifact config hash mismatch.", errors)
    _expect(
        artifact_manifest.get("analysis_source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/16_7_analyze_ranked_length_evaluation.py"),
        "Analysis script hash mismatch.",
        errors,
    )
    _expect(
        artifact_manifest.get("analysis_library_sha256")
        == file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_evaluation_analysis.py"),
        "Analysis library hash mismatch.",
        errors,
    )
    artifact_count = 0
    for artifact in artifact_manifest.get("artifacts", []):
        path = Path(str(artifact.get("path", "")))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            errors.append(f"Missing analysis artifact: {path}")
            continue
        _expect(artifact.get("sha256") == file_sha256(path), f"Analysis artifact hash mismatch: {path}", errors)
        _expect(int(artifact.get("size_bytes", -1)) == path.stat().st_size, f"Analysis artifact size mismatch: {path}", errors)
        artifact_count += 1
    _expect(artifact_count == 6, f"Analysis artifact count mismatch: {artifact_count}", errors)
    _expect(marker.get("status") == "complete", "Analysis marker status mismatch.", errors)
    _expect(marker.get("config_hash") == eval_hash, "Analysis marker config mismatch.", errors)
    _expect(
        marker.get("artifact_manifest_sha256") == file_sha256(artifact_manifest_path),
        "Analysis marker artifact hash mismatch.",
        errors,
    )
    expected_figure_prefix = figure_root / "formal/ranked_length_accuracy_and_output_length"
    for suffix in (".png", ".pdf"):
        _expect(expected_figure_prefix.with_suffix(suffix).is_file(), f"Missing figure: {suffix}", errors)

    report = {
        "status": "passed" if not errors else "failed",
        "experiment_name": config["experiment_name"],
        "protocol_variant": config["protocol_variant"],
        "evidence_level": config["analysis"]["evidence_level"],
        "scope": config["analysis"]["scope"],
        "config_path": str(config_path),
        "config_hash": eval_hash,
        "config_file_sha256": file_sha256(config_path),
        "counts": {
            "generated_training_problems": generation_evidence.get("problem_count"),
            "trained_adapters": len(training_runs),
            "evaluated_models": len(evaluated_models),
            "predictions": len(evaluated_models) * int(evaluation["limit"]),
            "analysis_artifacts": artifact_count,
        },
        "evidence": {
            "generation": generation_evidence,
            "training_completion_marker": config["parent_training"]["completion_marker_path"],
            "training_completion_marker_sha256": config["parent_training"]["completion_marker_sha256"],
            "evaluation_manifest": str(eval_manifest_path),
            "evaluation_manifest_sha256": file_sha256(eval_manifest_path),
            "analysis": str(analysis_path),
            "analysis_sha256": file_sha256(analysis_path),
            "analysis_artifact_manifest": str(artifact_manifest_path),
            "analysis_artifact_manifest_sha256": file_sha256(artifact_manifest_path),
            "experiment_report": str(analysis_dir / "experiment_report.md"),
            "experiment_report_sha256": file_sha256(analysis_dir / "experiment_report.md"),
        },
        "errors": errors,
    }
    if errors:
        raise SystemExit("Ranked-length completion audit failed: " + " | ".join(errors))
    _write_json(audit_path, report)
    final_marker_path.write_text(
        f"status=passed\nconfig_hash={eval_hash}\n"
        f"completion_audit_sha256={file_sha256(audit_path)}\n"
        f"evaluation_manifest_sha256={file_sha256(eval_manifest_path)}\n"
        f"analysis_artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n"
        "evidence_level=revised_formal_single_seed\n"
        "scope=GSM8K_test_50_1319_only\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "marker": str(final_marker_path)}, indent=2))


def _audit_generation(config: Mapping[str, Any], errors: List[str]) -> Dict[str, Any]:
    input_path = _resolve(config["parent_training"]["input_manifest_path"])
    prepared = _read_json(input_path)
    generation_config_path = _resolve(prepared.get("generation_config_path", ""))
    dataset_manifest_path = _resolve(prepared.get("generation_dataset_manifest", ""))
    marker_path = _resolve(prepared.get("generation_completion_marker", ""))
    for path in (generation_config_path, dataset_manifest_path, marker_path):
        _expect(path.is_file(), f"Missing generation evidence: {path}", errors)
    if not all(path.is_file() for path in (generation_config_path, dataset_manifest_path, marker_path)):
        return {}
    generation_config = _read_json(generation_config_path)
    generation_hash = canonical_sha256(generation_config)
    dataset_manifest = _read_json(dataset_manifest_path)
    marker = read_key_value_marker(marker_path)
    _expect(prepared.get("generation_config_hash") == generation_hash, "Generation config hash mismatch.", errors)
    _expect(
        prepared.get("generation_dataset_manifest_sha256") == file_sha256(dataset_manifest_path),
        "Generation dataset manifest hash mismatch.",
        errors,
    )
    _expect(
        prepared.get("generation_completion_marker_sha256") == file_sha256(marker_path),
        "Generation completion marker hash mismatch.",
        errors,
    )
    _expect(dataset_manifest.get("status") == "complete", "Generation dataset manifest incomplete.", errors)
    _expect(dataset_manifest.get("config_hash") == generation_hash, "Generation dataset config mismatch.", errors)
    _expect(marker.get("config_hash") == generation_hash, "Generation marker config mismatch.", errors)
    _expect(
        marker.get("dataset_manifest_sha256") == file_sha256(dataset_manifest_path),
        "Generation marker manifest mismatch.",
        errors,
    )
    datasets = dataset_manifest.get("datasets", [])
    problem_counts = {int(row.get("record_count", -1)) for row in datasets}
    _expect(len(datasets) == 3 and problem_counts == {881}, "Generation dataset counts mismatch.", errors)
    return {
        "config_path": str(generation_config_path),
        "config_hash": generation_hash,
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "completion_marker": str(marker_path),
        "completion_marker_sha256": file_sha256(marker_path),
        "problem_count": 881 if problem_counts == {881} else None,
    }


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _resolve(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
