#!/usr/bin/env python3
"""Audit and seal the ranked-length seed-17/42/73 replication evidence."""

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
from length_budget_distill.factorial import canonical_sha256, file_sha256, read_key_value_marker
from length_budget_distill.ranked_evaluation import completed_evaluation_evidence
from length_budget_distill.ranked_multiseed_evaluation import (
    model_id,
    validate_all_parent_trainings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    training_runs = validate_all_parent_trainings(config, PROJECT_ROOT)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    result_root = _resolve(config["outputs"]["result_root"])
    figure_root = _resolve(config["outputs"]["figure_root"])
    audit_path = result_root / "formal/completion_audit.json"
    final_marker_path = result_root / "MULTISEED_COMPLETE"
    if audit_path.exists() or final_marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite multi-seed evidence: {result_root}")
    errors: List[str] = []

    expected_training = {str(run["model_id"]): run for run in training_runs}
    expected_models = set(expected_training) | {"base"}
    evaluation = dict(config["evaluation"])
    eval_manifest_path = result_root / "formal/eval/eval_manifest_formal_shard_00_of_01.json"
    eval_manifest = _read_json(eval_manifest_path)
    _expect(eval_manifest.get("status") == "complete", "Evaluation manifest is incomplete.", errors)
    _expect(eval_manifest.get("config_hash") == config_hash, "Evaluation config hash mismatch.", errors)
    _expect(
        eval_manifest.get("config_file_sha256") == file_sha256(config_path),
        "Evaluation config file hash mismatch.",
        errors,
    )
    _expect(
        int(eval_manifest.get("run_count", -1)) == int(evaluation["expected_run_count"]),
        "Evaluation run count mismatch.",
        errors,
    )
    source_bindings = {
        "evaluation_source_sha256": PROJECT_ROOT / "scripts/4_1_eval_model.py",
        "launcher_source_sha256": PROJECT_ROOT / "scripts/18_5_eval_ranked_multiseed_students.py",
        "validation_source_sha256": PROJECT_ROOT / "src/length_budget_distill/ranked_multiseed_evaluation.py",
    }
    for field, path in source_bindings.items():
        _expect(eval_manifest.get(field) == file_sha256(path), f"Evaluation source mismatch: {field}", errors)

    evaluated_models: Dict[str, Dict[str, Any]] = {}
    support: List[str] | None = None
    for run in eval_manifest.get("runs", []):
        current_id = str(run.get("model_id", ""))
        if current_id in evaluated_models:
            errors.append(f"Duplicate evaluation model: {current_id}")
            continue
        _expect(current_id in expected_models, f"Unexpected evaluation model: {current_id}", errors)
        _expect(
            run.get("eval_status") in {"complete", "skipped_complete"},
            f"Incomplete evaluation model: {current_id}",
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
            errors.append(f"Invalid evaluation artifacts: {current_id}")
            continue
        for field in ("prediction_sha256", "summary_sha256"):
            _expect(run.get(field) == evidence[field], f"Evaluation hash mismatch: {current_id} {field}", errors)
        if support is None:
            support = evidence["problem_ids"]
        else:
            _expect(support == evidence["problem_ids"], f"Evaluation support mismatch: {current_id}", errors)
        if current_id != "base" and current_id in expected_training:
            parent = expected_training[current_id]
            for eval_field, parent_field in (
                ("adapter_path", "adapter_path"),
                ("adapter_config_sha256", "adapter_config_sha256"),
                ("adapter_model_sha256", "adapter_model_sha256"),
                ("training_data_sha256", "train_sha256"),
                ("training_examples", "n"),
                ("supervised_tokens", "supervised_tokens"),
                ("seed", "seed"),
                ("budget_name", "budget_name"),
            ):
                _expect(
                    run.get(eval_field) == parent.get(parent_field),
                    f"Evaluation/training binding mismatch: {current_id} {eval_field}",
                    errors,
                )
        evaluated_models[current_id] = dict(run)
    _expect(set(evaluated_models) == expected_models, "Evaluation model identities mismatch.", errors)

    analysis_dir = result_root / "formal/analysis"
    analysis_path = analysis_dir / "ranked_multiseed_analysis.json"
    artifact_manifest_path = analysis_dir / "analysis_artifact_manifest.json"
    analysis_marker_path = analysis_dir / "ANALYSIS_COMPLETE"
    analysis = _read_json(analysis_path)
    artifact_manifest = _read_json(artifact_manifest_path)
    marker = read_key_value_marker(analysis_marker_path)
    _expect(analysis.get("status") == "complete", "Analysis is incomplete.", errors)
    _expect(analysis.get("config_hash") == config_hash, "Analysis config hash mismatch.", errors)
    _expect(
        int(analysis.get("validated_adapter_count", -1)) == 9,
        "Analysis adapter count mismatch.",
        errors,
    )
    _expect(
        int(analysis.get("evaluated_model_count", -1)) == 10,
        "Analysis model count mismatch.",
        errors,
    )
    _expect(int(analysis.get("training_seed_count", -1)) == 3, "Analysis seed count mismatch.", errors)
    _expect(int(analysis.get("problem_count", -1)) == 1269, "Analysis problem count mismatch.", errors)
    _expect(
        {str(row.get("model_id")) for row in analysis.get("metrics", [])} == expected_models,
        "Analysis model identities mismatch.",
        errors,
    )
    _expect(
        len(analysis.get("aggregate_metrics", [])) == 3,
        "Aggregate metric count mismatch.",
        errors,
    )
    _expect(
        len(analysis.get("per_seed_contrasts", [])) == 9,
        "Per-seed contrast count mismatch.",
        errors,
    )
    _expect(
        len(analysis.get("aggregate_contrasts", [])) == 3,
        "Aggregate contrast count mismatch.",
        errors,
    )
    for row in analysis.get("aggregate_contrasts", []):
        _expect(int(row.get("seed_count", -1)) == 3, "Aggregate contrast seed count mismatch.", errors)
        _expect(int(row.get("problem_count", -1)) == 1269, "Aggregate contrast problem count mismatch.", errors)
        _expect(
            row.get("resampling_units") == ["training_seed", "paired_problem"],
            "Aggregate contrast resampling-unit mismatch.",
            errors,
        )

    _expect(artifact_manifest.get("status") == "complete", "Analysis artifact manifest incomplete.", errors)
    _expect(artifact_manifest.get("config_hash") == config_hash, "Artifact config hash mismatch.", errors)
    _expect(
        artifact_manifest.get("analysis_source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/18_6_analyze_ranked_multiseed.py"),
        "Analysis script hash mismatch.",
        errors,
    )
    _expect(
        artifact_manifest.get("analysis_library_sha256")
        == file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_multiseed_analysis.py"),
        "Analysis library hash mismatch.",
        errors,
    )
    _expect(
        artifact_manifest.get("shared_analysis_library_sha256")
        == file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_evaluation_analysis.py"),
        "Shared analysis library hash mismatch.",
        errors,
    )
    artifact_count = 0
    for artifact in artifact_manifest.get("artifacts", []):
        path = _resolve(artifact.get("path", ""))
        if not path.is_file():
            errors.append(f"Missing analysis artifact: {path}")
            continue
        _expect(artifact.get("sha256") == file_sha256(path), f"Analysis artifact hash mismatch: {path}", errors)
        _expect(
            int(artifact.get("size_bytes", -1)) == path.stat().st_size,
            f"Analysis artifact size mismatch: {path}",
            errors,
        )
        artifact_count += 1
    _expect(artifact_count == 7, f"Analysis artifact count mismatch: {artifact_count}", errors)
    _expect(int(artifact_manifest.get("artifact_count", -1)) == 7, "Artifact manifest count mismatch.", errors)
    _expect(marker.get("status") == "complete", "Analysis marker status mismatch.", errors)
    _expect(marker.get("config_hash") == config_hash, "Analysis marker config mismatch.", errors)
    _expect(
        marker.get("artifact_manifest_sha256") == file_sha256(artifact_manifest_path),
        "Analysis marker artifact hash mismatch.",
        errors,
    )
    figure_prefix = figure_root / "formal/ranked_multiseed_accuracy_and_output_length"
    for suffix in (".png", ".pdf"):
        _expect(figure_prefix.with_suffix(suffix).is_file(), f"Missing figure: {suffix}", errors)

    report = {
        "status": "passed" if not errors else "failed",
        "experiment_name": config["experiment_name"],
        "protocol_variant": config["protocol_variant"],
        "evidence_level": config["analysis"]["evidence_level"],
        "scope": config["analysis"]["scope"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "counts": {
            "training_parents": len(config["parent_trainings"]),
            "trained_adapters": len(training_runs),
            "training_seeds": 3,
            "evaluated_models": len(evaluated_models),
            "evaluation_questions": int(evaluation["limit"]),
            "predictions": len(evaluated_models) * int(evaluation["limit"]),
            "analysis_artifacts": artifact_count,
        },
        "evidence": {
            "training_parents": config["parent_trainings"],
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
        raise SystemExit("Ranked multi-seed completion audit failed: " + " | ".join(errors))
    _write_json(audit_path, report)
    final_marker_path.write_text(
        f"status=passed\nconfig_hash={config_hash}\n"
        f"completion_audit_sha256={file_sha256(audit_path)}\n"
        f"evaluation_manifest_sha256={file_sha256(eval_manifest_path)}\n"
        f"analysis_artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n"
        "evidence_level=comparative_multiseed_replication\n"
        "training_seeds=17,42,73\ntrained_adapter_count=9\n"
        "evaluation_question_count=1269\n"
        "scope=GSM8K_test_50_1319_previously_observed_only\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "marker": str(final_marker_path)}, indent=2))


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
