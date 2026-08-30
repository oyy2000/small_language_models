#!/usr/bin/env python3
"""Audit and seal the complete 36-adapter ranked multiteacher main matrix."""

from __future__ import annotations

import argparse
import json
import re
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
from length_budget_distill.ranked_multiteacher import (
    LAUNCHER_ASSIGNMENT_POLICY,
    RANK_NAMES,
    TEACHER_NAMES,
    TRAINING_SEEDS,
    ordered_matrix_runs,
    validate_launcher_assignment,
)
from length_budget_distill.ranked_multiteacher_evaluation import (
    matrix_model_id,
    validated_matrix_training_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    training_runs = validated_matrix_training_runs(config, PROJECT_ROOT)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    result_root = _resolve(config["outputs"]["result_root"])
    figure_root = _resolve(config["outputs"]["figure_root"])
    checkpoint_root = _resolve(config["outputs"]["checkpoint_root"])
    audit_path = result_root / "formal/completion_audit.json"
    marker_path = result_root / "MATRIX_COMPLETE"
    if audit_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite main-matrix evidence: {result_root}")
    errors: List[str] = []
    storage = {
        "result_root": _beegfs_storage_evidence("result", result_root, errors),
        "checkpoint_root": _beegfs_storage_evidence("checkpoint", checkpoint_root, errors),
    }
    submission_path = config_path.parent / "submission_manifest.json"
    submission = _audit_submission_provenance(
        submission_path,
        config_hash,
        config_path.parent / "frozen_sft_overlay.json",
        config_path.parent / "generation_configs/generation_config_manifest.json",
        errors,
    )
    submission_annotation_path = config_path.parent / "submission_manifest_annotation.json"
    submission_annotation = _audit_submission_annotation(
        submission_annotation_path, submission_path, result_root, submission, errors
    )
    expected_training = {str(run["model_id"]): run for run in training_runs}
    expected_models = set(expected_training) | {"base"}
    evaluation = dict(config["evaluation"])
    eval_path = result_root / "formal/eval/eval_manifest_formal_shard_00_of_01.json"
    eval_manifest = _read_json(eval_path)
    _expect(eval_manifest.get("status") == "complete", "Evaluation manifest incomplete.", errors)
    _expect(eval_manifest.get("config_hash") == config_hash, "Evaluation config hash mismatch.", errors)
    _expect(int(eval_manifest.get("run_count", -1)) == 37, "Evaluation run count mismatch.", errors)
    for field, path in {
        "evaluation_source_sha256": PROJECT_ROOT / "scripts/4_1_eval_model.py",
        "launcher_source_sha256": PROJECT_ROOT / "scripts/19_6_eval_ranked_multiteacher_matrix.py",
        "validation_source_sha256": PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher_evaluation.py",
    }.items():
        _expect(eval_manifest.get(field) == file_sha256(path), f"Evaluation source mismatch: {field}", errors)
    evaluated: Dict[str, Dict[str, Any]] = {}
    support: List[str] | None = None
    for run in eval_manifest.get("runs", []):
        current_id = str(run.get("model_id", ""))
        if current_id in evaluated:
            errors.append(f"Duplicate evaluation model: {current_id}")
            continue
        _expect(run.get("eval_status") in {"complete", "skipped_complete"}, f"Incomplete evaluation: {current_id}", errors)
        evidence = completed_evaluation_evidence(
            run.get("prediction_path", ""), run.get("summary_path", ""),
            expected_n=int(evaluation["limit"]), expected_start_index=int(evaluation["start_index"]),
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
                ("adapter_path", "adapter_path"), ("adapter_config_sha256", "adapter_config_sha256"),
                ("adapter_model_sha256", "adapter_model_sha256"), ("training_data_sha256", "train_sha256"),
                ("training_examples", "n"), ("supervised_tokens", "supervised_tokens"),
            ):
                _expect(run.get(eval_field) == parent.get(parent_field), f"Evaluation/training binding mismatch: {current_id} {eval_field}", errors)
        evaluated[current_id] = dict(run)
    _expect(set(evaluated) == expected_models, "Evaluation identities mismatch.", errors)

    analysis_dir = result_root / "formal/analysis"
    analysis_path = analysis_dir / "ranked_multiteacher_analysis.json"
    artifact_path = analysis_dir / "analysis_artifact_manifest.json"
    analysis_marker_path = analysis_dir / "ANALYSIS_COMPLETE"
    analysis = _read_json(analysis_path)
    artifacts = _read_json(artifact_path)
    analysis_marker = read_key_value_marker(analysis_marker_path)
    _expect(analysis.get("status") == "complete", "Analysis incomplete.", errors)
    _expect(analysis.get("config_hash") == config_hash, "Analysis config mismatch.", errors)
    _expect(int(analysis.get("validated_adapter_count", -1)) == 36, "Analysis adapter count mismatch.", errors)
    _expect(int(analysis.get("evaluated_model_count", -1)) == 37, "Analysis model count mismatch.", errors)
    _expect(int(analysis.get("training_seed_count", -1)) == 3, "Analysis seed count mismatch.", errors)
    _expect(int(analysis.get("problem_count", -1)) == 1269, "Analysis problem count mismatch.", errors)
    operational_assignment = dict(analysis.get("operational_training_assignment", {}))
    active_launcher_plan = config_path.parent / "launcher_assignment_plan.json"
    _expect(
        operational_assignment.get("policy") == LAUNCHER_ASSIGNMENT_POLICY,
        "Analysis launcher assignment policy mismatch.",
        errors,
    )
    _expect(
        _resolve(operational_assignment.get("launcher_plan", "")) == active_launcher_plan,
        "Analysis launcher plan path mismatch.",
        errors,
    )
    _expect(
        operational_assignment.get("launcher_plan_sha256") == file_sha256(active_launcher_plan),
        "Analysis launcher plan hash mismatch.",
        errors,
    )
    _expect(
        operational_assignment.get("wave_barrier_policy")
        == "declared_launcher_wave_barrier_v1",
        "Analysis wave-barrier policy mismatch.",
        errors,
    )
    _expect(len(analysis.get("aggregate_metrics", [])) == 12, "Aggregate cell count mismatch.", errors)
    _expect(len(analysis.get("within_teacher_contrasts", [])) == 12, "Within-teacher contrast count mismatch.", errors)
    _expect(len(analysis.get("teacher_by_rank_interactions", [])) == 6, "Interaction count mismatch.", errors)
    for row in analysis.get("within_teacher_contrasts", []) + analysis.get("teacher_by_rank_interactions", []):
        _expect(int(row.get("seed_count", -1)) == 3, "Contrast seed count mismatch.", errors)
        _expect(int(row.get("problem_count", -1)) == 1269, "Contrast problem count mismatch.", errors)
        _expect(row.get("resampling_units") == ["training_seed", "paired_problem"], "Contrast resampling units mismatch.", errors)
    _expect(artifacts.get("status") == "complete", "Artifact manifest incomplete.", errors)
    _expect(artifacts.get("config_hash") == config_hash, "Artifact config mismatch.", errors)
    _expect(artifacts.get("analysis_source_sha256") == file_sha256(PROJECT_ROOT / "scripts/19_7_analyze_ranked_multiteacher_matrix.py"), "Analysis source hash mismatch.", errors)
    _expect(artifacts.get("bootstrap_library_sha256") == file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_multiseed_analysis.py"), "Bootstrap source hash mismatch.", errors)
    _expect(
        artifacts.get("matrix_analysis_library_sha256")
        == file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher_analysis.py"),
        "Main-matrix analysis-library hash mismatch.",
        errors,
    )
    artifact_count = 0
    for artifact in artifacts.get("artifacts", []):
        path = _resolve(artifact.get("path", ""))
        if not path.is_file():
            errors.append(f"Missing analysis artifact: {path}")
            continue
        _expect(artifact.get("sha256") == file_sha256(path), f"Artifact hash mismatch: {path}", errors)
        _expect(int(artifact.get("size_bytes", -1)) == path.stat().st_size, f"Artifact size mismatch: {path}", errors)
        artifact_count += 1
    _expect(artifact_count == 8, f"Analysis artifact count mismatch: {artifact_count}", errors)
    _expect(analysis_marker.get("status") == "complete", "Analysis marker status mismatch.", errors)
    _expect(analysis_marker.get("config_hash") == config_hash, "Analysis marker config mismatch.", errors)
    _expect(analysis_marker.get("artifact_manifest_sha256") == file_sha256(artifact_path), "Analysis marker artifact mismatch.", errors)
    figure_prefix = figure_root / "formal/teacher_capacity_by_rank_accuracy_and_output_length"
    for suffix in (".png", ".pdf"):
        _expect(figure_prefix.with_suffix(suffix).is_file(), f"Missing figure: {suffix}", errors)

    report = {
        "status": "passed" if not errors else "failed",
        "experiment_name": config["experiment_name"],
        "protocol_variant": config["protocol_variant"],
        "scope": config["analysis"]["scope"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "audit_source_sha256": file_sha256(Path(__file__).resolve()),
        "storage": storage,
        "counts": {
            "teachers": 4, "ranks": 3, "training_seeds": 3,
            "trained_adapters": len(training_runs), "evaluated_models": len(evaluated),
            "evaluation_questions": int(evaluation["limit"]),
            "predictions": len(evaluated) * int(evaluation["limit"]),
            "analysis_artifacts": artifact_count,
        },
        "evidence": {
            "phase_a": config["phase_a_evidence"],
            "submission_manifest": str(submission_path),
            "submission_manifest_sha256": file_sha256(submission_path),
            "submission_summary": submission,
            "submission_annotation": str(submission_annotation_path),
            "submission_annotation_sha256": file_sha256(submission_annotation_path),
            "submission_annotation_summary": submission_annotation,
            "training_audit": str(result_root / "formal/training/audit/training_audit.json"),
            "training_audit_sha256": file_sha256(result_root / "formal/training/audit/training_audit.json"),
            "evaluation_manifest": str(eval_path), "evaluation_manifest_sha256": file_sha256(eval_path),
            "analysis": str(analysis_path), "analysis_sha256": file_sha256(analysis_path),
            "analysis_artifact_manifest": str(artifact_path),
            "analysis_artifact_manifest_sha256": file_sha256(artifact_path),
            "experiment_report": str(analysis_dir / "experiment_report.md"),
            "experiment_report_sha256": file_sha256(analysis_dir / "experiment_report.md"),
        },
        "errors": errors,
    }
    if errors:
        raise SystemExit("Main-matrix completion audit failed: " + " | ".join(errors))
    _write_json(audit_path, report)
    marker_path.write_text(
        f"status=passed\nconfig_hash={config_hash}\ncompletion_audit_sha256={file_sha256(audit_path)}\n"
        f"evaluation_manifest_sha256={file_sha256(eval_path)}\n"
        f"analysis_artifact_manifest_sha256={file_sha256(artifact_path)}\n"
        f"submission_manifest_sha256={file_sha256(submission_path)}\n"
        f"submission_annotation_sha256={file_sha256(submission_annotation_path)}\n"
        f"audit_source_sha256={file_sha256(Path(__file__).resolve())}\n"
        "teacher_count=4\nrank_count=3\nseed_count=3\ntrained_adapter_count=36\n"
        "evaluation_question_count=1269\nscope=GSM8K_test_50_1319_previously_observed_only\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "marker": str(marker_path)}, indent=2))


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _beegfs_storage_evidence(
    label: str, stable_path: Path, errors: List[str]
) -> Dict[str, Any]:
    resolved = stable_path.resolve()
    symlink_anchor = _nearest_symlink_anchor(stable_path)
    _expect(
        symlink_anchor is not None,
        f"{label.capitalize()} root does not use a symlinked stable project path.",
        errors,
    )
    _expect(resolved.is_dir(), f"{label.capitalize()} BeeGFS target is missing.", errors)
    _expect(
        str(resolved).startswith("/mnt/beegfs/"),
        f"{label.capitalize()} root does not resolve to BeeGFS: {resolved}",
        errors,
    )
    return {
        "stable_path": str(stable_path),
        "resolved_path": str(resolved),
        "is_symlink": stable_path.is_symlink(),
        "symlink_anchor": str(symlink_anchor) if symlink_anchor is not None else None,
        "uses_symlinked_stable_path": symlink_anchor is not None,
        "is_beegfs": str(resolved).startswith("/mnt/beegfs/"),
    }


def _nearest_symlink_anchor(path: Path) -> Path | None:
    """Return the closest symlink that makes ``path`` a stable project path."""

    current = path
    while current != PROJECT_ROOT.parent:
        if current.is_symlink():
            return current
        if current == PROJECT_ROOT or current.parent == current:
            break
        current = current.parent
    return None


def _audit_submission_provenance(
    path: Path,
    config_hash: str,
    overlay_path: Path,
    generation_manifest_path: Path,
    errors: List[str],
) -> Dict[str, Any]:
    payload = _read_json(path)
    _expect(payload.get("status") == "recorded", "Submission manifest status mismatch.", errors)
    _expect(
        payload.get("evidence_class") == "submission_provenance_not_completion_evidence",
        "Submission evidence class mismatch.",
        errors,
    )
    _expect(payload.get("config_hash") == config_hash, "Submission config hash mismatch.", errors)
    _expect(
        payload.get("training_overlay_sha256") == file_sha256(overlay_path),
        "Submission training-overlay hash mismatch.",
        errors,
    )
    _expect(
        payload.get("generation_config_manifest_sha256") == file_sha256(generation_manifest_path),
        "Submission generation-manifest hash mismatch.",
        errors,
    )
    dag = dict(payload.get("active_dag", {}))
    generation_jobs = dict(dag.get("generation_jobs", {}))
    training_jobs = dict(dag.get("training_jobs", {}))
    _expect(
        set(generation_jobs) == {"qwen2p5_1p5b", "qwen2p5_3b", "qwen2p5_14b"},
        "Submission generation-job identities mismatch.",
        errors,
    )
    _expect(set(training_jobs) == {"0", "1", "2"}, "Submission training shards mismatch.", errors)
    job_ids = (
        list(generation_jobs.values())
        + [dag.get("data_job")]
        + list(training_jobs.values())
        + [dag.get("training_audit_job"), dag.get("evaluation_job")]
    )
    _expect(int(dag.get("job_count", -1)) == 9, "Submission job count mismatch.", errors)
    _expect(len(job_ids) == len(set(job_ids)) == 9, "Submission job IDs are not unique.", errors)
    _expect(
        all(re.fullmatch(r"[1-9][0-9]*", str(job_id or "")) for job_id in job_ids),
        "Submission contains invalid Slurm job IDs.",
        errors,
    )
    snapshots = {
        str(row.get("job_id", "")): dict(row)
        for row in dag.get("scheduler_snapshots", [])
        if isinstance(row, dict)
    }
    _expect(set(snapshots) == set(job_ids), "Submission scheduler snapshots mismatch.", errors)
    dependencies = {
        str(job_id): set(re.findall(r"afterok:([0-9]+)", str(snapshots.get(str(job_id), {}).get("dependency", ""))))
        for job_id in job_ids
    }
    _expect(
        dependencies.get(str(dag.get("data_job"))) == set(generation_jobs.values()),
        "Recorded data dependency mismatch.",
        errors,
    )
    for job_id in training_jobs.values():
        _expect(
            dependencies.get(str(job_id)) == {str(dag.get("data_job"))},
            f"Recorded training dependency mismatch: {job_id}",
            errors,
        )
    _expect(
        dependencies.get(str(dag.get("training_audit_job"))) == set(training_jobs.values()),
        "Recorded training-audit dependency mismatch.",
        errors,
    )
    _expect(
        dependencies.get(str(dag.get("evaluation_job"))) == {str(dag.get("training_audit_job"))},
        "Recorded evaluation dependency mismatch.",
        errors,
    )
    source_files = payload.get("source_files", [])
    source_paths = [str(row.get("path", "")) for row in source_files if isinstance(row, dict)]
    _expect(len(source_paths) >= 20, "Submission source inventory is incomplete.", errors)
    _expect(len(source_paths) == len(set(source_paths)), "Submission source inventory is duplicated.", errors)
    _expect(
        all(
            re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
            for row in source_files
            if isinstance(row, dict)
        ),
        "Submission source inventory contains invalid hashes.",
        errors,
    )
    return {
        "job_count": len(job_ids),
        "generation_jobs": generation_jobs,
        "data_job": dag.get("data_job"),
        "training_jobs": training_jobs,
        "training_audit_job": dag.get("training_audit_job"),
        "evaluation_job": dag.get("evaluation_job"),
        "superseded_generation_job_count": len(payload.get("superseded_generation_jobs", [])),
        "source_file_count": len(source_paths),
    }


def _audit_submission_annotation(
    path: Path,
    submission_path: Path,
    result_root: Path,
    submission_summary: Mapping[str, Any],
    errors: List[str],
) -> Dict[str, Any]:
    payload = _read_json(path)
    _expect(payload.get("status") == "recorded", "Submission annotation status mismatch.", errors)
    _expect(
        payload.get("artifact_type") == "submission_provenance_annotation",
        "Submission annotation type mismatch.",
        errors,
    )
    _expect(
        _resolve(payload.get("parent_submission_manifest", "")) == submission_path,
        "Submission annotation parent path mismatch.",
        errors,
    )
    _expect(
        payload.get("parent_submission_manifest_sha256") == file_sha256(submission_path),
        "Submission annotation parent hash mismatch.",
        errors,
    )
    _expect(
        payload.get("source_inventory_scope")
        == "working_tree_at_manifest_recording; not a snapshot of Slurm-spooled batch scripts and not completion evidence",
        "Submission source-hash scope is not explicit.",
        errors,
    )
    migration = dict(payload.get("storage_migration", {}))
    _expect(migration.get("status") == "passed", "Result migration status mismatch.", errors)
    _expect(
        _resolve(migration.get("stable_result_path", "")) == result_root,
        "Result migration stable path mismatch.",
        errors,
    )
    _expect(
        Path(str(migration.get("resolved_result_path", ""))).resolve() == result_root.resolve(),
        "Result migration resolved path mismatch.",
        errors,
    )
    backup = Path(str(migration.get("recoverable_nfs_backup", "")))
    _expect(backup.is_dir(), "Recoverable NFS migration backup is missing.", errors)
    expected_paused = {
        str(submission_summary.get("generation_jobs", {}).get("qwen2p5_1p5b", "")),
        str(submission_summary.get("generation_jobs", {}).get("qwen2p5_3b", "")),
    }
    _expect(
        set(str(value) for value in migration.get("paused_jobs", [])) == expected_paused,
        "Result migration paused-job identities mismatch.",
        errors,
    )
    _expect(migration.get("pause_signal") == "SIGSTOP", "Migration pause signal mismatch.", errors)
    _expect(migration.get("resume_signal") == "SIGCONT", "Migration resume signal mismatch.", errors)
    remediation = dict(payload.get("training_assignment_remediation", {}))
    _expect(
        remediation.get("status") == "passed_pre_training",
        "Training assignment remediation status mismatch.",
        errors,
    )
    _expect(
        remediation.get("protocol_effect")
        == "No teacher, rank, seed, dataset, student, or SFT hyperparameter changed; only the operational mapping of the registered 36 cells to three launcher shards changed.",
        "Training assignment protocol-scope statement mismatch.",
        errors,
    )
    launcher_plan_path = _resolve(remediation.get("launcher_plan", ""))
    launcher_plan = _read_json(launcher_plan_path)
    _expect(
        remediation.get("launcher_plan_sha256") == file_sha256(launcher_plan_path),
        "Training launcher-plan hash mismatch.",
        errors,
    )
    _expect(
        launcher_plan.get("assignment_sha256") == remediation.get("assignment_sha256"),
        "Training launcher assignment hash mismatch.",
        errors,
    )
    _expect(
        remediation.get("assignment_policy") == LAUNCHER_ASSIGNMENT_POLICY,
        "Training launcher assignment policy mismatch.",
        errors,
    )
    _expect(
        remediation.get("runtime_wave_barrier")
        == "declared_launcher_wave_barrier_v1 waits for all three runs in a node-local wave before launching the next wave.",
        "Training runtime wave-barrier statement mismatch.",
        errors,
    )
    superseded_plan_path = _resolve(remediation.get("superseded_node_only_plan", ""))
    _expect(
        remediation.get("superseded_node_only_plan_sha256")
        == file_sha256(superseded_plan_path),
        "Superseded node-only launcher-plan hash mismatch.",
        errors,
    )
    _expect(
        superseded_plan_path != launcher_plan_path,
        "Active and superseded launcher plans share a path.",
        errors,
    )
    superseded_pre_barrier_path = _resolve(
        remediation.get("superseded_pre_wave_barrier_source_plan", "")
    )
    _expect(
        remediation.get("superseded_pre_wave_barrier_source_plan_sha256")
        == file_sha256(superseded_pre_barrier_path),
        "Superseded pre-wave-barrier source-plan hash mismatch.",
        errors,
    )
    _expect(
        superseded_pre_barrier_path not in {launcher_plan_path, superseded_plan_path},
        "Launcher plan provenance paths are not unique.",
        errors,
    )
    launcher_runs = [dict(run) for run in launcher_plan.get("runs", [])]
    validate_launcher_assignment(launcher_runs)
    _expect(
        canonical_sha256(launcher_runs) == canonical_sha256(ordered_matrix_runs()),
        "Training launcher plan no longer matches the registered 36 runs.",
        errors,
    )
    return {
        "source_inventory_scope": payload.get("source_inventory_scope"),
        "storage_migration_status": migration.get("status"),
        "resolved_result_path": migration.get("resolved_result_path"),
        "recoverable_nfs_backup": str(backup),
        "paused_jobs": migration.get("paused_jobs", []),
        "training_assignment_status": remediation.get("status"),
        "launcher_plan": str(launcher_plan_path),
        "launcher_plan_sha256": file_sha256(launcher_plan_path),
        "launcher_assignment_sha256": remediation.get("assignment_sha256"),
        "superseded_node_only_plan": str(superseded_plan_path),
        "superseded_node_only_plan_sha256": remediation.get(
            "superseded_node_only_plan_sha256"
        ),
        "superseded_pre_wave_barrier_source_plan": str(superseded_pre_barrier_path),
        "superseded_pre_wave_barrier_source_plan_sha256": remediation.get(
            "superseded_pre_wave_barrier_source_plan_sha256"
        ),
    }


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
