#!/usr/bin/env python3
"""Record and validate Slurm submission provenance for the Phase-C main matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.ranked_multiteacher import validate_protocol


GENERATION_TEACHERS = ("qwen2p5_1p5b", "qwen2p5_3b", "qwen2p5_14b")
TRAINING_SHARDS = ("0", "1", "2")
ACTIVE_STATES = {"PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "COMPLETED"}
SOURCE_PATHS = (
    "scripts/19_0_submit_ranked_multiteacher_matrix.sh",
    "scripts/19_0_freeze_ranked_multiteacher_protocol.py",
    "scripts/19_1_materialize_ranked_teacher_configs.py",
    "scripts/19_1_materialize_ranked_launcher_plan.py",
    "scripts/19_1_record_ranked_multiteacher_submission.py",
    "scripts/19_3_build_ranked_multiteacher_training_data.py",
    "scripts/19_5_audit_ranked_multiteacher_training.py",
    "scripts/19_6_eval_ranked_multiteacher_matrix.py",
    "scripts/19_7_analyze_ranked_multiteacher_matrix.py",
    "scripts/19_8_audit_ranked_multiteacher_experiment.py",
    "scripts/slurm/19_2_generate_ranked_teacher.sh",
    "scripts/slurm/19_3_build_ranked_multiteacher_data.sh",
    "scripts/slurm/19_4_train_ranked_multiteacher_matrix.sh",
    "scripts/slurm/19_5_audit_ranked_multiteacher_training.sh",
    "scripts/slurm/19_6_eval_analyze_audit_ranked_multiteacher.sh",
    "scripts/16_1_generate_ranked_length_samples.py",
    "scripts/16_2_merge_ranked_length_samples.py",
    "scripts/6_1_train_capacity_length_students.py",
    "scripts/2_1_train_student_sft.py",
    "src/length_budget_distill/ranked_multiteacher.py",
    "src/length_budget_distill/ranked_multiteacher_analysis.py",
    "src/length_budget_distill/ranked_multiteacher_evaluation.py",
    "src/length_budget_distill/ranked_sampling.py",
    "src/length_budget_distill/training.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-overlay", required=True)
    parser.add_argument("--generation-config-manifest", required=True)
    parser.add_argument("--launcher-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--generation-job", action="append", required=True, metavar="TEACHER=JOBID"
    )
    parser.add_argument(
        "--training-job", action="append", required=True, metavar="SHARD=JOBID"
    )
    parser.add_argument("--data-job", required=True)
    parser.add_argument("--training-audit-job", required=True)
    parser.add_argument("--evaluation-job", required=True)
    parser.add_argument(
        "--superseded-generation-job", action="append", default=[], metavar="TEACHER=JOBID"
    )
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    overlay_path = _resolve(args.training_overlay)
    generation_manifest_path = _resolve(args.generation_config_manifest)
    launcher_plan_path = _resolve(args.launcher_plan)
    output_path = _resolve(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite submission provenance: {output_path}")

    config = _read_json(config_path)
    validate_protocol(config, require_frozen=True)
    config_hash = canonical_sha256(config)
    overlay = _read_json(overlay_path)
    if overlay.get("parent_config_sha256") != config_hash:
        raise ValueError("Training overlay is not bound to the frozen protocol.")
    generation_manifest = _read_json(generation_manifest_path)
    if generation_manifest.get("status") != "complete":
        raise ValueError("Generation config materialization is incomplete.")
    if generation_manifest.get("parent_protocol_sha256") != config_hash:
        raise ValueError("Generation config manifest is not bound to the frozen protocol.")
    launcher_plan = _read_json(launcher_plan_path)
    if launcher_plan.get("status") != "complete":
        raise ValueError("Launcher plan is incomplete.")
    if launcher_plan.get("config_hash") != config_hash:
        raise ValueError("Launcher plan is not bound to the frozen protocol.")

    generation_jobs = _parse_bindings(args.generation_job, GENERATION_TEACHERS, "teacher")
    training_jobs = _parse_bindings(args.training_job, TRAINING_SHARDS, "training shard")
    superseded_jobs = _parse_optional_bindings(
        args.superseded_generation_job, GENERATION_TEACHERS, "superseded teacher"
    )
    active_job_ids = (
        list(generation_jobs.values())
        + [str(args.data_job)]
        + list(training_jobs.values())
        + [str(args.training_audit_job), str(args.evaluation_job)]
    )
    if len(active_job_ids) != 9 or len(set(active_job_ids)) != 9:
        raise ValueError("The active main-matrix DAG must contain nine unique Slurm jobs.")
    for job_id in active_job_ids + list(superseded_jobs.values()):
        _validate_job_id(job_id)

    snapshots = {job_id: _scheduler_snapshot(job_id) for job_id in active_job_ids}
    expected_names = {
        **{job_id: "19_2_ranked_teacher_gen" for job_id in generation_jobs.values()},
        str(args.data_job): "19_3_ranked_matrix_data",
        **{job_id: "19_4_ranked_matrix_train" for job_id in training_jobs.values()},
        str(args.training_audit_job): "19_5_ranked_matrix_audit",
        str(args.evaluation_job): "19_6_ranked_matrix_eval",
    }
    for job_id, expected_name in expected_names.items():
        snapshot = snapshots[job_id]
        if snapshot.get("job_name") != expected_name:
            raise ValueError(
                f"Slurm job-name mismatch for {job_id}: "
                f"expected={expected_name} actual={snapshot.get('job_name')}"
            )
        if snapshot.get("state") not in ACTIVE_STATES:
            raise ValueError(f"Active DAG job is not viable: {job_id} {snapshot.get('state')}")

    _require_dependency(snapshots[str(args.data_job)], generation_jobs.values(), "data")
    for shard, job_id in training_jobs.items():
        _require_dependency(snapshots[job_id], [str(args.data_job)], f"training shard {shard}")
    _require_dependency(
        snapshots[str(args.training_audit_job)], training_jobs.values(), "training audit"
    )
    _require_dependency(
        snapshots[str(args.evaluation_job)], [str(args.training_audit_job)], "evaluation"
    )

    superseded_snapshots = {
        teacher: _scheduler_snapshot(job_id, allow_accounting_fallback=True)
        for teacher, job_id in superseded_jobs.items()
    }
    payload = {
        "status": "recorded",
        "evidence_class": "submission_provenance_not_completion_evidence",
        "experiment_name": config["experiment_name"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "config_hash": config_hash,
        "training_overlay_path": str(overlay_path),
        "training_overlay_sha256": file_sha256(overlay_path),
        "generation_config_manifest_path": str(generation_manifest_path),
        "generation_config_manifest_sha256": file_sha256(generation_manifest_path),
        "launcher_plan_path": str(launcher_plan_path),
        "launcher_plan_sha256": file_sha256(launcher_plan_path),
        "launcher_assignment_sha256": launcher_plan.get("assignment_sha256"),
        "active_dag": {
            "generation_jobs": generation_jobs,
            "data_job": str(args.data_job),
            "training_jobs": training_jobs,
            "training_audit_job": str(args.training_audit_job),
            "evaluation_job": str(args.evaluation_job),
            "job_count": len(active_job_ids),
            "scheduler_snapshots": [snapshots[job_id] for job_id in active_job_ids],
        },
        "superseded_generation_jobs": [
            {
                "teacher_name": teacher,
                "job_id": superseded_jobs[teacher],
                "scheduler_snapshot": superseded_snapshots[teacher],
            }
            for teacher in GENERATION_TEACHERS
            if teacher in superseded_jobs
        ],
        "source_hash_scope": (
            "working_tree_at_manifest_recording; this is not a snapshot of Slurm-spooled "
            "batch scripts and is not completion evidence"
        ),
        "source_files": [
            {
                "path": str(PROJECT_ROOT / relative_path),
                "sha256": file_sha256(PROJECT_ROOT / relative_path),
            }
            for relative_path in SOURCE_PATHS
        ],
        "note": str(args.note),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "recorded",
                "output": str(output_path),
                "sha256": file_sha256(output_path),
                "active_job_count": len(active_job_ids),
                "superseded_job_count": len(superseded_jobs),
            },
            indent=2,
        )
    )


def _parse_bindings(values: Iterable[str], expected: Iterable[str], label: str) -> Dict[str, str]:
    result = _parse_optional_bindings(values, expected, label)
    if set(result) != set(expected):
        raise ValueError(f"{label} identities mismatch: expected={sorted(expected)} actual={sorted(result)}")
    return result


def _parse_optional_bindings(
    values: Iterable[str], allowed: Iterable[str], label: str
) -> Dict[str, str]:
    allowed_set = set(allowed)
    result: Dict[str, str] = {}
    for value in values:
        key, separator, job_id = str(value).partition("=")
        if separator != "=" or key not in allowed_set or key in result:
            raise ValueError(f"Invalid or duplicate {label} binding: {value}")
        _validate_job_id(job_id)
        result[key] = job_id
    return result


def _validate_job_id(job_id: str) -> None:
    if not re.fullmatch(r"[1-9][0-9]*", str(job_id)):
        raise ValueError(f"Invalid Slurm job ID: {job_id}")


def _scheduler_snapshot(job_id: str, *, allow_accounting_fallback: bool = False) -> Dict[str, Any]:
    command = ["scontrol", "show", "job", "-o", str(job_id)]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0 and completed.stdout.strip():
        fields = _parse_scontrol(completed.stdout.strip())
        return {
            "snapshot_source": "scontrol",
            "job_id": str(job_id),
            "job_name": fields.get("JobName"),
            "user_id": fields.get("UserId"),
            "state": fields.get("JobState"),
            "reason": fields.get("Reason"),
            "dependency": fields.get("Dependency"),
            "partition": fields.get("Partition"),
            "requested_nodes": fields.get("ReqNodeList"),
            "assigned_nodes": fields.get("NodeList"),
            "submit_time": fields.get("SubmitTime"),
            "start_time": fields.get("StartTime"),
            "end_time": fields.get("EndTime"),
            "time_limit": fields.get("TimeLimit"),
            "command": fields.get("Command"),
            "work_dir": fields.get("WorkDir"),
            "stdout": fields.get("StdOut"),
            "stderr": fields.get("StdErr"),
        }
    if not allow_accounting_fallback:
        raise RuntimeError(
            f"Unable to inspect active Slurm job {job_id}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return _sacct_snapshot(job_id)


def _parse_scontrol(text: str) -> Dict[str, str]:
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9_]*=|$)", text)
    }


def _sacct_snapshot(job_id: str) -> Dict[str, Any]:
    fields = "JobIDRaw,JobName,Partition,State,Submit,Start,End,NodeList,Reason"
    command = ["sacct", "-X", "-j", str(job_id), "--noheader", "--parsable2", "--format", fields]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    for line in completed.stdout.splitlines():
        values = line.split("|")
        if values and values[0] == str(job_id):
            values += [""] * (9 - len(values))
            return {
                "snapshot_source": "sacct",
                "job_id": values[0],
                "job_name": values[1],
                "partition": values[2],
                "state": values[3],
                "submit_time": values[4],
                "start_time": values[5],
                "end_time": values[6],
                "assigned_nodes": values[7],
                "reason": values[8],
            }
    raise RuntimeError(f"Unable to inspect superseded Slurm job {job_id} via sacct.")


def _require_dependency(snapshot: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    observed = set(re.findall(r"afterok:([0-9]+)", str(snapshot.get("dependency", ""))))
    expected_set = {str(job_id) for job_id in expected}
    if observed != expected_set:
        raise ValueError(
            f"{label} dependency mismatch: expected={sorted(expected_set)} actual={sorted(observed)}"
        )


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


if __name__ == "__main__":
    main()
