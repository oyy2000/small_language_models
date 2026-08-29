#!/usr/bin/env python3
"""Audit and seal all 36 adapters in the teacher-capacity by rank matrix."""

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

from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    nonempty_line_count,
    read_key_value_marker,
    validated_adapter_evidence,
)
from length_budget_distill.ranked_multiteacher import (
    LAUNCHER_ASSIGNMENT_POLICY,
    LAUNCHER_SHARDS,
    ordered_matrix_runs,
    validate_launcher_assignment,
    validate_protocol,
)
from length_budget_distill.ranked_multiseed import manifest_field_equal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--launcher-shards", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    dataset_path = _resolve(args.dataset_manifest)
    work_dir = _resolve(args.work_dir)
    output_dir = _resolve(args.output_dir)
    audit_path = output_dir / "training_audit.json"
    marker_path = output_dir / "TRAINING_COMPLETE"
    if audit_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite training audit: {output_dir}")
    config = _read_json(config_path)
    validate_protocol(config, require_frozen=True)
    config_hash = canonical_sha256(config)
    dataset = _read_json(dataset_path)
    errors: List[str] = []
    _expect(dataset.get("status") == "complete", "Dataset manifest is incomplete.", errors)
    _expect(dataset.get("config_hash") == config_hash, "Dataset config hash mismatch.", errors)
    _expect(
        dataset.get("builder_source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/19_3_build_ranked_multiteacher_training_data.py"),
        "Dataset builder source hash mismatch.",
        errors,
    )
    _expect(
        dataset.get("matrix_protocol_source_sha256")
        == file_sha256(PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher.py"),
        "Dataset matrix-protocol source hash mismatch.",
        errors,
    )
    _expect(
        dataset.get("launcher_selection_source_sha256")
        == file_sha256(PROJECT_ROOT / "src/length_budget_distill/factorial.py"),
        "Dataset launcher-selection source hash mismatch.",
        errors,
    )
    _expect(int(dataset.get("run_count", -1)) == 36, "Dataset run count mismatch.", errors)
    _expect(int(dataset.get("dataset_count", -1)) == 12, "Dataset cell count mismatch.", errors)
    _expect(
        int(dataset.get("launcher_shards", -1)) == LAUNCHER_SHARDS == args.launcher_shards,
        "Dataset launcher topology mismatch.",
        errors,
    )
    _expect(
        dataset.get("launcher_assignment_policy") == LAUNCHER_ASSIGNMENT_POLICY,
        "Dataset launcher assignment policy mismatch.",
        errors,
    )
    launcher_plan_path = _resolve(dataset.get("launcher_plan_path", ""))
    launcher_plan = _read_json(launcher_plan_path)
    _expect(launcher_plan.get("status") == "complete", "Launcher plan is incomplete.", errors)
    _expect(
        dataset.get("launcher_plan_sha256") == file_sha256(launcher_plan_path),
        "Dataset launcher-plan hash mismatch.",
        errors,
    )
    launcher_runs = [dict(run) for run in launcher_plan.get("runs", [])]
    validate_launcher_assignment(launcher_runs)
    _expect(
        dataset.get("launcher_assignment_sha256") == canonical_sha256(launcher_runs),
        "Dataset launcher assignment hash mismatch.",
        errors,
    )
    global_n = int(dataset.get("global_common_problem_count", -1))
    _expect(
        global_n >= int(config["matrix"]["minimum_global_common_problems"]),
        "Global problem support is below the registered gate.",
        errors,
    )
    data_marker_path = dataset_path.parent / "DATA_COMPLETE"
    data_marker = read_key_value_marker(data_marker_path)
    _expect(data_marker.get("status") == "passed", "Data marker status mismatch.", errors)
    _expect(data_marker.get("config_hash") == config_hash, "Data marker config mismatch.", errors)
    _expect(
        data_marker.get("dataset_manifest_sha256") == file_sha256(dataset_path),
        "Data marker manifest hash mismatch.",
        errors,
    )
    _expect(
        data_marker.get("launcher_plan_sha256") == file_sha256(launcher_plan_path),
        "Data marker launcher-plan hash mismatch.",
        errors,
    )
    expected_by_name = {row["run_name"]: row for row in dataset.get("runs", [])}
    expected_names = {row["run_name"] for row in ordered_matrix_runs()}
    _expect(set(expected_by_name) == expected_names, "Dataset run identities mismatch.", errors)

    launch_by_name: Dict[str, Dict[str, Any]] = {}
    launch_evidence = []
    for shard_index in range(args.launcher_shards):
        path = work_dir / f"training_manifest_shard_{shard_index:02d}_of_{args.launcher_shards:02d}.json"
        manifest = _read_json(path)
        _expect(manifest.get("status") == "complete", f"Training shard incomplete: {shard_index}", errors)
        _expect(manifest.get("config_hash") == config_hash, f"Training shard config mismatch: {shard_index}", errors)
        _expect(
            _resolve(manifest.get("dataset_manifest", "")) == dataset_path,
            f"Training shard dataset mismatch: {shard_index}",
            errors,
        )
        _expect(
            int(manifest.get("launcher_shard_index", -1)) == shard_index,
            f"Training shard index mismatch: {shard_index}",
            errors,
        )
        _expect(
            int(manifest.get("launcher_shards", -1)) == args.launcher_shards,
            f"Training topology mismatch: {shard_index}",
            errors,
        )
        _expect(
            manifest.get("wave_barrier_policy") == "declared_launcher_wave_barrier_v1",
            f"Training wave-barrier policy mismatch: {shard_index}",
            errors,
        )
        _expect(
            int(manifest.get("launcher_wave_count", -1)) == 4,
            f"Training wave count mismatch: {shard_index}",
            errors,
        )
        for run in manifest.get("runs", []):
            name = str(run.get("run_name", ""))
            if not name or name in launch_by_name:
                errors.append(f"Missing or duplicate launched run: {name}")
            else:
                launch_by_name[name] = dict(run)
            expected_run = expected_by_name.get(name)
            _expect(
                expected_run is not None
                and int(expected_run.get("launcher_shard_index", -1)) == shard_index,
                f"Run assigned to the wrong launcher shard: {name}",
                errors,
            )
            _expect(
                int(run.get("launcher_shard_index", -1)) == shard_index,
                f"Launch manifest shard metadata mismatch: {name}",
                errors,
            )
        _expect(
            len(manifest.get("runs", [])) == 12,
            f"Training shard run count mismatch: {shard_index}",
            errors,
        )
        launch_evidence.append(
            {"path": str(path), "sha256": file_sha256(path), "run_count": len(manifest.get("runs", []))}
        )
    _expect(set(launch_by_name) == expected_names, "Launched run identities mismatch.", errors)

    audited_runs: List[Dict[str, Any]] = []
    output_dirs: set[str] = set()
    for run_name in sorted(expected_names):
        expected = expected_by_name.get(run_name)
        launch = launch_by_name.get(run_name)
        if expected is None or launch is None:
            continue
        _expect(
            launch.get("status") in {"complete", "skipped_complete"},
            f"Training run incomplete: {run_name}",
            errors,
        )
        for field in (
            "mode", "generator_name", "budget_name", "seed", "train_path",
            "train_sha256", "n", "supervised_tokens", "launcher_shards",
            "launcher_shard_index", "launcher_wave_index", "launcher_assignment_policy",
        ):
            _expect(
                manifest_field_equal(field, launch.get(field), expected.get(field)),
                f"Dataset/launch mismatch for {run_name}: {field}",
                errors,
            )
        train_path = _resolve(expected["train_path"])
        _expect(file_sha256(train_path) == expected["train_sha256"], f"Training data hash mismatch: {run_name}", errors)
        _expect(nonempty_line_count(train_path) == global_n, f"Training data count mismatch: {run_name}", errors)
        run_config_path = _resolve(launch.get("config_path", ""))
        _expect(
            run_config_path.is_file() and file_sha256(run_config_path) == launch.get("run_config_sha256"),
            f"Run config evidence mismatch: {run_name}",
            errors,
        )
        output_path = _resolve(launch.get("output_dir", ""))
        _expect(str(output_path) not in output_dirs, f"Duplicate adapter output: {output_path}", errors)
        output_dirs.add(str(output_path))
        evidence = validated_adapter_evidence(output_path)
        if evidence is None:
            errors.append(f"Adapter evidence incomplete: {run_name}")
            continue
        expected_marker = {
            "run_name": run_name,
            "seed": str(expected["seed"]),
            "train_sha256": expected["train_sha256"],
            "run_config_sha256": launch.get("run_config_sha256"),
            "training_source_sha256": launch.get("training_source_sha256"),
            "launcher_source_sha256": launch.get("launcher_source_sha256"),
        }
        for field, value in expected_marker.items():
            _expect(str(evidence.get(field, "")) == str(value), f"Adapter marker mismatch: {run_name} {field}", errors)
        audited_runs.append(
            {
                **expected,
                "run_config_path": str(run_config_path),
                "run_config_sha256": launch["run_config_sha256"],
                "output_dir": str(output_path),
                "training_source_sha256": evidence["training_source_sha256"],
                "launcher_source_sha256": evidence["launcher_source_sha256"],
                "adapter_config_sha256": evidence["adapter_config_sha256"],
                "adapter_model_sha256": evidence["adapter_model_sha256"],
            }
        )
    _expect(len(audited_runs) == 36, f"Validated adapter count mismatch: {len(audited_runs)}", errors)
    report = {
        "status": "passed" if not errors else "failed",
        "experiment_name": config["experiment_name"],
        "protocol_variant": config["protocol_variant"],
        "config_path": str(config_path),
        "config_hash": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": file_sha256(dataset_path),
        "global_common_problem_count": global_n,
        "launcher_shards": args.launcher_shards,
        "launch_manifests": launch_evidence,
        "validated_run_count": len(audited_runs),
        "runs": audited_runs,
        "errors": errors,
    }
    if errors:
        raise SystemExit("Main-matrix training audit failed: " + " | ".join(errors))
    _write_json(audit_path, report)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"status=passed\nconfig_hash={config_hash}\n"
        f"dataset_manifest_sha256={file_sha256(dataset_path)}\n"
        f"training_audit_sha256={file_sha256(audit_path)}\n"
        f"global_common_problem_count={global_n}\n"
        "teacher_count=4\nrank_count=3\nseed_count=3\nrun_count=36\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "marker": str(marker_path), "run_count": 36}, indent=2))


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
