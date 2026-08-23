#!/usr/bin/env python3
"""Audit and seal the complete single-seed 7B-to-1.5B logit-KD experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    file_sha256,
    kd_run_name,
    load_protocol,
    protocol_hash,
    read_json,
    read_jsonl,
    resolve_project_path,
    supervision_mode,
    validate_budget_dataset,
    validated_training_marker,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    return parser.parse_args()


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _valid_eval_marker(path: Path, expected_n: int, protocol_digest: str, errors: List[str]) -> Dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing evaluation marker: {path}")
        return {}
    marker = read_json(path)
    prediction = Path(str(marker.get("prediction_path")))
    summary = Path(str(marker.get("summary_path")))
    _expect(marker.get("status") == "complete", f"incomplete evaluation marker: {path}", errors)
    _expect(marker.get("protocol_hash") == protocol_digest, f"evaluation protocol hash mismatch: {path}", errors)
    _expect(prediction.is_file(), f"missing evaluation prediction: {prediction}", errors)
    _expect(summary.is_file(), f"missing evaluation summary: {summary}", errors)
    if prediction.is_file():
        _expect(marker.get("prediction_sha256") == file_sha256(prediction), f"prediction hash mismatch: {prediction}", errors)
        rows = read_jsonl(prediction)
        _expect(len(rows) == expected_n, f"prediction count mismatch: {prediction}", errors)
        problem_ids = [str(row.get("problem_id")) for row in rows]
        _expect(len(problem_ids) == len(set(problem_ids)), f"duplicate prediction identities: {prediction}", errors)
    if summary.is_file():
        _expect(marker.get("summary_sha256") == file_sha256(summary), f"summary hash mismatch: {summary}", errors)
    _expect(bool(marker.get("source_sha256")), f"evaluation source evidence is missing: {path}", errors)
    for relative_path, expected_hash in marker.get("source_sha256", {}).items():
        source_path = PROJECT_ROOT / relative_path
        _expect(source_path.is_file(), f"evaluation source is missing: {source_path}", errors)
        if source_path.is_file():
            _expect(expected_hash == file_sha256(source_path), f"evaluation source hash mismatch: {source_path}", errors)
    return marker


def _validate_training_source_evidence(marker: Dict[str, Any], protocol_digest: str, label: str, errors: List[str]) -> None:
    _expect(marker.get("protocol_hash") == protocol_digest, f"training protocol hash mismatch: {label}", errors)
    config_path = Path(str(marker.get("config_path")))
    _expect(config_path.is_file(), f"training config is missing: {label}", errors)
    if config_path.is_file():
        _expect(marker.get("config_file_sha256") == file_sha256(config_path), f"training config hash mismatch: {label}", errors)
    sources = marker.get("source_sha256", {})
    _expect(bool(sources), f"training source evidence is missing: {label}", errors)
    for relative_path, expected_hash in sources.items():
        path = PROJECT_ROOT / relative_path
        _expect(path.is_file(), f"training source is missing: {path}", errors)
        if path.is_file():
            _expect(expected_hash == file_sha256(path), f"training source hash mismatch: {path}", errors)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load_protocol(args.config)
    digest = protocol_hash(protocol)
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    output_path = result_root / "formal" / "completion_audit.json"
    root_marker = result_root / "FORMAL_COMPLETE"
    if root_marker.exists():
        raise FileExistsError(f"Logit-KD experiment is already sealed: {root_marker}")
    errors: List[str] = []
    counts: Dict[str, int] = {}

    prepared_marker_path = result_root / "PREPARED"
    preflight_path = result_root / "preflight" / "parent_evidence.json"
    _expect(prepared_marker_path.is_file(), "missing PREPARED marker", errors)
    _expect(preflight_path.is_file(), "missing parent evidence", errors)
    if prepared_marker_path.is_file() and preflight_path.is_file():
        prepared = read_json(prepared_marker_path)
        preflight = read_json(preflight_path)
        _expect(prepared.get("status") == "complete", "preflight marker is incomplete", errors)
        _expect(prepared.get("protocol_hash") == digest, "preflight protocol hash mismatch", errors)
        _expect(
            prepared.get("parent_evidence_sha256") == file_sha256(preflight_path),
            "preflight evidence hash mismatch",
            errors,
        )
        _expect(preflight.get("status") == "passed", "parent evidence did not pass", errors)

    for budget_name in protocol["budgets"]:
        try:
            validate_budget_dataset(protocol, budget_name)
        except Exception as exc:
            errors.append(f"registered dataset failed validation for {budget_name}: {exc}")

    smoke_count = 0
    for label in ("a5000ada",):
        smoke_path = result_root / "smoke" / f"gpu_smoke_{label}.json"
        smoke_marker_path = result_root / "smoke" / f"SMOKE_COMPLETE_{label}"
        if not smoke_path.is_file() or not smoke_marker_path.is_file():
            errors.append(f"missing required GPU smoke evidence: {label}")
            continue
        smoke = read_json(smoke_path)
        smoke_marker = read_json(smoke_marker_path)
        _expect(smoke.get("status") == "passed", f"GPU smoke did not pass: {label}", errors)
        _expect(smoke.get("protocol_hash") == digest, f"GPU smoke protocol mismatch: {label}", errors)
        _expect(smoke_marker.get("smoke_sha256") == file_sha256(smoke_path), f"GPU smoke hash mismatch: {label}", errors)
        smoke_count += 1
    counts["gpu_smoke_profiles"] = smoke_count

    selection_path = result_root / "validation" / "selection.json"
    selection_marker_path = result_root / "validation" / "VALIDATION_COMPLETE"
    selection: Dict[str, Any] = {}
    if not selection_path.is_file() or not selection_marker_path.is_file():
        errors.append("validation selection evidence is incomplete")
    else:
        selection = read_json(selection_path)
        selection_marker = read_json(selection_marker_path)
        _expect(selection.get("status") == "complete", "validation selection is incomplete", errors)
        _expect(selection.get("protocol_hash") == digest, "selection protocol hash mismatch", errors)
        _expect(selection_marker.get("selection_sha256") == file_sha256(selection_path), "selection hash mismatch", errors)
        _expect(bool(selection.get("source_sha256")), "selection source evidence is missing", errors)
        for relative_path, expected_hash in selection.get("source_sha256", {}).items():
            source_path = PROJECT_ROOT / relative_path
            _expect(source_path.is_file(), f"selection source is missing: {source_path}", errors)
            if source_path.is_file():
                _expect(expected_hash == file_sha256(source_path), f"selection source hash mismatch: {source_path}", errors)

    validation_adapters = 0
    validation_evals = 0
    checkpoint_root = resolve_project_path(protocol["outputs"]["checkpoint_root"])
    validation_eval_root = result_root / "validation" / "eval" / "markers"
    _valid_eval_marker(
        validation_eval_root / "base_qwen2p5_1p5b_instruct.json",
        int(protocol["validation"]["limit"]),
        digest,
        errors,
    )
    validation_evals += 1
    for budget_name in protocol["budgets"]:
        _valid_eval_marker(
            validation_eval_root / f"sft__{budget_name}__seed_17.json",
            int(protocol["validation"]["limit"]),
            digest,
            errors,
        )
        validation_evals += 1
    for alpha in protocol["kd"]["alpha_grid"]:
        for temperature in protocol["kd"]["temperature_grid"]:
            for budget_name in protocol["budgets"]:
                run_name = kd_run_name(budget_name, alpha, temperature)
                marker = validated_training_marker(checkpoint_root / "validation" / run_name)
                if marker is None:
                    errors.append(f"missing or invalid validation adapter: {run_name}")
                else:
                    _validate_training_source_evidence(marker, digest, run_name, errors)
                validation_adapters += 1
                _valid_eval_marker(
                    validation_eval_root / f"kd__{run_name}.json",
                    int(protocol["validation"]["limit"]),
                    digest,
                    errors,
                )
                validation_evals += 1
    counts["validation_adapters"] = validation_adapters
    counts["validation_evaluations"] = validation_evals

    if selection:
        selected_alpha = float(selection["selected_alpha"])
        selected_temperature = float(selection["selected_temperature"])
    else:
        selected_alpha = selected_temperature = -1.0
    formal_adapters = 0
    formal_evals = 0
    formal_eval_root = result_root / "formal" / "eval" / "markers"
    for budget_name in protocol["budgets"]:
        adapter_dir = checkpoint_root / "formal" / f"{budget_name}__seed_17"
        marker = validated_training_marker(adapter_dir)
        if marker is None:
            errors.append(f"missing or invalid formal adapter: {budget_name}")
        else:
            _validate_training_source_evidence(marker, digest, budget_name, errors)
            _expect(float(marker.get("alpha", -1)) == selected_alpha, f"formal alpha mismatch: {budget_name}", errors)
            _expect(
                float(marker.get("temperature", -1)) == selected_temperature,
                f"formal temperature mismatch: {budget_name}",
                errors,
            )
        formal_adapters += 1
        run_name = f"kd__{kd_run_name(budget_name, selected_alpha, selected_temperature)}"
        _valid_eval_marker(
            formal_eval_root / f"{run_name}.json",
            int(protocol["formal"]["limit"]),
            digest,
            errors,
        )
        formal_evals += 1
    counts["formal_adapters"] = formal_adapters
    counts["formal_evaluations"] = formal_evals
    counts["formal_predictions"] = formal_evals * int(protocol["formal"]["limit"])

    num_shards = int(protocol["outputs"]["logit_shards_per_snapshot"])
    logit_snapshots = 0
    logit_shards = 0
    logit_records = 0
    for budget_name in protocol["budgets"]:
        expected_records = int(protocol["budgets"][budget_name]["expected_records"])
        for method in ("teacher", "base", "sft", "kd"):
            snapshot_dir = result_root / "formal" / "logits" / budget_name / method
            observed_indices = set()
            source_indices = set()
            for shard_index in range(num_shards):
                stem = f"shard_{shard_index:02d}_of_{num_shards:02d}"
                marker_path = snapshot_dir / f"{stem}.complete.json"
                if not marker_path.is_file():
                    errors.append(f"missing logit marker: {marker_path}")
                    continue
                marker = read_json(marker_path)
                tensor_path = Path(str(marker.get("tensor_path")))
                metadata_path = Path(str(marker.get("metadata_path")))
                _expect(marker.get("status") == "complete", f"incomplete logit marker: {marker_path}", errors)
                _expect(marker.get("protocol_hash") == digest, f"logit protocol hash mismatch: {marker_path}", errors)
                _expect(tensor_path.is_file(), f"missing logit tensor: {tensor_path}", errors)
                _expect(metadata_path.is_file(), f"missing logit metadata: {metadata_path}", errors)
                if tensor_path.is_file():
                    _expect(marker.get("tensor_sha256") == file_sha256(tensor_path), f"logit tensor hash mismatch: {tensor_path}", errors)
                if metadata_path.is_file():
                    _expect(
                        marker.get("metadata_sha256") == file_sha256(metadata_path),
                        f"logit metadata hash mismatch: {metadata_path}",
                        errors,
                    )
                    metadata = read_json(metadata_path)
                    _expect(
                        bool(metadata.get("source_code_sha256")),
                        f"logit source evidence is missing: {metadata_path}",
                        errors,
                    )
                    for relative_path, expected_hash in metadata.get("source_code_sha256", {}).items():
                        source_path = PROJECT_ROOT / relative_path
                        _expect(source_path.is_file(), f"logit source is missing: {source_path}", errors)
                        if source_path.is_file():
                            _expect(
                                expected_hash == file_sha256(source_path),
                                f"logit source hash mismatch: {source_path}",
                                errors,
                            )
                    for record in metadata.get("record_metadata", []):
                        source_index = int(record["source_index"])
                        if source_index in source_indices:
                            errors.append(f"duplicate logit source index: {budget_name}/{method}/{source_index}")
                        source_indices.add(source_index)
                observed_indices.add(shard_index)
                logit_shards += 1
            _expect(observed_indices == set(range(num_shards)), f"logit shard coverage mismatch: {budget_name}/{method}", errors)
            _expect(
                source_indices == set(range(expected_records)),
                f"logit record coverage mismatch: {budget_name}/{method}",
                errors,
            )
            logit_records += len(source_indices)
            logit_snapshots += 1
    counts["logit_snapshots"] = logit_snapshots
    counts["logit_shards"] = logit_shards
    counts["logit_records"] = logit_records

    analysis_dir = result_root / "formal" / "analysis"
    analysis_path = analysis_dir / "logit_kd_analysis.json"
    artifact_manifest_path = analysis_dir / "analysis_artifact_manifest.json"
    analysis_marker_path = analysis_dir / "ANALYSIS_COMPLETE"
    for path, label in (
        (analysis_path, "analysis"),
        (artifact_manifest_path, "analysis artifact manifest"),
        (analysis_marker_path, "analysis completion marker"),
    ):
        _expect(path.is_file(), f"missing {label}: {path}", errors)
    if analysis_path.is_file():
        analysis = read_json(analysis_path)
        _expect(analysis.get("status") == "complete", "analysis status is incomplete", errors)
        _expect(analysis.get("protocol_hash") == digest, "analysis protocol hash mismatch", errors)
        _expect(bool(analysis.get("source_sha256")), "analysis source evidence is missing", errors)
        for relative_path, expected_hash in analysis.get("source_sha256", {}).items():
            source_path = PROJECT_ROOT / relative_path
            _expect(source_path.is_file(), f"analysis source is missing: {source_path}", errors)
            if source_path.is_file():
                _expect(expected_hash == file_sha256(source_path), f"analysis source hash mismatch: {source_path}", errors)
    if artifact_manifest_path.is_file():
        artifact_manifest = read_json(artifact_manifest_path)
        for artifact in artifact_manifest.get("artifacts", []):
            path = Path(str(artifact.get("path")))
            _expect(path.is_file(), f"missing registered analysis artifact: {path}", errors)
            if path.is_file():
                _expect(artifact.get("sha256") == file_sha256(path), f"analysis artifact hash mismatch: {path}", errors)
    if analysis_marker_path.is_file() and artifact_manifest_path.is_file():
        analysis_marker = read_json(analysis_marker_path)
        _expect(
            analysis_marker.get("artifact_manifest_sha256") == file_sha256(artifact_manifest_path),
            "analysis marker hash mismatch",
            errors,
        )

    report = {
        "status": "passed" if not errors else "failed",
        "experiment_name": protocol["experiment_name"],
        "protocol_variant": protocol["protocol_variant"],
        "supervision_mode": supervision_mode(protocol),
        "scope": protocol["scope"],
        "protocol_hash": digest,
        "counts": counts,
        "errors": errors,
        "evidence": {
            "parent_evidence": str(preflight_path),
            "validation_selection": str(selection_path),
            "analysis": str(analysis_path),
            "analysis_artifact_manifest": str(artifact_manifest_path),
        },
    }
    write_json(output_path, report)
    if errors:
        raise SystemExit(f"Logit-KD completion audit failed with {len(errors)} errors: {output_path}")
    write_json(
        root_marker,
        {
            "status": "complete",
            "protocol_hash": digest,
            "completion_audit_path": str(output_path),
            "completion_audit_sha256": file_sha256(output_path),
            "formal_adapters": counts["formal_adapters"],
            "formal_evaluations": counts["formal_evaluations"],
            "logit_snapshots": counts["logit_snapshots"],
        },
    )
    logging.info("logit_kd_completion_audit_passed marker=%s", root_marker)


if __name__ == "__main__":
    main()
