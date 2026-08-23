#!/usr/bin/env python3
"""Independently audit completion and hashes for the exploratory MATH-mix pilot."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import file_sha256, nonempty_line_count, validated_adapter_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--figure-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root) / "pilot"
    errors: List[str] = []
    counts: Dict[str, int] = {}

    source_manifest_path = root / "source" / "math_train_source_manifest.json"
    source = _checked_json(source_manifest_path, errors)
    if source:
        counts["math_source_problems"] = int(source.get("sample_count", -1))
        _expect(counts["math_source_problems"] == 1000, "MATH source count is not 1000", errors)
        _check_file_hash(source.get("source_path"), source.get("source_sha256"), errors)

    selection_path = root / "selected" / "selection_audit.json"
    selection = _checked_json(selection_path, errors)
    if selection:
        _expect(selection.get("status") == "passed", "MATH selection audit did not pass", errors)
        counts["raw_candidates"] = int(selection.get("actual_total_candidates", -1))
        counts["math_common_problems"] = int(selection.get("common_problem_count", -1))
        _expect(counts["raw_candidates"] == 9000, "Raw candidate count is not 9000", errors)
        _expect(counts["math_common_problems"] >= 300, "MATH common-problem count is below 300", errors)
        for shard in selection.get("input_shards", []):
            _check_file_hash(shard.get("path"), shard.get("sha256"), errors)
            path = Path(str(shard.get("path")))
            if path.is_file():
                _expect(
                    nonempty_line_count(path) == int(shard.get("record_count", -1)),
                    f"Raw shard row-count mismatch: {path}",
                    errors,
                )

    sft_manifest_path = root / "sft_data" / "dataset_manifest.json"
    sft = _checked_json(sft_manifest_path, errors)
    if sft:
        _expect(sft.get("status") == "complete", "SFT dataset manifest is incomplete", errors)
        counts["sft_runs"] = len(sft.get("runs", []))
        _expect(counts["sft_runs"] == 6, "SFT run count is not 6", errors)
        for run in sft.get("runs", []):
            _check_file_hash(run.get("train_path"), run.get("train_sha256"), errors)
            path = Path(str(run.get("train_path")))
            if path.is_file():
                _expect(
                    nonempty_line_count(path) == int(run.get("n", -1)),
                    f"SFT row-count mismatch: {path}",
                    errors,
                )

    training_paths = [Path(path) for path in sorted(glob.glob(str(root / "training" / "training_manifest_shard_*.json")))]
    training_runs = []
    for path in training_paths:
        manifest = _checked_json(path, errors)
        if not manifest:
            continue
        _expect(manifest.get("status") == "complete", f"Training manifest incomplete: {path}", errors)
        training_runs.extend(manifest.get("runs", []))
    counts["training_runs"] = len(training_runs)
    _expect(counts["training_runs"] == 6, "Completed training run count is not 6", errors)
    for run in training_runs:
        evidence = validated_adapter_evidence(str(run.get("output_dir")))
        _expect(evidence is not None, f"Invalid adapter evidence: {run.get('output_dir')}", errors)

    suite_path = root / "eval_suite" / "eval_suite_manifest.json"
    suite = _checked_json(suite_path, errors)
    if suite:
        _expect(suite.get("status") == "complete", "Evaluation suite is incomplete", errors)
        counts["eval_datasets"] = len(suite.get("datasets", []))
        counts["eval_examples"] = sum(int(item.get("n", 0)) for item in suite.get("datasets", []))
        _expect(counts["eval_datasets"] == 3, "Evaluation dataset count is not 3", errors)
        _expect(counts["eval_examples"] == 330, "Evaluation example total is not 330", errors)
        for dataset in suite.get("datasets", []):
            _check_file_hash(dataset.get("path"), dataset.get("sha256"), errors)

    eval_manifests = [Path(path) for path in sorted(glob.glob(str(root / "eval" / "model_manifests" / "*.json")))]
    counts["evaluation_models"] = len(eval_manifests)
    _expect(counts["evaluation_models"] == 13, "Evaluation model count is not 13", errors)
    evaluation_artifacts = 0
    for path in eval_manifests:
        manifest = _checked_json(path, errors)
        if not manifest:
            continue
        _expect(manifest.get("status") == "complete", f"Evaluation manifest incomplete: {path}", errors)
        artifacts = manifest.get("artifacts", [])
        evaluation_artifacts += len(artifacts)
        for artifact in artifacts:
            _check_file_hash(artifact.get("prediction_path"), artifact.get("prediction_sha256"), errors)
            _check_file_hash(artifact.get("summary_path"), artifact.get("summary_sha256"), errors)
    counts["evaluation_model_datasets"] = evaluation_artifacts
    _expect(evaluation_artifacts == 39, "Evaluation model-dataset artifact count is not 39", errors)

    analysis_dir = root / "analysis"
    artifact_manifest = _checked_json(analysis_dir / "analysis_artifact_manifest.json", errors)
    if artifact_manifest:
        _expect(artifact_manifest.get("status") == "complete", "Analysis artifact manifest incomplete", errors)
        for artifact in artifact_manifest.get("artifacts", []):
            _check_file_hash(artifact.get("path"), artifact.get("sha256"), errors)
    for filename in ("gsm_math500_aime25_accuracy.png", "gsm_math500_aime25_accuracy.pdf"):
        _expect((Path(args.figure_root) / filename).is_file(), f"Missing publication figure: {filename}", errors)

    report = {
        "status": "passed" if not errors else "failed",
        "evidence_level": "exploratory_single_seed_pilot",
        "counts": counts,
        "errors": errors,
    }
    report_path = analysis_dir / "completion_audit.json"
    _write_json(report_path, report)
    marker_path = root / "PILOT_COMPLETE"
    if errors:
        raise SystemExit(f"Pilot completion audit failed with {len(errors)} error(s): {report_path}")
    marker_path.write_text(
        "status=passed\nevidence_level=exploratory_single_seed_pilot\n"
        f"completion_audit_sha256={file_sha256(report_path)}\n",
        encoding="utf-8",
    )
    print(f"pilot_completion_audit=passed report={report_path}")


def _checked_json(path: Path, errors: List[str]) -> Dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"Missing JSON artifact: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        errors.append(f"Could not read JSON artifact {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"JSON artifact is not an object: {path}")
        return None
    return payload


def _check_file_hash(raw_path: Any, expected_hash: Any, errors: List[str]) -> None:
    if not raw_path or not expected_hash:
        errors.append(f"Missing path/hash evidence: path={raw_path!r} hash={expected_hash!r}")
        return
    path = Path(str(raw_path))
    if not path.is_file():
        errors.append(f"Missing file: {path}")
        return
    actual = file_sha256(path)
    if actual != expected_hash:
        errors.append(f"Hash mismatch: {path} expected={expected_hash} actual={actual}")


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
