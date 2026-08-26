"""Evidence validation helpers for ranked-length student evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .factorial import (
    canonical_sha256,
    file_sha256,
    nonempty_line_count,
    read_key_value_marker,
    validated_adapter_evidence,
)


EXPECTED_ADAPTERS = {
    "equal_example__qwen2p5_7b__relative_short__seed_17": "relative_short",
    "equal_example__qwen2p5_7b__relative_medium__seed_17": "relative_medium",
    "equal_example__qwen2p5_7b__relative_long__seed_17": "relative_long",
}


def protocol_hash(config: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})


def validate_evaluation_protocol(config: Mapping[str, Any]) -> None:
    evaluation = config.get("evaluation", {})
    if str(config.get("protocol_variant")) != "revised_formal_single_seed_ranked_length_evaluation":
        raise ValueError("Unexpected ranked-length evaluation protocol variant.")
    if str(config.get("student", {}).get("model_name")) != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Evaluation student must remain Qwen2.5-1.5B-Instruct.")
    locked = {
        "dataset_split": "test",
        "start_index": 50,
        "limit": 1269,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "batch_size": 32,
        "include_base_model": True,
        "expected_adapter_count": 3,
        "expected_run_count": 4,
    }
    for key, expected in locked.items():
        if evaluation.get(key) != expected:
            raise ValueError(
                f"Locked ranked-length evaluation field changed: {key}="
                f"{evaluation.get(key)!r}, expected={expected!r}"
            )
    dataset = config.get("dataset", {})
    if (
        dataset.get("source") != "hf_dataset"
        or dataset.get("dataset_name") != "openai/gsm8k"
        or dataset.get("dataset_config") != "main"
    ):
        raise ValueError("Evaluation dataset must remain openai/gsm8k main.")


def validate_parent_training(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> List[Dict[str, Any]]:
    """Validate the sealed training chain and return its three adapter runs."""

    project = Path(project_root)
    parent = dict(config["parent_training"])
    training_config_path = _resolve(project, parent["config_path"])
    training_config = _read_json(training_config_path)
    actual_training_hash = canonical_sha256(training_config)
    _require_equal(
        "training canonical config hash",
        actual_training_hash,
        parent["canonical_config_sha256"],
    )
    _require_file_hash(training_config_path, parent["config_file_sha256"])

    input_manifest_path = _resolve(project, parent["input_manifest_path"])
    launch_manifest_path = _resolve(project, parent["launch_manifest_path"])
    audit_path = _resolve(project, parent["audit_path"])
    marker_path = _resolve(project, parent["completion_marker_path"])
    for path, expected_hash in (
        (input_manifest_path, parent["input_manifest_sha256"]),
        (launch_manifest_path, parent["launch_manifest_sha256"]),
        (audit_path, parent["audit_sha256"]),
        (marker_path, parent["completion_marker_sha256"]),
    ):
        _require_file_hash(path, expected_hash)

    input_manifest = _read_json(input_manifest_path)
    launch_manifest = _read_json(launch_manifest_path)
    audit = _read_json(audit_path)
    marker = read_key_value_marker(marker_path)
    _require_equal("training input status", input_manifest.get("status"), "complete")
    _require_equal("training launch status", launch_manifest.get("status"), "complete")
    _require_equal("training audit status", audit.get("status"), "passed")
    _require_equal("training audit errors", audit.get("errors"), [])
    for label, payload in (
        ("input", input_manifest),
        ("launch", launch_manifest),
        ("audit", audit),
    ):
        _require_equal(
            f"training {label} config hash",
            payload.get("config_hash"),
            actual_training_hash,
        )
    _require_equal("training marker status", marker.get("status"), "passed")
    _require_equal("training marker config hash", marker.get("config_hash"), actual_training_hash)
    _require_equal(
        "training marker audit hash",
        marker.get("training_audit_sha256"),
        file_sha256(audit_path),
    )
    _require_equal(
        "training marker launch hash",
        marker.get("launch_manifest_sha256"),
        file_sha256(launch_manifest_path),
    )

    runs: Dict[str, Dict[str, Any]] = {}
    for run in launch_manifest.get("runs", []):
        run_name = str(run.get("run_name", ""))
        if run_name in runs:
            raise ValueError(f"Duplicate training run: {run_name}")
        if run_name not in EXPECTED_ADAPTERS:
            raise ValueError(f"Unexpected ranked-length training run: {run_name}")
        if run.get("status") not in {"complete", "skipped_complete"}:
            raise ValueError(f"Incomplete ranked-length adapter: {run_name}")
        adapter_path = _resolve(project, run["output_dir"])
        evidence = validated_adapter_evidence(adapter_path)
        if evidence is None:
            raise ValueError(f"Invalid adapter evidence: {adapter_path}")
        for field in (
            "train_sha256",
            "run_config_sha256",
            "training_source_sha256",
            "launcher_source_sha256",
            "adapter_config_sha256",
            "adapter_model_sha256",
        ):
            _require_equal(
                f"adapter evidence {run_name} {field}",
                run.get(field),
                evidence.get(field),
            )
        runs[run_name] = {
            **run,
            "model_id": EXPECTED_ADAPTERS[run_name],
            "adapter_path": str(adapter_path),
        }
    _require_equal("ranked-length adapter identities", set(runs), set(EXPECTED_ADAPTERS))
    _require_equal("ranked-length adapter count", len(runs), 3)
    return [runs[name] for name in EXPECTED_ADAPTERS]


def completed_evaluation_evidence(
    prediction_path: str | Path,
    summary_path: str | Path,
    *,
    expected_n: int,
    expected_start_index: int,
    expected_split: str,
) -> Dict[str, Any] | None:
    prediction = Path(prediction_path)
    summary_file = Path(summary_path)
    if not prediction.is_file() or not summary_file.is_file():
        return None
    try:
        summary = _read_json(summary_file)
        if int(summary.get("n", -1)) != expected_n:
            return None
        if int(summary.get("start_index", -1)) != expected_start_index:
            return None
        if summary.get("split") != expected_split:
            return None
        if nonempty_line_count(prediction) != expected_n:
            return None
        seen: set[str] = set()
        correct = 0
        with prediction.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                problem_id = str(row["problem_id"])
                if problem_id in seen:
                    return None
                seen.add(problem_id)
                correct += int(bool(row["is_correct"]))
        if len(seen) != expected_n or int(summary.get("correct", -1)) != correct:
            return None
        expected_accuracy = correct / expected_n
        if abs(float(summary.get("accuracy", -1.0)) - expected_accuracy) > 1e-12:
            return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return {
        "prediction_sha256": file_sha256(prediction),
        "summary_sha256": file_sha256(summary_file),
        "n": expected_n,
        "correct": correct,
        "accuracy": expected_accuracy,
        "problem_ids": sorted(seen),
    }


def _resolve(project: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _require_file_hash(path: Path, expected: Any) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    _require_equal(f"file hash {path}", file_sha256(path), str(expected))


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")
