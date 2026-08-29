"""Training-evidence validation for the 36-adapter ranked main matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .factorial import canonical_sha256, file_sha256, read_key_value_marker, validated_adapter_evidence
from .ranked_multiteacher import ordered_matrix_runs, validate_protocol


def matrix_model_id(teacher_name: str, rank_name: str, seed: int) -> str:
    return f"{teacher_name}__{rank_name}__seed_{int(seed)}"


def validated_matrix_training_runs(
    config: Mapping[str, Any], project_root: str | Path
) -> List[Dict[str, Any]]:
    validate_protocol(config, require_frozen=True)
    project = Path(project_root)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    result_root = _resolve(project, config["outputs"]["result_root"])
    audit_path = result_root / "formal/training/audit/training_audit.json"
    marker_path = result_root / "formal/training/audit/TRAINING_COMPLETE"
    audit = _read_json(audit_path)
    marker = read_key_value_marker(marker_path)
    _require_equal("training audit status", audit.get("status"), "passed")
    _require_equal("training audit errors", audit.get("errors"), [])
    _require_equal("training audit config hash", audit.get("config_hash"), config_hash)
    _require_equal("training marker status", marker.get("status"), "passed")
    _require_equal("training marker config hash", marker.get("config_hash"), config_hash)
    _require_equal("training marker audit hash", marker.get("training_audit_sha256"), file_sha256(audit_path))
    _require_equal("training audit run count", audit.get("validated_run_count"), 36)
    expected = {row["run_name"]: row for row in ordered_matrix_runs()}
    audit_by_name = _unique_by_name(audit.get("runs", []))
    _require_equal("training run identities", set(audit_by_name), set(expected))
    result: List[Dict[str, Any]] = []
    for run_name in sorted(expected):
        run = audit_by_name[run_name]
        spec = expected[run_name]
        for field in ("generator_name", "budget_name", "seed"):
            _require_equal(f"training run field {run_name} {field}", run.get(field), spec[field])
        adapter_path = _resolve(project, run["output_dir"])
        evidence = validated_adapter_evidence(adapter_path)
        if evidence is None:
            raise ValueError(f"Invalid adapter evidence: {adapter_path}")
        for field in (
            "train_sha256", "run_config_sha256", "training_source_sha256",
            "launcher_source_sha256", "adapter_config_sha256", "adapter_model_sha256",
        ):
            _require_equal(f"adapter evidence {run_name} {field}", run.get(field), evidence.get(field))
        result.append(
            {
                **run,
                "model_id": matrix_model_id(
                    str(run["generator_name"]), str(run["budget_name"]), int(run["seed"])
                ),
                "adapter_path": str(adapter_path),
            }
        )
    ids = [row["model_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate main-matrix evaluation model identity.")
    return result


def _unique_by_name(rows: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("Training audit runs must be a list.")
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("run_name", "")) if isinstance(row, dict) else ""
        if not name or name in result:
            raise ValueError(f"Missing or duplicate training run: {name}")
        result[name] = dict(row)
    return result


def _resolve(project: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")
