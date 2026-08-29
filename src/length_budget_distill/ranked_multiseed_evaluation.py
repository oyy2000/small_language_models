"""Evidence helpers for combining sealed ranked-length training seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
    validated_adapter_evidence,
)


EVAL_PROTOCOL_VARIANT = "comparative_multiseed_ranked_length_evaluation"
LENGTH_BUDGETS = ("relative_short", "relative_medium", "relative_long")


def training_run_field_equal(
    project_root: str | Path,
    field_name: str,
    left: Any,
    right: Any,
) -> bool:
    """Compare training evidence fields across historical serialization formats."""

    if field_name == "seed":
        try:
            return int(left) == int(right)
        except (TypeError, ValueError):
            return False
    if field_name == "output_dir":
        project = Path(project_root)
        return _resolve(project, left).resolve() == _resolve(project, right).resolve()
    return left == right


def validate_eval_settings(config: Mapping[str, Any]) -> None:
    if str(config.get("protocol_variant")) != EVAL_PROTOCOL_VARIANT:
        raise ValueError(f"Unexpected multi-seed evaluation protocol: {config.get('protocol_variant')}")
    if str(config.get("student", {}).get("model_name")) != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Multi-seed evaluation student must remain Qwen2.5-1.5B-Instruct.")
    locked = {
        "dataset_split": "test",
        "start_index": 50,
        "limit": 1269,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "batch_size": 32,
        "include_base_model": True,
        "expected_adapter_count": 9,
        "expected_run_count": 10,
    }
    evaluation = dict(config.get("evaluation", {}))
    for key, expected in locked.items():
        if evaluation.get(key) != expected:
            raise ValueError(
                f"Locked multi-seed evaluation field changed: {key}="
                f"{evaluation.get(key)!r}, expected={expected!r}"
            )
    dataset = dict(config.get("dataset", {}))
    if (
        dataset.get("source") != "hf_dataset"
        or dataset.get("dataset_name") != "openai/gsm8k"
        or dataset.get("dataset_config") != "main"
    ):
        raise ValueError("Evaluation dataset must remain openai/gsm8k main.")
    analysis = dict(config.get("analysis", {}))
    if [int(seed) for seed in analysis.get("training_seeds", [])] != [17, 42, 73]:
        raise ValueError("Multi-seed analysis is locked to training seeds 17, 42, and 73.")


def model_id(generator_name: str, budget_name: str, seed: int) -> str:
    if budget_name not in LENGTH_BUDGETS:
        raise ValueError(f"Unexpected ranked budget: {budget_name}")
    return f"{generator_name}__{budget_name}__seed_{int(seed)}"


def freeze_parent_training(
    spec: Mapping[str, Any],
    project_root: str | Path,
) -> Dict[str, Any]:
    """Validate a completed training parent and record all immutable hashes."""

    project = Path(project_root)
    config_path = _resolve(project, spec["config_path"])
    input_path = _resolve(project, spec["input_manifest_path"])
    launch_path = _resolve(project, spec["launch_manifest_path"])
    audit_path = _resolve(project, spec["audit_path"])
    marker_path = _resolve(project, spec["completion_marker_path"])
    for path in (config_path, input_path, launch_path, audit_path, marker_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    training_config = _read_json(config_path)
    input_manifest = _read_json(input_path)
    launch_manifest = _read_json(launch_path)
    audit = _read_json(audit_path)
    marker = read_key_value_marker(marker_path)
    config_hash = canonical_sha256(training_config)
    expected_seeds = [int(seed) for seed in spec["expected_seeds"]]
    actual_seeds = sorted({int(run["seed"]) for run in audit.get("runs", [])})
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"Training parent seed mismatch: expected={expected_seeds} actual={actual_seeds}"
        )
    _require_equal("training input status", input_manifest.get("status"), "complete")
    _require_equal("training launch status", launch_manifest.get("status"), "complete")
    _require_equal("training audit status", audit.get("status"), "passed")
    _require_equal("training audit errors", audit.get("errors"), [])
    for label, payload in (
        ("input", input_manifest),
        ("launch", launch_manifest),
        ("audit", audit),
    ):
        _require_equal(f"training {label} config hash", payload.get("config_hash"), config_hash)
    _require_equal("training marker status", marker.get("status"), "passed")
    _require_equal("training marker config hash", marker.get("config_hash"), config_hash)
    _require_equal(
        "training marker audit hash",
        marker.get("training_audit_sha256"),
        file_sha256(audit_path),
    )
    _require_equal(
        "training marker launch hash",
        marker.get("launch_manifest_sha256"),
        file_sha256(launch_path),
    )
    validated_runs = validate_parent_training_runs(
        {
            "config_path": str(config_path),
            "canonical_config_sha256": config_hash,
            "config_file_sha256": file_sha256(config_path),
            "input_manifest_path": str(input_path),
            "input_manifest_sha256": file_sha256(input_path),
            "launch_manifest_path": str(launch_path),
            "launch_manifest_sha256": file_sha256(launch_path),
            "audit_path": str(audit_path),
            "audit_sha256": file_sha256(audit_path),
            "completion_marker_path": str(marker_path),
            "completion_marker_sha256": file_sha256(marker_path),
            "expected_seeds": expected_seeds,
            "expected_run_count": int(spec["expected_run_count"]),
        },
        project,
    )
    return {
        "label": str(spec["label"]),
        "config_path": str(config_path),
        "canonical_config_sha256": config_hash,
        "config_file_sha256": file_sha256(config_path),
        "input_manifest_path": str(input_path),
        "input_manifest_sha256": file_sha256(input_path),
        "launch_manifest_path": str(launch_path),
        "launch_manifest_sha256": file_sha256(launch_path),
        "audit_path": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "completion_marker_path": str(marker_path),
        "completion_marker_sha256": file_sha256(marker_path),
        "expected_seeds": expected_seeds,
        "expected_run_count": int(spec["expected_run_count"]),
        "validated_run_count": len(validated_runs),
    }


def validate_parent_training_runs(
    parent: Mapping[str, Any],
    project_root: str | Path,
) -> List[Dict[str, Any]]:
    """Revalidate a frozen parent and return adapter records for evaluation."""

    project = Path(project_root)
    config_path = _resolve(project, parent["config_path"])
    input_path = _resolve(project, parent["input_manifest_path"])
    launch_path = _resolve(project, parent["launch_manifest_path"])
    audit_path = _resolve(project, parent["audit_path"])
    marker_path = _resolve(project, parent["completion_marker_path"])
    expected_files = (
        (config_path, parent["config_file_sha256"]),
        (input_path, parent["input_manifest_sha256"]),
        (launch_path, parent["launch_manifest_sha256"]),
        (audit_path, parent["audit_sha256"]),
        (marker_path, parent["completion_marker_sha256"]),
    )
    for path, expected_hash in expected_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        _require_equal(f"file hash {path}", file_sha256(path), str(expected_hash))

    training_config = _read_json(config_path)
    input_manifest = _read_json(input_path)
    launch_manifest = _read_json(launch_path)
    audit = _read_json(audit_path)
    marker = read_key_value_marker(marker_path)
    config_hash = canonical_sha256(training_config)
    _require_equal(
        "training canonical config hash",
        config_hash,
        parent["canonical_config_sha256"],
    )
    _require_equal("training input status", input_manifest.get("status"), "complete")
    _require_equal("training launch status", launch_manifest.get("status"), "complete")
    _require_equal("training audit status", audit.get("status"), "passed")
    _require_equal("training audit errors", audit.get("errors"), [])
    _require_equal("training marker status", marker.get("status"), "passed")
    _require_equal("training marker config hash", marker.get("config_hash"), config_hash)
    _require_equal(
        "training marker audit hash",
        marker.get("training_audit_sha256"),
        file_sha256(audit_path),
    )

    launch_by_name = _unique_by_name(launch_manifest.get("runs", []), "launch")
    audit_by_name = _unique_by_name(audit.get("runs", []), "audit")
    _require_equal("training run identities", set(launch_by_name), set(audit_by_name))
    _require_equal(
        "training run count",
        len(launch_by_name),
        int(parent["expected_run_count"]),
    )
    expected_seeds = [int(seed) for seed in parent["expected_seeds"]]
    actual_seeds = sorted({int(run["seed"]) for run in audit_by_name.values()})
    _require_equal("training seeds", actual_seeds, expected_seeds)

    result: List[Dict[str, Any]] = []
    for run_name in sorted(launch_by_name):
        launch_run = launch_by_name[run_name]
        audit_run = audit_by_name[run_name]
        if launch_run.get("status") not in {"complete", "skipped_complete"}:
            raise ValueError(f"Incomplete training run: {run_name}")
        for field in (
            "budget_name",
            "seed",
            "n",
            "supervised_tokens",
            "train_sha256",
            "run_config_sha256",
            "output_dir",
            "training_source_sha256",
            "launcher_source_sha256",
            "adapter_config_sha256",
            "adapter_model_sha256",
        ):
            if not training_run_field_equal(
                project,
                field,
                launch_run.get(field),
                audit_run.get(field),
            ):
                raise ValueError(
                    f"training launch/audit field {run_name} {field} mismatch: "
                    f"expected={audit_run.get(field)!r} actual={launch_run.get(field)!r}"
                )
        adapter_path = _resolve(project, audit_run["output_dir"])
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
                audit_run.get(field),
                evidence.get(field),
            )
        generator_name = str(audit_run.get("generator_name") or launch_run.get("generator_name"))
        if generator_name != "qwen2p5_7b":
            raise ValueError(f"Unexpected Phase-A generator: {generator_name}")
        budget_name = str(audit_run["budget_name"])
        seed = int(audit_run["seed"])
        result.append(
            {
                **audit_run,
                "generator_name": generator_name,
                "model_id": model_id(generator_name, budget_name, seed),
                "adapter_path": str(adapter_path),
            }
        )
    return result


def validate_all_parent_trainings(
    config: Mapping[str, Any],
    project_root: str | Path,
) -> List[Dict[str, Any]]:
    validate_eval_settings(config)
    parents = config.get("parent_trainings", [])
    if not isinstance(parents, list) or len(parents) != 2:
        raise ValueError("Frozen multi-seed config must contain two training parents.")
    runs: List[Dict[str, Any]] = []
    for parent in parents:
        runs.extend(validate_parent_training_runs(parent, project_root))
    ids = [str(run["model_id"]) for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model identity across training parents.")
    if len(runs) != int(config["evaluation"]["expected_adapter_count"]):
        raise ValueError(f"Adapter count mismatch across parents: {len(runs)}")
    expected_seeds = [17, 42, 73]
    for seed in expected_seeds:
        budgets = {
            str(run["budget_name"])
            for run in runs
            if int(run["seed"]) == seed
        }
        if budgets != set(LENGTH_BUDGETS):
            raise ValueError(f"Rank coverage mismatch for seed {seed}: {sorted(budgets)}")
    return sorted(runs, key=lambda run: (int(run["seed"]), LENGTH_BUDGETS.index(run["budget_name"])))


def _unique_by_name(raw_runs: Any, label: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_runs, list):
        raise ValueError(f"{label} runs must be a list.")
    result: Dict[str, Dict[str, Any]] = {}
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} run is not an object.")
        name = str(raw.get("run_name", ""))
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate {label} run name: {name}")
        result[name] = dict(raw)
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
