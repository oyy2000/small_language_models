#!/usr/bin/env python3
"""Freeze the 36-adapter main matrix after Phase-A evidence passes audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    read_key_value_marker,
)
from length_budget_distill.ranked_multiteacher import validate_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-config",
        default="configs/capacity_length_ranked_sampling_multiteacher_v1.json",
    )
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-training-overlay", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = _resolve(args.template_config)
    output_path = _resolve(args.output_config)
    overlay_path = _resolve(args.output_training_overlay)
    if output_path.exists() or overlay_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen main-matrix protocol: {output_path}, {overlay_path}"
        )
    template = _read_json(template_path)
    validate_protocol(template)
    spec = dict(template["phase_a_parent_spec"])
    marker_path = _resolve(spec["completion_marker_path"])
    audit_path = _resolve(spec["completion_audit_path"])
    eval_config_path = _resolve(spec["frozen_eval_config_path"])
    for path in (marker_path, audit_path, eval_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    marker = read_key_value_marker(marker_path)
    audit = _read_json(audit_path)
    eval_config = _read_json(eval_config_path)
    eval_hash = canonical_sha256(eval_config)
    _require_equal("Phase-A marker status", marker.get("status"), "passed")
    _require_equal("Phase-A audit status", audit.get("status"), "passed")
    _require_equal("Phase-A audit errors", audit.get("errors"), [])
    _require_equal("Phase-A audit config hash", audit.get("config_hash"), eval_hash)
    _require_equal("Phase-A marker config hash", marker.get("config_hash"), eval_hash)
    _require_equal(
        "Phase-A marker audit hash",
        marker.get("completion_audit_sha256"),
        file_sha256(audit_path),
    )
    counts = dict(audit.get("counts", {}))
    for label, actual, expected in (
        ("trained adapters", counts.get("trained_adapters"), 9),
        ("training seeds", counts.get("training_seeds"), 3),
        ("evaluated models", counts.get("evaluated_models"), 10),
        ("evaluation questions", counts.get("evaluation_questions"), 1269),
    ):
        _require_equal(f"Phase-A {label}", actual, expected)

    frozen = {key: value for key, value in template.items() if key != "phase_a_parent_spec"}
    frozen["template_config_path"] = str(template_path)
    frozen["template_config_sha256"] = file_sha256(template_path)
    frozen["template_protocol_sha256"] = canonical_sha256(template)
    frozen["phase_a_evidence"] = {
        "status": "passed",
        "completion_marker_path": str(marker_path),
        "completion_marker_sha256": file_sha256(marker_path),
        "completion_audit_path": str(audit_path),
        "completion_audit_sha256": file_sha256(audit_path),
        "frozen_eval_config_path": str(eval_config_path),
        "frozen_eval_config_sha256": file_sha256(eval_config_path),
        "frozen_eval_protocol_sha256": eval_hash,
        "trained_adapter_count": 9,
        "training_seed_count": 3,
        "evaluation_question_count": 1269,
        "gate_policy": "completion_only_no_hyperparameter_selection",
    }
    validate_protocol(frozen, require_frozen=True)
    frozen["preseal_protocol_sha256"] = canonical_sha256(frozen)
    _write_json(output_path, frozen)
    frozen_hash = canonical_sha256(frozen)
    overlay = {
        "parent_config_sha256": frozen_hash,
        "training_overrides": {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 1,
        },
        "rationale": (
            "Reuse the registered batch-size-4 correction for all 36 main-matrix adapters; "
            "no setting is selected from Phase-A accuracy."
        ),
    }
    _write_json(overlay_path, overlay)
    print(
        json.dumps(
            {
                "status": "complete",
                "frozen_config": str(output_path),
                "frozen_config_sha256": file_sha256(output_path),
                "frozen_protocol_sha256": frozen_hash,
                "training_overlay": str(overlay_path),
                "training_overlay_sha256": file_sha256(overlay_path),
            },
            indent=2,
        )
    )


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON config must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r} actual={actual!r}")


if __name__ == "__main__":
    main()
