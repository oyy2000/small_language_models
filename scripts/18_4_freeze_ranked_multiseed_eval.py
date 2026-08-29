#!/usr/bin/env python3
"""Freeze the seed-17/42/73 evaluation protocol after both training parents pass audit."""

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

from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.ranked_multiseed_evaluation import (
    freeze_parent_training,
    validate_all_parent_trainings,
    validate_eval_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-config",
        default="configs/capacity_length_ranked_sampling_7b_eval_multiseed_v1.json",
    )
    parser.add_argument("--output-config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = _resolve(args.template_config)
    output_path = _resolve(args.output_config)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen evaluation config: {output_path}")
    template = _read_json(template_path)
    validate_eval_settings(template)
    specs = template.get("parent_training_specs", [])
    if not isinstance(specs, list) or len(specs) != 2:
        raise ValueError("Evaluation template must contain exactly two training parent specs.")
    frozen = {
        key: value
        for key, value in template.items()
        if key != "parent_training_specs"
    }
    frozen["template_config_path"] = str(template_path)
    frozen["template_config_sha256"] = file_sha256(template_path)
    frozen["template_protocol_sha256"] = canonical_sha256(template)
    frozen["parent_trainings"] = [
        freeze_parent_training(spec, PROJECT_ROOT)
        for spec in specs
    ]
    validate_all_parent_trainings(frozen, PROJECT_ROOT)
    frozen["frozen_protocol_sha256"] = canonical_sha256(frozen)
    _write_json(output_path, frozen)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_config": str(output_path),
                "config_sha256": file_sha256(output_path),
                "protocol_sha256": canonical_sha256(frozen),
                "adapter_count": frozen["evaluation"]["expected_adapter_count"],
            },
            indent=2,
        )
    )


def _resolve(value: str) -> Path:
    path = Path(value)
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


if __name__ == "__main__":
    main()
