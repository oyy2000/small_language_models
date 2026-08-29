#!/usr/bin/env python3
"""Materialize three hash-bound generation configs for the new main-matrix teachers."""

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
from length_budget_distill.ranked_multiteacher import (
    TEACHER_NAMES,
    generation_config_for_teacher,
    validate_protocol,
)
from length_budget_distill.ranked_sampling import validate_ranked_sampling_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "generation_config_manifest.json"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite materialized configs: {output_dir}")
    frozen = _read_json(config_path)
    validate_protocol(frozen, require_frozen=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for teacher_name in TEACHER_NAMES:
        if teacher_name == "qwen2p5_7b":
            continue
        payload = generation_config_for_teacher(
            frozen,
            teacher_name,
            protocol_path=str(config_path),
            protocol_file_sha256=file_sha256(config_path),
        )
        validate_ranked_sampling_config(payload)
        path = output_dir / f"{teacher_name}.json"
        _write_json(path, payload)
        entries.append(
            {
                "teacher_name": teacher_name,
                "config_path": str(path),
                "canonical_config_sha256": canonical_sha256(payload),
                "config_file_sha256": file_sha256(path),
            }
        )
    _write_json(
        manifest_path,
        {
            "status": "complete",
            "parent_config_path": str(config_path),
            "parent_config_sha256": file_sha256(config_path),
            "parent_protocol_sha256": canonical_sha256(frozen),
            "teacher_count": len(entries),
            "teachers": entries,
        },
    )
    print(json.dumps({"status": "complete", "manifest": str(manifest_path)}, indent=2))


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
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
