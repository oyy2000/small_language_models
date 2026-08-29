#!/usr/bin/env python3
"""Materialize the balanced three-node launch plan for the 36-adapter matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.ranked_multiteacher import (
    ordered_matrix_runs,
    validate_launcher_assignment,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    output_path = _resolve(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite launcher plan: {output_path}")
    config = _read_json(config_path)
    validate_protocol(config, require_frozen=True)
    runs = ordered_matrix_runs()
    summary = validate_launcher_assignment(runs)
    payload = {
        "status": "complete",
        "evidence_class": "operational_balance_plan_not_scientific_protocol",
        "config_path": str(config_path),
        "config_hash": canonical_sha256(config),
        "config_file_sha256": file_sha256(config_path),
        "assignment_sha256": canonical_sha256(runs),
        "materializer_source_sha256": file_sha256(Path(__file__).resolve()),
        "matrix_protocol_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher.py"
        ),
        "launcher_selection_source_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/factorial.py"
        ),
        "summary": summary,
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path),
                "sha256": file_sha256(output_path),
                "assignment_sha256": payload["assignment_sha256"],
                "run_count": len(runs),
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
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


if __name__ == "__main__":
    main()
