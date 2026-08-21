#!/usr/bin/env python3
"""Audit and seal one capacity-length experiment stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.completion_audit import audit_capacity_length_completion
from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument("--stage", choices=["smoke", "formal"], required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_capacity_length_completion(
        config=load_config(args.config),
        stage=args.stage,
        stage_root=args.stage_root,
        project_root=PROJECT_ROOT,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(f"Completion audit failed with {len(report['errors'])} errors: {output_path}")
    marker_path = Path(args.stage_root) / f"{args.stage.upper()}_COMPLETE"
    marker_path.write_text(
        f"config_hash={report['config_hash']}\n"
        f"completion_audit_sha256={file_sha256(output_path)}\n"
        f"trained_adapters={report['counts']['trained_adapters']}\n"
        f"evaluation_runs={report['counts']['evaluation_runs']}\n",
        encoding="utf-8",
    )
    print(f"COMPLETION_AUDIT_PASSED stage={args.stage} report={output_path} marker={marker_path}")


if __name__ == "__main__":
    main()
