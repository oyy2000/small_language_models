#!/usr/bin/env python3
"""Audit and seal the exploratory multi-benchmark, multi-teacher KD pilot."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.multiteacher_kd import audit_multiteacher_kd_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json",
    )
    parser.add_argument(
        "--output-root",
        default="results/capacity_length_multibench_multiteacher_kd_pilot_v1",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/capacity_length_multibench_multiteacher_kd_pilot_v1/pilot",
    )
    parser.add_argument(
        "--figure-root",
        default="figures/capacity_length_multibench_multiteacher_kd_pilot_v1",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    audit = audit_multiteacher_kd_completion(
        config_path=args.config,
        output_root=args.output_root,
        checkpoint_root=args.checkpoint_root,
        figure_root=args.figure_root,
        output_json=args.output_json,
    )
    logging.info(
        "multiteacher_kd_audit status=%s errors=%d",
        audit["status"],
        len(audit["errors"]),
    )
    if audit["status"] != "passed":
        raise SystemExit(f"Completion audit failed: {args.output_json}")


if __name__ == "__main__":
    main()
