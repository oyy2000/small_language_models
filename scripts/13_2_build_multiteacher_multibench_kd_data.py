#!/usr/bin/env python3
"""Build globally equal-token GSM8K+MATH data for all teacher-length conditions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.multiteacher_kd import build_multiteacher_equal_token_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = build_multiteacher_equal_token_data(
        load_config(args.config),
        output_dir=args.output_dir,
    )
    logging.info(
        "multiteacher_multibench_data_complete runs=%d math_target=%d output=%s",
        len(manifest["runs"]),
        int(manifest["math_global_equal_token_target"]),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
