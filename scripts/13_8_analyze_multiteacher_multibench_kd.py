#!/usr/bin/env python3
"""Analyze matched SFT versus logit-KD across teachers, lengths, and benchmarks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import read_json
from length_budget_distill.multiteacher_analysis import analyze_multiteacher_kd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = analyze_multiteacher_kd(
        read_json(args.protocol),
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    logging.info(
        "multiteacher_kd_analysis_complete models=%d comparisons=%d output=%s",
        payload["model_count"],
        payload["paired_comparison_count"],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
