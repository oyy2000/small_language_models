#!/usr/bin/env python3
"""Register base, matched SFT, and matched-teacher KD models for evaluation."""

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
from length_budget_distill.multiteacher_kd import build_multiteacher_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--sft-manifest-glob", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    registry = build_multiteacher_model_registry(
        read_json(args.protocol),
        sft_manifest_glob=args.sft_manifest_glob,
        output_path=args.output_json,
    )
    logging.info("multiteacher_model_registry_complete models=%d", registry["model_count"])


if __name__ == "__main__":
    main()
