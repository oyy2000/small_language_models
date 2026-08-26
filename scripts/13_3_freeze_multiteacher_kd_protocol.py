#!/usr/bin/env python3
"""Freeze mixed-data hashes and inherit selected equal-token KD hyperparameters."""

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
from length_budget_distill.multiteacher_kd import freeze_multiteacher_kd_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json",
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frozen = freeze_multiteacher_kd_protocol(
        load_config(args.config),
        config_path=args.config,
        dataset_manifest_path=args.dataset_manifest,
        output_path=args.output_json,
    )
    logging.info(
        "multiteacher_kd_protocol_frozen conditions=%d alpha=%s temperature=%s output=%s",
        len(frozen["conditions"]),
        frozen["kd"]["alpha"],
        frozen["kd"]["temperature"],
        args.output_json,
    )


if __name__ == "__main__":
    main()
