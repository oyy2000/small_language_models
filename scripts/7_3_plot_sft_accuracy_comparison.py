#!/usr/bin/env python3
"""Plot equal-example and equal-token SFT accuracy from analysis metrics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.sft_accuracy_plot import plot_accuracy, read_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Analysis run_metrics.csv")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path without an extension; both PNG and PDF are written",
    )
    parser.add_argument("--seed", type=int, default=None, help="Training seed to plot")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows, base_row, seed = read_metrics(args.input_csv, args.seed)
    logging.info(
        "Input=%s factorial_rows=%d seed=%d n=%d base_accuracy=%.6f",
        args.input_csv,
        len(rows),
        seed,
        rows[0]["n"],
        base_row["accuracy"],
    )
    plot_accuracy(rows, base_row, seed, args.output_prefix, args.dpi)


if __name__ == "__main__":
    main()
