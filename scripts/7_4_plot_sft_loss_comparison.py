#!/usr/bin/env python3
"""Plot short/medium/long completion-only SFT loss from formal training logs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.sft_loss_plot import plot_loss_curves, read_loss_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-dir",
        type=Path,
        required=True,
        help="Formal training directory containing configs/ and logs/",
    )
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
    runs, seed = read_loss_runs(args.training_dir, args.seed)
    logging.info("Input=%s factorial_runs=%d seed=%d", args.training_dir, len(runs), seed)
    for run in runs:
        logging.info(
            "run=%s n=%d supervised_tokens=%d steps=%d logged_points=%d train_loss=%.6f",
            run["run_name"],
            run["n_examples"],
            run["supervised_tokens"],
            run["total_steps"],
            len(run["points"]),
            run["train_loss"],
        )
    plot_loss_curves(runs, seed, args.output_prefix, args.dpi)


if __name__ == "__main__":
    main()
