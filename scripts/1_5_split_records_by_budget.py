#!/usr/bin/env python3
"""Split merged JSONL records into one file per length budget."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.budget_split import (
    DEFAULT_BUDGET_NAMES,
    parse_budget_names,
    split_records_by_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/real_length_budget/sft_merged.jsonl",
        help="Merged JSONL path. Records may use top-level budget_name or metadata.budget_name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for split JSONL files. Defaults to the input parent directory.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(DEFAULT_BUDGET_NAMES),
        help="Comma-separated budget names to materialize.",
    )
    parser.add_argument(
        "--output-prefix",
        default="sft",
        help="Output file prefix. The script writes '<prefix>_<budget>.jsonl'.",
    )
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="Skip records whose budget name is not listed in --budgets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    budget_names = parse_budget_names(args.budgets)

    logging.info("input=%s", input_path)
    logging.info("output_dir=%s", output_dir)
    logging.info("budgets=%s", ",".join(budget_names))
    logging.info("output_prefix=%s", args.output_prefix)
    counts = split_records_by_budget(
        input_path=input_path,
        output_dir=output_dir,
        budget_names=budget_names,
        output_prefix=args.output_prefix,
        ignore_unknown=args.ignore_unknown,
    )
    for budget_name in budget_names:
        output_path = output_dir / f"{args.output_prefix}_{budget_name}.jsonl"
        logging.info("wrote_%s=%d path=%s", budget_name, counts[budget_name], output_path)


if __name__ == "__main__":
    main()
