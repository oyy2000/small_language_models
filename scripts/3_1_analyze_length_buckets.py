#!/usr/bin/env python3
"""Summarize length-budget trace outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.analysis import maybe_plot_summary, summarize_by_budget, write_summary_csv, write_summary_json
from length_budget_distill.records import traces_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Merged raw trace JSONL path.")
    parser.add_argument("--output-json", required=True, help="Summary JSON path.")
    parser.add_argument("--output-csv", default=None, help="Optional summary CSV path.")
    parser.add_argument("--plot", default=None, help="Optional plot path. Requires matplotlib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    traces = traces_from_jsonl(Path(args.input))
    summary = summarize_by_budget(traces)
    write_summary_json(Path(args.output_json), summary)
    logging.info("wrote_json=%s", args.output_json)

    if args.output_csv:
        write_summary_csv(Path(args.output_csv), summary)
        logging.info("wrote_csv=%s", args.output_csv)

    if args.plot:
        plotted = maybe_plot_summary(Path(args.plot), summary)
        if plotted:
            logging.info("wrote_plot=%s", args.plot)
        else:
            logging.info("matplotlib is not installed; skipped plot=%s", args.plot)


if __name__ == "__main__":
    main()

