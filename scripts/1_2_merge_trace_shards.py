#!/usr/bin/env python3
"""Merge generated length-budget trace shards."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.records import read_jsonl, trace_from_dict, write_jsonl
from length_budget_distill.sft_format import trace_to_sft_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="Glob pattern for raw shard JSONL files.")
    parser.add_argument("--output", required=True, help="Merged raw trace JSONL path.")
    parser.add_argument("--sft-output", default=None, help="Optional merged SFT JSONL path.")
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include incorrect traces in SFT output. By default only verified-correct traces are exported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not input_paths:
        raise FileNotFoundError(f"No shards matched: {args.input_glob}")

    merged_by_id: Dict[str, dict] = {}
    for path in input_paths:
        for item in read_jsonl(path):
            merged_by_id[item["trace_id"]] = item

    merged: List[dict] = [merged_by_id[key] for key in sorted(merged_by_id)]
    raw_count = write_jsonl(Path(args.output), merged)
    logging.info("merged_shards=%d wrote_raw=%d path=%s", len(input_paths), raw_count, args.output)

    if args.sft_output:
        traces = [trace_from_dict(item) for item in merged]
        sft_records = [
            trace_to_sft_record(trace)
            for trace in traces
            if trace.is_correct or args.include_incorrect
        ]
        sft_count = write_jsonl(Path(args.sft_output), sft_records)
        logging.info("wrote_sft=%d path=%s", sft_count, args.sft_output)


if __name__ == "__main__":
    main()

