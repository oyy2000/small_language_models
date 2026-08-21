#!/usr/bin/env python3
"""Tests for splitting merged JSONL records by budget."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.budget_split import split_records_by_budget
from length_budget_distill.records import read_jsonl


class BudgetSplitTest(unittest.TestCase):
    def write_rows(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_split_sft_records_by_metadata_budget(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "sft_merged.jsonl"
            rows = [
                {"id": "p1:small", "metadata": {"budget_name": "small"}},
                {"id": "p1:medium", "metadata": {"budget_name": "medium"}},
                {"id": "p1:large", "metadata": {"budget_name": "large"}},
                {"id": "p2:small", "metadata": {"budget_name": "small"}},
            ]
            self.write_rows(input_path, rows)

            counts = split_records_by_budget(input_path, tmp_path, output_prefix="sft")

            self.assertEqual(counts, {"small": 2, "medium": 1, "large": 1})
            self.assertEqual(
                [row["id"] for row in read_jsonl(tmp_path / "sft_small.jsonl")],
                ["p1:small", "p2:small"],
            )
            self.assertEqual(
                [row["id"] for row in read_jsonl(tmp_path / "sft_medium.jsonl")],
                ["p1:medium"],
            )
            self.assertEqual(
                [row["id"] for row in read_jsonl(tmp_path / "sft_large.jsonl")],
                ["p1:large"],
            )

    def test_split_trace_records_by_top_level_budget(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "traces_merged.jsonl"
            rows = [
                {"trace_id": "p1:small", "budget_name": "small"},
                {"trace_id": "p1:medium", "budget_name": "medium"},
                {"trace_id": "p1:large", "budget_name": "large"},
            ]
            self.write_rows(input_path, rows)

            counts = split_records_by_budget(input_path, tmp_path, output_prefix="traces")

            self.assertEqual(counts, {"small": 1, "medium": 1, "large": 1})
            self.assertEqual(
                [row["trace_id"] for row in read_jsonl(tmp_path / "traces_large.jsonl")],
                ["p1:large"],
            )

    def test_split_records_by_budget_rejects_unknown_budget(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "sft_merged.jsonl"
            self.write_rows(input_path, [{"id": "p1:extra", "metadata": {"budget_name": "extra"}}])

            with self.assertRaisesRegex(ValueError, "Unexpected budget names"):
                split_records_by_budget(input_path, tmp_path)


if __name__ == "__main__":
    unittest.main()
