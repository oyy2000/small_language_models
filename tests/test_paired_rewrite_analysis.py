#!/usr/bin/env python3
"""Tests for paired-rewrite recipe selection and advancement gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.paired_rewrite_analysis import (
    advancement_gate,
    gradient_clip_rate,
    select_shared_recipe,
)


class PairedRewriteAnalysisTest(unittest.TestCase):
    def test_clip_rate_uses_only_rows_through_snapshot(self) -> None:
        rate, count = gradient_clip_rate(
            [
                {"epoch": 0.2, "grad_norm": 0.5},
                {"epoch": 0.4, "grad_norm": 1.2},
                {"epoch": 0.8, "grad_norm": 9.0},
            ],
            through_epoch=0.5,
            max_grad_norm=1.0,
        )
        self.assertEqual(count, 2)
        self.assertEqual(rate, 0.5)

    def test_selection_uses_stability_within_accuracy_tolerance(self) -> None:
        rows = []
        for condition in ("standard", "rewrite"):
            rows.extend(
                [
                    {
                        "condition": condition,
                        "learning_rate": 1e-5,
                        "epoch": 1.0,
                        "accuracy": 0.70,
                        "clip_rate": 0.4,
                    },
                    {
                        "condition": condition,
                        "learning_rate": 5e-6,
                        "epoch": 0.5,
                        "accuracy": 0.697,
                        "clip_rate": 0.0,
                    },
                ]
            )
        selected = select_shared_recipe(
            rows,
            conditions=["standard", "rewrite"],
            accuracy_tie_pp=0.5,
        )
        self.assertEqual(selected["learning_rate"], 5e-6)
        self.assertEqual(selected["epoch"], 0.5)

    def test_advancement_requires_accuracy_and_length(self) -> None:
        passed = advancement_gate(
            baseline_accuracy=0.70,
            candidate_accuracy=0.685,
            baseline_output_tokens=200,
            candidate_output_tokens=150,
            max_accuracy_drop_pp=2.0,
            max_output_token_ratio=0.8,
        )
        failed = advancement_gate(
            baseline_accuracy=0.70,
            candidate_accuracy=0.67,
            baseline_output_tokens=200,
            candidate_output_tokens=150,
            max_accuracy_drop_pp=2.0,
            max_output_token_ratio=0.8,
        )
        self.assertEqual(passed["status"], "pass")
        self.assertEqual(failed["status"], "fail")


if __name__ == "__main__":
    unittest.main()
