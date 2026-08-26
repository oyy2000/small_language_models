#!/usr/bin/env python3
"""Sanity tests for paired, structure-preserving rationale rewrites."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.paired_rewrite import (
    assess_rewrite_candidate,
    extract_equation_checks,
    minimum_target_token_count,
    paired_sft_record,
    select_adaptive_rewrite,
    stable_rewrite_seed,
    target_token_count,
)
from length_budget_distill.tokenization import WhitespaceTokenCounter


class PairedRewriteTest(unittest.TestCase):
    def test_seed_and_target_are_deterministic(self) -> None:
        first = stable_rewrite_seed(17, "source-1", "rewrite_80")
        second = stable_rewrite_seed(17, "source-1", "rewrite_80")
        self.assertEqual(first, second)
        self.assertNotEqual(first, stable_rewrite_seed(17, "source-1", "rewrite_65"))
        self.assertEqual(target_token_count(101, 0.8), 81)
        self.assertEqual(minimum_target_token_count(81, 0.75), 61)

    def test_numeric_equation_checker_rejects_wrong_arithmetic(self) -> None:
        checks = extract_equation_checks("First compute 12 * 4 = 48; then 48 + 2 = 51.")
        self.assertEqual(len(checks), 2)
        self.assertTrue(checks[0].valid)
        self.assertFalse(checks[1].valid)

    def test_numeric_equation_checker_skips_symbolic_equations(self) -> None:
        checks = extract_equation_checks("Let 15x + 5 = 50, so x = 3. Also 6C = 24.")
        self.assertEqual(checks, [])

    def test_numeric_equation_checker_ignores_markdown_bullet_sign(self) -> None:
        checks = extract_equation_checks("- Remaining butter: 10 - 5 = 5 kg.")
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].valid)

    def test_numeric_equation_checker_skips_unit_and_prose_equivalences(self) -> None:
        checks = extract_equation_checks(
            "Half of 50 = 0.50 * 50 = 25; 1 gallon = 8 pints."
        )
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].valid)

    def test_numeric_equation_checker_handles_latex_percent(self) -> None:
        checks = extract_equation_checks(r"\[ 60\% \times 90 = 0.60 \times 90 = 54 \]")
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(check.valid for check in checks))

    def test_candidate_requires_answer_arithmetic_and_step_coverage(self) -> None:
        counter = WhitespaceTokenCounter()
        valid = assess_rewrite_candidate(
            "Compute 12 * 4 = 48. Add 2: 48 + 2 = 50.\nAnswer: 50",
            gold_answer="50",
            source_tokens=30,
            minimum_tokens=10,
            target_tokens=20,
            required_step_values=["48"],
            token_counter=counter,
            candidate_index=0,
        )
        invalid = assess_rewrite_candidate(
            "Compute 12 * 4 = 47.\nAnswer: 50",
            gold_answer="50",
            source_tokens=30,
            minimum_tokens=10,
            target_tokens=20,
            required_step_values=["48"],
            token_counter=counter,
            candidate_index=1,
        )
        self.assertTrue(valid["within_target"])
        self.assertFalse(invalid["structurally_valid"])
        self.assertFalse(invalid["equation_valid"])
        undershot = assess_rewrite_candidate(
            valid["solution"],
            gold_answer="50",
            source_tokens=30,
            minimum_tokens=16,
            target_tokens=20,
            required_step_values=["48"],
            token_counter=counter,
            candidate_index=2,
        )
        self.assertTrue(undershot["structurally_valid"])
        self.assertFalse(undershot["in_length_band"])
        self.assertFalse(undershot["within_target"])

    def test_adaptive_selection_uses_explicit_fallback(self) -> None:
        selected = select_adaptive_rewrite(
            {
                "rewrite_65": [{"candidate_index": 0, "within_target": False}],
                "rewrite_80": [
                    {
                        "candidate_index": 1,
                        "within_target": True,
                        "target_tokens": 80,
                        "actual_tokens": 78,
                        "equation_checked_count": 2,
                    }
                ],
            },
            preferred_ratio="rewrite_65",
            fallback_ratios=["rewrite_80"],
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["selected_ratio_name"], "rewrite_80")
        self.assertEqual(selected["fallback_level"], 1)

    def test_sft_record_keeps_prompt_and_source_binding(self) -> None:
        source = {
            "id": "trace-1",
            "prompt": "Problem:\n2 + 3?",
            "completion": "2 + 3 = 5.\nAnswer: 5",
            "metadata": {"problem_id": "gsm8k-1"},
        }
        row = paired_sft_record(
            source,
            condition="rewrite_80",
            completion="2 + 3 = 5.\nAnswer: 5",
            source_sha256="abc",
            selection={"selected_ratio_name": "rewrite_80"},
        )
        self.assertEqual(row["prompt"], source["prompt"])
        self.assertEqual(row["metadata"]["source_trace_id"], "trace-1")
        self.assertEqual(row["metadata"]["source_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
