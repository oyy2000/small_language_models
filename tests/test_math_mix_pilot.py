#!/usr/bin/env python3
"""Tests for the exploratory MATH-mixed SFT pilot."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import canonical_sha256
from length_budget_distill.math_mix import (
    diagnose_common_support_selection,
    normalized_question_sha256,
    proportional_stratified_sample,
)
from length_budget_distill.verifiers import (
    extract_last_boxed,
    extract_math_final_answer,
    verify_math_answer,
)


class MathMixPilotTest(unittest.TestCase):
    def test_nested_boxed_extraction_preserves_latex(self) -> None:
        text = r"Solution text. \boxed{\left(3, \frac{\pi}{2}\right)}."
        self.assertEqual(extract_last_boxed(text), r"\left(3, \frac{\pi}{2}\right)")
        self.assertEqual(extract_math_final_answer(text), r"\left(3, \frac{\pi}{2}\right)")

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify is not installed")
    def test_math_verifier_accepts_symbolic_equivalence(self) -> None:
        self.assertTrue(verify_math_answer(r"\frac{2}{4}", r"\frac{1}{2}"))
        self.assertTrue(verify_math_answer("p-q", "p - q"))
        self.assertFalse(verify_math_answer(r"\frac{2}{3}", r"\frac{1}{2}"))

    def test_proportional_sampling_is_deterministic_and_covers_strata(self) -> None:
        rows = []
        for subject, level, count in (("algebra", 1, 8), ("algebra", 2, 4), ("geometry", 1, 3)):
            for index in range(count):
                rows.append({"id": f"{subject}-{level}-{index}", "subject": subject, "level": level})
        first = proportional_stratified_sample(
            rows,
            8,
            stratum_fields=("subject", "level"),
            seed=17,
            minimum_per_stratum=1,
        )
        second = proportional_stratified_sample(
            rows,
            8,
            stratum_fields=("subject", "level"),
            seed=17,
            minimum_per_stratum=1,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len({row["id"] for row in first}), 8)
        self.assertEqual({(row["subject"], row["level"]) for row in first}, {
            ("algebra", 1),
            ("algebra", 2),
            ("geometry", 1),
        })

    def test_question_hash_normalizes_whitespace_only(self) -> None:
        self.assertEqual(
            normalized_question_sha256("Find   x.\nNow solve."),
            normalized_question_sha256("Find x. Now solve."),
        )
        self.assertNotEqual(
            normalized_question_sha256("Find x."),
            normalized_question_sha256("Find y."),
        )

    def test_common_support_diagnosis_measures_difficulty_shift(self) -> None:
        source = [
            {"id": "easy", "level": 1, "subject": "algebra", "question_sha256": "a"},
            {"id": "hard", "level": 5, "subject": "geometry", "question_sha256": "b"},
        ]
        evaluation = [
            {"id": "eval", "level": 5, "subject": "Geometry", "question_sha256": "c"},
        ]
        diagnosis = diagnose_common_support_selection(source, ["easy"], evaluation)
        self.assertEqual(diagnosis["source"]["mean_level"], 3.0)
        self.assertEqual(diagnosis["common_support"]["mean_level"], 1.0)
        self.assertEqual(diagnosis["evaluation"]["mean_level"], 5.0)
        self.assertEqual(diagnosis["retention_by_level"][0]["retention_rate"], 1.0)
        self.assertEqual(diagnosis["retention_by_level"][1]["retention_rate"], 0.0)
        self.assertEqual(diagnosis["question_hash_overlap"]["source_vs_evaluation"], 0)

    def test_pilot_overlay_is_bound_and_cardinalities_are_locked(self) -> None:
        config_path = PROJECT_ROOT / "configs/capacity_length_math_mix_pilot_v1.json"
        overlay_path = PROJECT_ROOT / "configs/capacity_length_math_mix_pilot_sft_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        self.assertEqual(overlay["parent_config_sha256"], canonical_sha256(config))
        self.assertEqual(config["math_train_source"]["sample_count"], 1000)
        self.assertEqual(config["generation"]["num_candidates"], 3)
        self.assertEqual(len(config["length_budgets"]), 3)
        self.assertEqual(1000 * 3 * 3, 9000)
        self.assertEqual(config["balancing"]["pilot_min_common_problems"], 300)
        self.assertEqual(
            sum(config["evaluation_suite"][name]["sample_count"] for name in ("gsm8k", "math500", "aime2025")),
            330,
        )


if __name__ == "__main__":
    unittest.main()
