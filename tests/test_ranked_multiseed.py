from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from length_budget_distill.ranked_multiseed import (
    expected_run_names,
    manifest_field_equal,
    ordered_run_specs,
    validate_training_scope,
)
from length_budget_distill.ranked_multiseed_analysis import (
    crossed_seed_problem_bootstrap,
    seed_problem_effects,
)
from length_budget_distill.ranked_multiseed_evaluation import (
    model_id,
    training_run_field_equal,
)


class RankedMultiseedProtocolTest(unittest.TestCase):
    def test_registered_scope_yields_six_new_runs(self) -> None:
        config = {
            "protocol_variant": "comparative_multiseed_ranked_length_sft_extension",
            "training_scope": {
                "labels": ["short", "medium", "long"],
                "mode": "equal_example",
                "training_seeds": [42, 73],
                "expected_examples_per_run": 881,
                "run_count": 6,
            },
        }
        scope = validate_training_scope(config)
        specs = ordered_run_specs("qwen2p5_7b", scope["training_seeds"])
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            {spec["run_name"] for spec in specs},
            expected_run_names("qwen2p5_7b", [42, 73]),
        )
        self.assertNotIn(
            "equal_example__qwen2p5_7b__relative_short__seed_17",
            expected_run_names("qwen2p5_7b", [42, 73]),
        )

    def test_seed_17_cannot_be_retrained(self) -> None:
        config = {
            "protocol_variant": "comparative_multiseed_ranked_length_sft_extension",
            "training_scope": {
                "labels": ["short", "medium", "long"],
                "mode": "equal_example",
                "training_seeds": [17, 42],
                "expected_examples_per_run": 881,
                "run_count": 6,
            },
        }
        with self.assertRaisesRegex(ValueError, "Seed 17"):
            validate_training_scope(config)

    def test_model_identity_includes_rank_and_seed(self) -> None:
        self.assertEqual(
            model_id("qwen2p5_7b", "relative_medium", 42),
            "qwen2p5_7b__relative_medium__seed_42",
        )

    def test_manifest_seed_comparison_accepts_marker_text(self) -> None:
        self.assertTrue(manifest_field_equal("seed", "42", 42))
        self.assertFalse(manifest_field_equal("seed", "73", 42))
        self.assertFalse(manifest_field_equal("seed", "not-an-integer", 42))
        self.assertFalse(manifest_field_equal("n", "881", 881))

    def test_training_evidence_normalizes_seed_and_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            relative = "checkpoints/example"
            absolute = str(project / relative)
            self.assertTrue(training_run_field_equal(project, "seed", "42", 42))
            self.assertTrue(
                training_run_field_equal(project, "output_dir", relative, absolute)
            )
            self.assertFalse(
                training_run_field_equal(project, "output_dir", relative, project / "other")
            )

    def test_crossed_bootstrap_preserves_seed_and_problem_units(self) -> None:
        predictions = {
            17: {
                "relative_short": {
                    "a": {"is_correct": True},
                    "b": {"is_correct": True},
                    "c": {"is_correct": False},
                },
                "relative_long": {
                    "a": {"is_correct": False},
                    "b": {"is_correct": False},
                    "c": {"is_correct": False},
                },
            },
            42: {
                "relative_short": {
                    "a": {"is_correct": True},
                    "b": {"is_correct": False},
                    "c": {"is_correct": True},
                },
                "relative_long": {
                    "a": {"is_correct": False},
                    "b": {"is_correct": False},
                    "c": {"is_correct": False},
                },
            },
            73: {
                "relative_short": {
                    "a": {"is_correct": True},
                    "b": {"is_correct": True},
                    "c": {"is_correct": True},
                },
                "relative_long": {
                    "a": {"is_correct": False},
                    "b": {"is_correct": False},
                    "c": {"is_correct": False},
                },
            },
        }
        effects = seed_problem_effects(
            predictions,
            "relative_short",
            "relative_long",
        )
        result = crossed_seed_problem_bootstrap(effects, samples=200, seed=9)
        self.assertAlmostEqual(result["estimate"], 7.0 / 9.0)
        self.assertEqual(result["seed_count"], 3)
        self.assertEqual(result["problem_count"], 3)
        self.assertEqual(result["resampling_units"], ["training_seed", "paired_problem"])


if __name__ == "__main__":
    unittest.main()
