from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.ranked_evaluation import (
    completed_evaluation_evidence,
    validate_evaluation_protocol,
)
from length_budget_distill.ranked_evaluation_analysis import (
    paired_contrast,
    summarize_predictions,
)


class RankedEvaluationTest(unittest.TestCase):
    def test_locked_protocol_validation(self) -> None:
        config = {
            "protocol_variant": "revised_formal_single_seed_ranked_length_evaluation",
            "dataset": {
                "source": "hf_dataset",
                "dataset_name": "openai/gsm8k",
                "dataset_config": "main",
            },
            "student": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
            "evaluation": {
                "dataset_split": "test",
                "start_index": 50,
                "limit": 1269,
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
                "batch_size": 32,
                "include_base_model": True,
                "expected_adapter_count": 3,
                "expected_run_count": 4,
            },
        }
        validate_evaluation_protocol(config)
        config["evaluation"]["start_index"] = 0
        with self.assertRaises(ValueError):
            validate_evaluation_protocol(config)

    def test_prediction_summary_and_paired_contrast(self) -> None:
        left_rows = [
            {"problem_id": "a", "is_correct": True, "output_token_count": 10, "predicted_answer": "1"},
            {"problem_id": "b", "is_correct": True, "output_token_count": 20, "predicted_answer": "2"},
            {"problem_id": "c", "is_correct": False, "output_token_count": 512, "predicted_answer": None},
        ]
        right_rows = [
            {"problem_id": "a", "is_correct": False, "output_token_count": 15, "predicted_answer": "0"},
            {"problem_id": "b", "is_correct": True, "output_token_count": 25, "predicted_answer": "2"},
            {"problem_id": "c", "is_correct": True, "output_token_count": 30, "predicted_answer": "3"},
        ]
        summary = summarize_predictions(left_rows)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["max_token_hit_count"], 1)
        self.assertEqual(summary["answer_extraction_failures"], 1)
        contrast = paired_contrast(
            {row["problem_id"]: row for row in left_rows},
            {row["problem_id"]: row for row in right_rows},
            bootstrap_samples=200,
            bootstrap_seed=17,
        )
        self.assertEqual(contrast["left_only_correct"], 1)
        self.assertEqual(contrast["right_only_correct"], 1)
        self.assertAlmostEqual(contrast["accuracy_difference"], 0.0)

    def test_completed_evaluation_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "predictions.jsonl"
            summary_path = root / "summary.json"
            rows = [
                {"problem_id": "hf-000050", "is_correct": True},
                {"problem_id": "hf-000051", "is_correct": False},
            ]
            prediction_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "split": "test",
                        "start_index": 50,
                        "n": 2,
                        "correct": 1,
                        "accuracy": 0.5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            evidence = completed_evaluation_evidence(
                prediction_path,
                summary_path,
                expected_n=2,
                expected_start_index=50,
                expected_split="test",
            )
            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence["correct"], 1)
            self.assertEqual(evidence["problem_ids"], ["hf-000050", "hf-000051"])


if __name__ == "__main__":
    unittest.main()
