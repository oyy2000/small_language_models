from __future__ import annotations

import importlib.util
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256
from length_budget_distill.opd import (
    binary_auc,
    build_bounded_concise_prompt,
    clipped_opd_loss,
    preflight_summary,
    protocol_hash,
    reference_length_bounds,
    sampled_token_advantage,
    topk_overlap,
    validate_opd_protocol,
    validate_reference_manifest,
)
from length_budget_distill.opd_analysis import (
    completed_opd_evaluation,
    opd_advancement_decision,
    paired_opd_contrast,
    summarize_opd_predictions,
)
from length_budget_distill.student_prompts import build_student_math_prompt


class OPDTest(unittest.TestCase):
    def test_registered_protocol_is_pure_dual_prompt_opd(self) -> None:
        protocol = load_config(
            str(PROJECT_ROOT / "configs/capacity_length_opd_prompt_pilot_v1.json")
        )
        validate_opd_protocol(protocol)
        self.assertEqual(protocol["arms"], ["standard_prompt", "bounded_concise_prompt"])
        self.assertEqual(protocol["preflight"]["num_shards"], 2)
        self.assertEqual(protocol["objective"]["teacher_context_mode"], "common_standard_prompt")
        self.assertFalse(protocol["objective"]["correctness_reward"])
        self.assertFalse(protocol["objective"]["length_reward"])
        self.assertFalse(protocol["objective"]["value_head"])

    def test_registered_protocol_rejects_auxiliary_reward(self) -> None:
        protocol = load_config(
            str(PROJECT_ROOT / "configs/capacity_length_opd_prompt_pilot_v1.json")
        )
        protocol["objective"]["correctness_reward"] = True
        with self.assertRaisesRegex(ValueError, "forbids"):
            validate_opd_protocol(protocol)

    def test_relative_length_bounds_are_ratio_based_and_clamped(self) -> None:
        self.assertEqual(reference_length_bounds(100), (96, 96))
        self.assertEqual(reference_length_bounds(200), (140, 180))
        self.assertEqual(reference_length_bounds(300), (210, 256))
        self.assertEqual(reference_length_bounds(500), (256, 256))

    def test_concise_prompt_preserves_reasoning_priority(self) -> None:
        prompt = build_bounded_concise_prompt("What is 2 + 2?", 96, 128)
        self.assertIn("Aim for 96 to 128 solution tokens", prompt)
        self.assertIn("do not give an answer-only response", prompt)
        self.assertIn("preserve the reasoning", prompt)
        self.assertTrue(prompt.endswith("Answer: <final answer>."))

    def test_auc_and_preflight_gate(self) -> None:
        self.assertEqual(binary_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)
        rows = []
        for arm in ("standard_prompt", "bounded_concise_prompt"):
            for index in range(10):
                rows.append(
                    {
                        "arm": arm,
                        "mean_advantage": float(index),
                        "is_correct": index >= 5,
                        "in_length_band": arm == "standard_prompt" or index < 7,
                    }
                )
        summary = preflight_summary(rows)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["concise_in_band_rate"], 0.7)
        rows[-1]["in_length_band"] = False
        rows[-4]["in_length_band"] = False
        self.assertEqual(preflight_summary(rows)["status"], "failed")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_sampled_token_advantage_and_clipped_surrogate(self) -> None:
        import torch

        old = torch.tensor([-2.0, -1.0])
        teacher = torch.tensor([-1.0, -2.0])
        advantages = sampled_token_advantage(teacher, old)
        self.assertTrue(torch.equal(advantages, torch.tensor([1.0, -1.0])))
        new = old.clone().requires_grad_(True)
        loss, metrics = clipped_opd_loss(new, old, advantages, clip_ratio=0.2)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(metrics["mean_ratio"]), 1.0, places=6)
        loss.backward()
        self.assertIsNotNone(new.grad)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_topk_overlap_is_positionwise_intersection_fraction(self) -> None:
        import torch

        student = torch.tensor([[1, 2], [3, 4]])
        teacher = torch.tensor([[2, 5], [3, 4]])
        self.assertAlmostEqual(topk_overlap(student, teacher), 0.75)

    def test_reference_manifest_validates_hash_and_bounds(self) -> None:
        protocol = {
            "splits": {"training": {"start_index": 0, "limit": 2}},
            "concise_prompt": {
                "lower_ratio": 0.70,
                "upper_ratio": 0.90,
                "minimum_tokens": 96,
                "maximum_tokens": 256,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "references.jsonl"
            rows = []
            for index in range(2):
                question = f"What is {index} + 1?"
                rows.append(
                    {
                    "problem_id": f"p{index}",
                    "question": question,
                    "gold_answer": str(index + 1),
                    "source_index": index,
                    "standard_prompt": build_student_math_prompt(question),
                    "concise_prompt": build_bounded_concise_prompt(question, 140, 180),
                    "reference_output_tokens": 200,
                    "concise_lower_tokens": 140,
                    "concise_upper_tokens": 180,
                    "reference_policy": "frozen_qwen2p5_1p5b_instruct_base",
                    "reference_decoding": "greedy",
                    "protocol_hash": protocol_hash(protocol),
                }
                )
            reference_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol_hash": protocol_hash(protocol),
                        "reference_path": str(reference_path),
                        "reference_sha256": file_sha256(reference_path),
                        "record_count": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _manifest, observed = validate_reference_manifest(protocol, manifest_path)
            self.assertEqual(len(observed), 2)

    def test_paired_contrast_and_advancement_gate(self) -> None:
        standard_rows = [
            self._prediction("a", True, 100),
            self._prediction("b", True, 100),
            self._prediction("c", False, 100),
            self._prediction("d", False, 100),
        ]
        concise_rows = [
            self._prediction("a", True, 80),
            self._prediction("b", True, 80),
            self._prediction("c", False, 80),
            self._prediction("d", False, 80),
        ]
        contrast = paired_opd_contrast(
            {row["problem_id"]: row for row in concise_rows},
            {row["problem_id"]: row for row in standard_rows},
            bootstrap_samples=200,
            bootstrap_seed=17,
        )
        self.assertEqual(contrast["accuracy_difference"], 0.0)
        self.assertEqual(contrast["mean_output_token_ratio"], 0.8)
        decision = opd_advancement_decision(
            contrast,
            {
                "accuracy_noninferiority_margin_pp": 1.0,
                "maximum_mean_output_token_ratio": 0.90,
                "maximum_extraction_failure_increase_pp": 1.0,
                "maximum_truncation_increase_pp": 1.0,
            },
        )
        self.assertEqual(decision["classification"], "bounded_concise_more_suitable")

    def test_preflight_merge_requires_complete_two_shard_support(self) -> None:
        config_path = PROJECT_ROOT / "configs/capacity_length_opd_prompt_pilot_v1.json"
        protocol = load_config(str(config_path))
        generator_path = PROJECT_ROOT / "scripts/17_3_preflight_opd_signal.py"
        merge_path = PROJECT_ROOT / "scripts/17_3_merge_opd_preflight.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            for shard_index in range(2):
                suffix = f"shard_{shard_index:05d}_of_00002"
                rollout_path = input_dir / "shards" / f"{suffix}.jsonl.gz"
                manifest_path = input_dir / "manifests" / f"{suffix}.json"
                rollout_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for offset in range(shard_index, 100, 2):
                    for arm in ("standard_prompt", "bounded_concise_prompt"):
                        for candidate_index in range(4):
                            row = {
                                "problem_id": f"problem-{offset}",
                                "source_index": 2000 + offset,
                                "arm": arm,
                                "candidate_index": candidate_index,
                                "mean_advantage": 0.5,
                                "is_correct": candidate_index % 2 == 0,
                                "in_length_band": arm == "standard_prompt" or candidate_index < 3,
                                "output_token_count": 100,
                                "predicted_answer": "1",
                                "completion_token_ids": [1, 2],
                                "old_student_logprobs": [-1.0, -1.0],
                                "teacher_logprobs": [-0.5, -0.5],
                                "advantages": [0.5, 0.5],
                                "teacher_context_mode": "common_standard_prompt",
                                "valid_vocab_size": 151665,
                                "scalar_reward_used": False,
                                "value_head_used": False,
                                "correctness_is_diagnostic_only": True,
                                "length_is_diagnostic_only": True,
                            }
                            if shard_index == 0 and offset == 0 and candidate_index < 2:
                                row["topk_diagnostic"] = {"k": 2}
                            rows.append(row)
                with gzip.open(rollout_path, "wt", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")
                manifest_path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "stage": "preflight_shard",
                            "protocol_hash": protocol_hash(protocol),
                            "shard_index": shard_index,
                            "num_shards": 2,
                            "prompt_count": 50,
                            "rollout_path": str(rollout_path),
                            "rollout_sha256": file_sha256(rollout_path),
                            "rollout_count": len(rows),
                            "source_sha256": file_sha256(generator_path),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            subprocess.run(
                [
                    sys.executable,
                    str(merge_path),
                    "--config",
                    str(config_path),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            marker = json.loads((output_dir / "PREFLIGHT_COMPLETE").read_text(encoding="utf-8"))
            summary = json.loads((output_dir / "preflight_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(marker["status"], "passed")
            self.assertEqual(summary["n"], 800)
            self.assertEqual(summary["concise_in_band_rate"], 0.75)

    def test_completed_evaluation_recomputes_answers_and_metrics(self) -> None:
        protocol = {
            "splits": {
                "primary_evaluation": {
                    "dataset_split": "train",
                    "start_index": 10,
                    "limit": 2,
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "predictions.jsonl"
            summary_path = root / "summary.json"
            rows = [
                {
                    **self._prediction("a", True, 20),
                    "model_id": "base",
                    "split_name": "primary_evaluation",
                    "prediction_text": "Reasoning. Answer: 2",
                    "predicted_answer": "2",
                    "gold_answer": "#### 2",
                    "prompt_mode": "common_standard_prompt",
                },
                {
                    **self._prediction("b", False, 30),
                    "model_id": "base",
                    "split_name": "primary_evaluation",
                    "prediction_text": "Reasoning. Answer: 1",
                    "predicted_answer": "1",
                    "gold_answer": "#### 2",
                    "prompt_mode": "common_standard_prompt",
                },
            ]
            prediction_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = {
                "status": "complete",
                "model_id": "base",
                "split_name": "primary_evaluation",
                "dataset_split": "train",
                "start_index": 10,
                "limit": 2,
                "prompt_mode": "common_standard_prompt",
                "protocol_hash": protocol_hash(protocol),
                "prediction_path": str(prediction_path),
                "prediction_sha256": file_sha256(prediction_path),
                "metrics": summarize_opd_predictions(rows),
            }
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            evidence = completed_opd_evaluation(
                protocol,
                split_name="primary_evaluation",
                model_id="base",
                prediction_path=prediction_path,
                summary_path=summary_path,
            )
            self.assertIsNotNone(evidence)
            assert evidence is not None
            self.assertEqual(evidence["prediction_count"], 2)

    @staticmethod
    def _prediction(problem_id: str, correct: bool, tokens: int) -> dict:
        return {
            "problem_id": problem_id,
            "is_correct": correct,
            "output_token_count": tokens,
            "predicted_answer": "2" if correct else "1",
            "eos_emitted": True,
            "hit_max_new_tokens": False,
        }


if __name__ == "__main__":
    unittest.main()
