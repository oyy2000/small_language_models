from __future__ import annotations

import json
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    baseline_sft_run_name,
    compliance_gate_satisfied,
    float_slug,
    hybrid_kd_loss,
    invalid_vocab_probability_mass,
    kd_context_fields,
    kd_run_name,
    load_protocol,
    supervision_mode,
    tokenize_completion_record,
    tokenize_kd_record,
    validate_budget_dataset,
)
from length_budget_distill.factorial import file_sha256


class FakeChatTokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        self.assert_tokenize = tokenize
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("Prompt-only formatting must request a generation prompt.")
            return [1, 2, 3]
        if add_generation_prompt:
            raise AssertionError("Full conversation must not request another generation prompt.")
        return [1, 2, 3, 4, 5, 9]


class DualContextFakeChatTokenizer:
    eos_token_id = 9

    def __init__(self, mismatch: bool = False) -> None:
        self.mismatch = mismatch

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        if not tokenize:
            raise AssertionError("Tests require tokenized chat templates.")
        prompt = messages[0]["content"]
        is_teacher = prompt.startswith("teacher")
        prefix = [7, 8, 3] if is_teacher else [1, 2, 3]
        if len(messages) == 1:
            if not add_generation_prompt:
                raise AssertionError("Prompt-only formatting must request a generation prompt.")
            return prefix
        if add_generation_prompt:
            raise AssertionError("Full conversation must not request another generation prompt.")
        targets = [6, 5, 9] if self.mismatch and is_teacher else [4, 5, 9]
        return prefix + targets


class LogitKDTest(unittest.TestCase):
    def test_registered_protocol_loads_and_is_single_seed(self) -> None:
        config = load_protocol(PROJECT_ROOT / "configs/capacity_length_logit_kd_seed17_v1.json")
        self.assertEqual(config["training"]["seed"], 17)
        self.assertEqual(config["models"]["tokenizer"]["expected_length"], 151665)
        self.assertEqual(config["kd"]["top_k"], 64)
        self.assertEqual(supervision_mode(config), "equal_example")
        self.assertEqual(
            baseline_sft_run_name(config, "short_128"),
            "equal_example__qwen2p5_7b__short_128__seed_17",
        )

    def test_equal_token_protocol_registers_variable_record_counts(self) -> None:
        config = load_protocol(
            PROJECT_ROOT / "configs/capacity_length_logit_kd_equal_token_seed17_v1.json"
        )
        self.assertEqual(supervision_mode(config), "equal_token")
        self.assertEqual(
            [config["budgets"][name]["expected_records"] for name in config["budgets"]],
            [881, 381, 179],
        )
        self.assertEqual(
            baseline_sft_run_name(config, "long_512"),
            "equal_token__qwen2p5_7b__long_512__seed_17",
        )

    def test_float_and_run_slugs_are_stable(self) -> None:
        self.assertEqual(float_slug(0.25), "0p25")
        self.assertEqual(float_slug(2.0), "2")
        self.assertEqual(kd_run_name("short_128", 0.25, 2.0), "short_128__a0p25__t2__seed_17")

    def test_completion_tokenization_uses_exact_chat_prefix(self) -> None:
        encoded = tokenize_completion_record(
            FakeChatTokenizer(),
            {
                "id": "row-1",
                "prompt": "question",
                "completion": "answer",
                "metadata": {"problem_id": "problem-1"},
            },
            max_length=16,
        )
        self.assertEqual(encoded["input_ids"], [1, 2, 3, 4, 5, 9])
        self.assertEqual(encoded["prompt_token_count"], 3)
        self.assertEqual(encoded["target_ids"], [4, 5, 9])

    def test_completion_tokenization_rejects_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds registered max_length"):
            tokenize_completion_record(
                FakeChatTokenizer(),
                {"id": "row-1", "prompt": "question", "completion": "answer", "metadata": {}},
                max_length=5,
            )

    def test_dual_context_tokenization_aligns_one_completion(self) -> None:
        protocol = {
            "kd": {
                "context_mode": "dual_prompt_teacher_forced",
                "student_context_field": "prompt",
                "teacher_context_field": "teacher_prompt",
            }
        }
        encoded = tokenize_kd_record(
            protocol,
            DualContextFakeChatTokenizer(),
            {
                "id": "row-1",
                "prompt": "student question",
                "teacher_prompt": "teacher budget question",
                "completion": "answer",
                "metadata": {"problem_id": "problem-1"},
            },
            max_length=16,
        )
        self.assertEqual(encoded["student_input_ids"], [1, 2, 3, 4, 5, 9])
        self.assertEqual(encoded["teacher_input_ids"], [7, 8, 3, 4, 5, 9])
        self.assertEqual(encoded["target_ids"], [4, 5, 9])
        self.assertEqual(encoded["context_mode"], "dual_prompt_teacher_forced")

    def test_dual_context_tokenization_rejects_target_mismatch(self) -> None:
        protocol = {
            "kd": {
                "context_mode": "dual_prompt_teacher_forced",
                "student_context_field": "prompt",
                "teacher_context_field": "teacher_prompt",
            }
        }
        with self.assertRaisesRegex(ValueError, "identical completion targets"):
            tokenize_kd_record(
                protocol,
                DualContextFakeChatTokenizer(mismatch=True),
                {
                    "id": "row-1",
                    "prompt": "student question",
                    "teacher_prompt": "teacher budget question",
                    "completion": "answer",
                    "metadata": {},
                },
                max_length=16,
            )

    def test_legacy_context_defaults_to_prompt_for_both_models(self) -> None:
        self.assertEqual(
            kd_context_fields({"kd": {}}),
            ("prompt", "prompt", "same_prompt_teacher_forced"),
        )

    def test_dual_context_protocol_loads(self) -> None:
        config = load_protocol(
            PROJECT_ROOT
            / "configs/capacity_length_logit_kd_teacher_prompt_equal_token_seed17_v1.json"
        )
        self.assertEqual(
            kd_context_fields(config),
            ("prompt", "teacher_prompt", "dual_prompt_teacher_forced"),
        )
        self.assertEqual(config["validation"]["max_compliance_drop"], 0.02)

    def test_compliance_gate_uses_registered_two_point_margin(self) -> None:
        self.assertTrue(
            compliance_gate_satisfied(
                {
                    "short_128": {"compliance_delta_vs_sft": -0.02},
                    "medium_256": {"compliance_delta_vs_sft": -0.019},
                    "long_512": {"compliance_delta_vs_sft": 0.0},
                },
                0.02,
            )
        )
        self.assertFalse(
            compliance_gate_satisfied(
                {
                    "short_128": {"compliance_delta_vs_sft": -0.021},
                    "medium_256": {"compliance_delta_vs_sft": 0.0},
                    "long_512": {"compliance_delta_vs_sft": 0.0},
                },
                0.02,
            )
        )

    def test_multiteacher_condition_validates_teacher_and_benchmark_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "mixed.jsonl"
            rows = []
            for source, tokens in (("gsm8k", 7), ("hendrycks_math", 11)):
                rows.append(
                    {
                        "id": f"{source}::row",
                        "prompt": "question",
                        "completion": "answer",
                        "metadata": {
                            "problem_id": f"{source}-problem",
                            "budget_name": "medium_256",
                            "generator_name": "qwen2p5_3b",
                            "dataset_source": source,
                            "solution_token_count": tokens,
                            "is_correct": True,
                            "budget_compliant": True,
                        },
                    }
                )
            data_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            protocol = {
                "budgets": {
                    "qwen2p5_3b__medium_256": {
                        "budget_name": "medium_256",
                        "train_path": str(data_path),
                        "train_sha256": file_sha256(data_path),
                        "expected_records": 2,
                        "expected_solution_tokens": 18,
                        "expected_generator_name": "qwen2p5_3b",
                        "expected_dataset_sources": ["gsm8k", "hendrycks_math"],
                        "expected_source_counts": {"gsm8k": 1, "hendrycks_math": 1},
                        "expected_source_solution_tokens": {
                            "gsm8k": 7,
                            "hendrycks_math": 11,
                        },
                    }
                }
            }
            _, observed = validate_budget_dataset(
                protocol, "qwen2p5_3b__medium_256"
            )
            self.assertEqual(len(observed), 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable in this environment")
    def test_alpha_zero_matches_hard_cross_entropy(self) -> None:
        import torch
        import torch.nn.functional as functional

        student = torch.tensor([[2.0, 0.0, -1.0, -4.0], [0.0, 1.0, 2.0, -4.0]], requires_grad=True)
        teacher = torch.tensor([[1.0, 0.0, -1.0, -3.0], [2.0, 1.0, 0.0, -3.0]])
        targets = torch.tensor([0, 2])
        loss, metrics = hybrid_kd_loss(
            student,
            teacher,
            targets,
            alpha=0.0,
            temperature=2.0,
            valid_vocab_size=3,
        )
        expected = functional.cross_entropy(student.float(), targets)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertGreaterEqual(float(metrics["kd"]), 0.0)
        loss.backward()
        self.assertIsNotNone(student.grad)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable in this environment")
    def test_identical_distributions_have_zero_kl(self) -> None:
        import torch

        logits = torch.tensor([[1.0, 0.0, -1.0, -5.0], [0.0, 1.0, 2.0, -5.0]], requires_grad=True)
        targets = torch.tensor([0, 2])
        _, metrics = hybrid_kd_loss(
            logits,
            logits.detach().clone(),
            targets,
            alpha=0.5,
            temperature=4.0,
            valid_vocab_size=3,
        )
        self.assertAlmostEqual(float(metrics["kd"]), 0.0, places=6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable in this environment")
    def test_invalid_vocab_mass_matches_softmax_tail(self) -> None:
        import torch

        logits = torch.tensor([[0.0, 0.0, math.log(2.0)]])
        mass = invalid_vocab_probability_mass(logits, valid_vocab_size=2)
        self.assertAlmostEqual(float(mass[0]), 0.5, places=6)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable in this environment")
    def test_loss_rejects_out_of_vocab_target(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "outside"):
            hybrid_kd_loss(
                torch.zeros(1, 4),
                torch.zeros(1, 5),
                torch.tensor([3]),
                alpha=0.5,
                temperature=2.0,
                valid_vocab_size=3,
            )


if __name__ == "__main__":
    unittest.main()
