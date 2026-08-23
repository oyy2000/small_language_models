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
    float_slug,
    hybrid_kd_loss,
    invalid_vocab_probability_mass,
    kd_run_name,
    load_protocol,
    tokenize_completion_record,
)


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


class LogitKDTest(unittest.TestCase):
    def test_registered_protocol_loads_and_is_single_seed(self) -> None:
        config = load_protocol(PROJECT_ROOT / "configs/capacity_length_logit_kd_seed17_v1.json")
        self.assertEqual(config["training"]["seed"], 17)
        self.assertEqual(config["models"]["tokenizer"]["expected_length"], 151665)
        self.assertEqual(config["kd"]["top_k"], 64)

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
