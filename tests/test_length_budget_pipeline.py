#!/usr/bin/env python3
"""Lightweight sanity tests for the length-budget pipeline."""

from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.analysis import summarize_by_budget
from length_budget_distill.backends import LocalRuleTeacherBackend, _max_new_tokens_for_budget
from length_budget_distill.bucketing import get_length_budgets
from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.prompts import build_length_budget_prompt
from length_budget_distill.records import TraceRecord
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.tokenization import WhitespaceTokenCounter
from length_budget_distill.verifiers import extract_final_answer, verify_answer


class LengthBudgetPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.data_path = self.tmp_path / "local_math.jsonl"
        rows = [
            {
                "id": "sample-001",
                "question": "Mia has 3 apples and buys 5 more. How many apples does she have?",
                "answer": "8",
            },
            {
                "id": "sample-002",
                "question": "A box has 4 rows of pencils with 6 pencils in each row. How many pencils are there?",
                "answer": "24",
            },
        ]
        self.data_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.config_path = self.tmp_path / "local_length_budget.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "experiment_name": "local_length_budget",
                    "dataset": {
                        "source": "local_jsonl",
                        "path": str(self.data_path),
                        "question_field": "question",
                        "answer_field": "answer",
                    },
                    "teacher": {
                        "backend": "local_rule",
                        "model_name": "local-rule-teacher",
                    },
                    "length_budgets": [
                        {
                            "name": "small",
                            "max_solution_tokens": 24,
                            "style_hint": "Use only the essential equation and final answer.",
                        },
                        {
                            "name": "medium",
                            "max_solution_tokens": 48,
                            "style_hint": "Use a compact step-by-step solution.",
                        },
                        {
                            "name": "large",
                            "max_solution_tokens": 96,
                            "style_hint": "Use explicit intermediate reasoning and a final answer.",
                        },
                    ],
                    "token_counter": {"backend": "whitespace"},
                }
            ),
            encoding="utf-8",
        )
        self.config = load_config(str(self.config_path))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_local_config_loads_local_records(self) -> None:
        records = load_problem_records(self.config)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].answer, "8")

    def test_budget_prompts_and_local_generation_are_verified(self) -> None:
        records = load_problem_records(self.config)
        budgets = get_length_budgets(self.config)
        teacher = LocalRuleTeacherBackend()
        counter = WhitespaceTokenCounter()

        counts = []
        for budget in budgets:
            prompt = build_length_budget_prompt(records[0], budget)
            self.assertIn(str(budget["max_solution_tokens"]), prompt)
            solution = teacher.generate(records[0], budget, prompt)
            predicted = extract_final_answer(solution)
            self.assertTrue(verify_answer(predicted, records[0].answer))
            counts.append(counter.count(solution))

        self.assertLess(counts[0], counts[-1])

    def test_sft_and_summary_shapes(self) -> None:
        trace = TraceRecord(
            trace_id="sample-001:small",
            problem_id="sample-001",
            question="What is 3 + 5?",
            answer="8",
            budget_name="small",
            max_solution_tokens=24,
            teacher_backend="local_rule",
            teacher_model="local-rule-teacher",
            prompt="prompt",
            solution="Compute directly. Answer: 8",
            predicted_answer="8",
            is_correct=True,
            solution_token_count=4,
        )
        sft = trace_to_sft_record(trace)
        self.assertEqual(sft["messages"][1]["role"], "assistant")
        summary = summarize_by_budget([trace])
        self.assertEqual(summary[0]["budget_name"], "small")
        self.assertEqual(summary[0]["correct_rate"], 1.0)

    def test_gsm8k_answer_extraction_for_local_records(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gsm8k_style.jsonl"
            row = {
                "id": "gsm8k-001",
                "question": "What is 40 + 2?",
                "answer": "Compute 40 + 2.\n#### 42",
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            config = {
                "dataset": {
                    "source": "local_jsonl",
                    "path": str(path),
                    "question_field": "question",
                    "answer_field": "answer",
                    "answer_format": "gsm8k",
                }
            }
            records = load_problem_records(config)
            self.assertEqual(records[0].answer, "42")
            self.assertIn("raw_answer", records[0].metadata)

    def test_real_templates_are_qwen_prefilled_without_approval_block(self) -> None:
        real_config = load_config(str(PROJECT_ROOT / "configs" / "real_length_budget_template.json"))
        student_config = load_config(str(PROJECT_ROOT / "configs" / "student_sft_template.json"))

        self.assertIn("Qwen", real_config["teacher"]["model_name"])
        self.assertIn("Qwen", real_config["token_counter"]["tokenizer_name"])
        self.assertIn("Qwen", student_config["student"]["model_name"])
        self.assertEqual([budget["name"] for budget in real_config["length_budgets"]], ["small", "medium", "large"])
        self.assertFalse(real_config["teacher"]["generation"]["cap_max_new_tokens_by_budget"])
        self.assertEqual(student_config["data"]["text_format"], "prompt_completion")
        self.assertTrue(student_config["training"]["completion_only_loss"])
        self.assertFalse(student_config["training"]["assistant_only_loss"])
        self.assertNotIn("approval", real_config)
        self.assertNotIn("approval", student_config)

    def test_generation_max_tokens_are_capped_by_budget(self) -> None:
        generation_config = {"max_new_tokens": 1024, "cap_max_new_tokens_by_budget": True}
        budget = {"name": "small", "max_solution_tokens": 128}
        self.assertEqual(_max_new_tokens_for_budget(generation_config, budget), 128)

    def test_generation_max_tokens_are_not_capped_by_default(self) -> None:
        generation_config = {"max_new_tokens": 1024}
        budget = {"name": "small", "max_solution_tokens": 128}
        self.assertEqual(_max_new_tokens_for_budget(generation_config, budget), 1024)

    def test_prompt_uses_prompt_level_length_budget_and_requires_answer(self) -> None:
        records = load_problem_records(self.config)
        prompt = build_length_budget_prompt(
            records[0],
            {
                "name": "small",
                "max_solution_tokens": 128,
                "style_hint": "Use compressed equations.",
            },
        )
        self.assertIn("solve in <= 128 solution tokens", prompt)
        self.assertIn("prompt-level length target", prompt)
        self.assertIn("do not omit the final answer", prompt)
        self.assertIn("Answer: <final answer>", prompt)


if __name__ == "__main__":
    unittest.main()
