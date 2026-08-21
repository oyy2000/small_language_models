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
from length_budget_distill.prompts import build_length_budget_prompt, build_teacher_prompt, get_prompt_strategy
from length_budget_distill.records import TraceRecord
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.tokenization import WhitespaceTokenCounter
import length_budget_distill.training as training_helpers
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
            metadata={"prompt_strategy": "standard", "problem_metadata": {"source": "unit"}},
        )
        sft = trace_to_sft_record(trace)
        self.assertEqual(sft["messages"][1]["role"], "assistant")
        self.assertEqual(sft["teacher_prompt"], "prompt")
        self.assertEqual(sft["metadata"]["prompt_strategy"], "standard")
        self.assertEqual(sft["metadata"]["problem_metadata"]["source"], "unit")
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

        self.assertEqual(real_config["teacher"]["model_name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(real_config["teacher"]["tokenizer_name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(real_config["token_counter"]["tokenizer_name"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertIn("Qwen", student_config["student"]["model_name"])
        self.assertEqual([budget["name"] for budget in real_config["length_budgets"]], ["small", "medium", "large"])
        self.assertFalse(real_config["teacher"]["generation"]["cap_max_new_tokens_by_budget"])
        self.assertEqual(student_config["data"]["text_format"], "prompt_completion")
        self.assertTrue(student_config["training"]["completion_only_loss"])
        self.assertFalse(student_config["training"]["assistant_only_loss"])
        self.assertNotIn("approval", real_config)
        self.assertNotIn("approval", student_config)

    def test_legacy_code_prefixed_training_path_resolves_to_project_root(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            data_dir = tmp_root / "results" / "real_length_budget"
            data_dir.mkdir(parents=True)
            train_file = data_dir / "sft_small.jsonl"
            train_file.write_text('{"prompt": "q", "completion": "a"}\n', encoding="utf-8")

            original_root = training_helpers.PROJECT_ROOT
            training_helpers.PROJECT_ROOT = tmp_root
            try:
                resolved = training_helpers._require_existing_project_file(
                    "data.train_path",
                    "code/results/real_length_budget/sft_small.jsonl",
                )
                output_dir = training_helpers._resolve_project_path("code/checkpoints/student_sft")
            finally:
                training_helpers.PROJECT_ROOT = original_root

            self.assertEqual(Path(resolved), train_file)
            self.assertEqual(output_dir, tmp_root / "checkpoints" / "student_sft")

    def test_sft_config_omits_model_init_kwargs_when_model_is_preloaded(self) -> None:
        class DummySFTConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        student_config = {"trust_remote_code": False, "torch_dtype": "bfloat16"}
        training_config = {
            "output_dir": "checkpoints/test_student_sft",
            "max_steps": 1,
            "seed": 17,
            "data_seed": 42,
        }

        fresh_model_args = training_helpers._make_sft_config(
            DummySFTConfig,
            training_config,
            student_config,
        )
        resumed_model_args = training_helpers._make_sft_config(
            DummySFTConfig,
            training_config,
            student_config,
            include_model_init_kwargs=False,
        )

        self.assertIn("model_init_kwargs", fresh_model_args.kwargs)
        self.assertEqual(fresh_model_args.kwargs["model_init_kwargs"]["device_map"], "auto")
        self.assertEqual(fresh_model_args.kwargs["max_steps"], 1)
        self.assertEqual(fresh_model_args.kwargs["seed"], 17)
        self.assertEqual(fresh_model_args.kwargs["data_seed"], 42)
        self.assertNotIn("model_init_kwargs", resumed_model_args.kwargs)
        self.assertEqual(resumed_model_args.kwargs["max_steps"], 1)

    def test_resumed_adapter_training_enables_input_grads_for_checkpointing(self) -> None:
        class DummyParameter:
            requires_grad = True

            def numel(self) -> int:
                return 3

        class DummyModel:
            def __init__(self) -> None:
                self.train_called = False
                self.enable_input_grads_called = False

            def train(self) -> None:
                self.train_called = True

            def parameters(self):
                return [DummyParameter()]

            def enable_input_require_grads(self) -> None:
                self.enable_input_grads_called = True

        model = DummyModel()
        training_helpers._prepare_resumed_adapter_for_training(
            model,
            {"gradient_checkpointing": True},
        )

        self.assertTrue(model.train_called)
        self.assertTrue(model.enable_input_grads_called)

    def test_resumed_adapter_training_rejects_adapter_without_trainable_parameters(self) -> None:
        class FrozenParameter:
            requires_grad = False

            def numel(self) -> int:
                return 3

        class FrozenModel:
            def train(self) -> None:
                pass

            def parameters(self):
                return [FrozenParameter()]

        with self.assertRaisesRegex(ValueError, "no trainable parameters"):
            training_helpers._prepare_resumed_adapter_for_training(
                FrozenModel(),
                {"gradient_checkpointing": True},
            )

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

    def test_prompt_strategy_switches_standard_and_chain_of_draft(self) -> None:
        record = load_problem_records(self.config)[0]
        budget = {
            "name": "cod",
            "max_solution_tokens": 128,
            "style_hint": "Use draft notes.",
        }

        standard_prompt = build_teacher_prompt(record, budget, {"prompt": {"strategy": "standard"}})
        cod_prompt = build_teacher_prompt(
            record,
            budget,
            {"prompt": {"strategy": "cod"}},
        )

        self.assertEqual(get_prompt_strategy({"prompt": {"strategy": "baseline"}}), "chain_of_thought")
        self.assertEqual(get_prompt_strategy({"prompt": {"strategy": "chain-of-thought"}}), "chain_of_thought")
        self.assertEqual(get_prompt_strategy({"prompt": {"strategy": "chain-of-draft"}}), "chain_of_draft")
        self.assertNotIn("Length budget", standard_prompt)
        self.assertNotIn("Length budget", cod_prompt)
        self.assertIn("Think step by step to answer the following question.", standard_prompt)
        self.assertIn("There are 15 trees in the grove.", standard_prompt)
        self.assertIn("There are 15 trees originally.", standard_prompt)
        self.assertIn("#### 6", standard_prompt)
        self.assertTrue(standard_prompt.rstrip().endswith(f"Q: {record.question}\nA:"))
        self.assertIn(
            "Think step by step, but only keep minimum draft for each thinking step, "
            "with 5 words at most.",
            cod_prompt,
        )
        self.assertIn("21 - 15 = 6. #### 6", cod_prompt)
        self.assertTrue(cod_prompt.rstrip().endswith(f"Q: {record.question}\nA:"))

    def test_prompt_fewshot_template_can_be_loaded_from_config_path(self) -> None:
        record = load_problem_records(self.config)[0]
        budget = {"name": "standard", "max_solution_tokens": 512}
        template_path = self.tmp_path / "custom_prompt_template.json"
        template_path.write_text(
            json.dumps(
                {
                    "system_prompt": "Use the configured examples.",
                    "format": "Question: {question}\nAnswer: {answer}",
                    "fewshot": [
                        {
                            "question": "What is 1 + 1?",
                            "answer": "1 + 1 = 2. #### 2",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        prompt = build_teacher_prompt(
            record,
            budget,
            {
                "_config_path": str(self.config_path),
                "prompt": {
                    "strategy": "chain_of_thought",
                    "fewshot_path": template_path.name,
                },
            },
        )

        self.assertIn("Use the configured examples.", prompt)
        self.assertIn("Question: What is 1 + 1?", prompt)
        self.assertIn("Answer: 1 + 1 = 2. #### 2", prompt)
        self.assertTrue(prompt.rstrip().endswith(f"Question: {record.question}\nAnswer:"))


if __name__ == "__main__":
    unittest.main()
