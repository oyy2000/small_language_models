#!/usr/bin/env python3
"""Tests for the capacity-by-length factorial protocol."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.backends import GenerationRequest, LocalRuleTeacherBackend
from length_budget_distill.factorial import (
    canonical_sha256,
    common_problem_ids,
    deterministic_equal_token_subset,
    file_sha256,
    select_shortest_correct,
    stable_generation_seed,
    validated_adapter_evidence,
)
from length_budget_distill.factorial_analysis import (
    exact_mcnemar_p_value,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_problem_effects,
    wilson_interval,
)
from length_budget_distill.records import ProblemRecord, TraceRecord, trace_from_dict, trace_to_dict
from length_budget_distill.sft_format import trace_to_sft_record
from length_budget_distill.student_prompts import build_student_math_prompt
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def make_trace(
    problem_id: str,
    generator_name: str,
    budget_name: str,
    candidate_index: int,
    tokens: int,
    correct: bool = True,
    compliant: bool = True,
) -> TraceRecord:
    return TraceRecord(
        trace_id=f"{problem_id}:{generator_name}:{budget_name}:{candidate_index}",
        problem_id=problem_id,
        question="What is 1 + 1?",
        answer="2",
        budget_name=budget_name,
        max_solution_tokens=128,
        teacher_backend="fixture",
        teacher_model="fixture-model",
        prompt="prompt",
        solution="Answer: 2",
        predicted_answer="2" if correct else "3",
        is_correct=correct,
        solution_token_count=tokens,
        generator_name=generator_name,
        generator_size_b=1.5,
        candidate_index=candidate_index,
        generation_seed=17,
        budget_compliant=compliant,
    )


class CapacityLengthFactorialTest(unittest.TestCase):
    def test_stable_generation_seed_is_reproducible_and_identity_specific(self) -> None:
        first = stable_generation_seed(7, "g1", "short", "p1")
        self.assertEqual(first, stable_generation_seed(7, "g1", "short", "p1"))
        self.assertNotEqual(first, stable_generation_seed(7, "g1", "short", "p2"))

    def test_local_backend_returns_requested_candidate_shape(self) -> None:
        problem = ProblemRecord(problem_id="p1", question="What is 1 + 1?", answer="2")
        budget = {"name": "short", "max_solution_tokens": 32}
        output = LocalRuleTeacherBackend().generate_batch(
            [GenerationRequest(problem=problem, budget=budget, prompt="prompt", seed=9)],
            num_candidates=3,
        )
        self.assertEqual(len(output), 1)
        self.assertEqual(len(output[0]), 3)
        self.assertTrue(all("Answer: 2" in candidate for candidate in output[0]))

    def test_gsm8k_verifier_accepts_numeric_answer_with_units_or_prose(self) -> None:
        self.assertTrue(verify_answer(extract_final_answer("Answer: 72 clips"), "72"))
        self.assertTrue(
            verify_answer(
                extract_final_answer("Answer: Natalia sold a total of 72 clips in April and May."),
                "72",
            )
        )
        self.assertTrue(verify_answer(extract_final_answer("Answer: $1,250.00"), "1250"))
        self.assertFalse(verify_answer(extract_final_answer("Answer: 71 clips"), "72"))

    def test_selection_uses_shortest_correct_compliant_candidate_and_tie_index(self) -> None:
        traces = [
            make_trace("p1", "g1", "short", 0, 10, correct=False),
            make_trace("p1", "g1", "short", 1, 4, compliant=False),
            make_trace("p1", "g1", "short", 2, 8),
            make_trace("p2", "g1", "short", 2, 6),
            make_trace("p2", "g1", "short", 1, 6),
        ]
        selected = select_shortest_correct(traces)
        self.assertEqual([(trace.problem_id, trace.candidate_index) for trace in selected], [("p1", 2), ("p2", 1)])
        self.assertTrue(all(trace.selected_for_sft for trace in selected))

    def test_sft_uses_the_registered_student_evaluation_prompt(self) -> None:
        trace = make_trace("p1", "g1", "short", 0, 4)
        record = trace_to_sft_record(trace)
        expected = build_student_math_prompt(trace.question)
        self.assertEqual(record["prompt"], expected)
        self.assertEqual(record["messages"][0]["content"], expected)

    def test_common_problem_intersection_requires_every_condition(self) -> None:
        traces = [
            make_trace("p1", "g1", "short", 0, 4),
            make_trace("p2", "g1", "short", 0, 4),
            make_trace("p1", "g2", "short", 0, 4),
        ]
        self.assertEqual(common_problem_ids(traces, [("g1", "short"), ("g2", "short")]), ["p1"])

    def test_equal_token_subset_never_exceeds_target(self) -> None:
        traces = [make_trace(f"p{index}", "g1", "short", 0, tokens) for index, tokens in enumerate([7, 5, 4, 3])]
        subset, total = deterministic_equal_token_subset(traces, target_tokens=12, seed=17)
        self.assertLessEqual(total, 12)
        self.assertEqual(total, sum(trace.solution_token_count for trace in subset))
        self.assertEqual(
            [trace.problem_id for trace in subset],
            [trace.problem_id for trace in deterministic_equal_token_subset(traces, 12, 17)[0]],
        )

    def test_old_trace_json_remains_readable(self) -> None:
        row = trace_to_dict(make_trace("p1", "g1", "short", 0, 4))
        for field in (
            "generator_name",
            "generator_size_b",
            "candidate_index",
            "generation_seed",
            "budget_compliant",
            "selected_for_sft",
            "config_hash",
            "source_hash",
        ):
            row.pop(field)
        restored = trace_from_dict(row)
        self.assertIsNone(restored.generator_name)
        self.assertEqual(restored.candidate_index, 0)

    def test_adapter_completion_evidence_detects_tampering(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "adapter_config.json"
            model_path = root / "adapter_model.safetensors"
            config_path.write_text("{}\n", encoding="utf-8")
            model_path.write_bytes(b"weights")
            (root / "TRAIN_COMPLETE").write_text(
                "run_name=fixture\n"
                "seed=17\n"
                "train_sha256=fixture-train\n"
                "run_config_sha256=fixture-config\n"
                "training_source_sha256=fixture-training-source\n"
                "launcher_source_sha256=fixture-launcher-source\n"
                f"adapter_config_sha256={file_sha256(config_path)}\n"
                f"adapter_model_sha256={file_sha256(model_path)}\n",
                encoding="utf-8",
            )
            self.assertIsNotNone(validated_adapter_evidence(root))
            model_path.write_bytes(b"changed")
            self.assertIsNone(validated_adapter_evidence(root))

    def test_adapter_completion_evidence_accepts_logit_kd_json_marker(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "adapter_config.json"
            model_path = root / "adapter_model.safetensors"
            metrics_path = root / "training_metrics.json"
            manifest_path = root / "train_manifest.json"
            config_path.write_text("{}\n", encoding="utf-8")
            model_path.write_bytes(b"weights")
            metrics_path.write_text("{}\n", encoding="utf-8")
            manifest = {"run_name": "logit-kd-fixture", "status": "complete"}
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker = {
                **manifest,
                "train_manifest_sha256": file_sha256(manifest_path),
                "adapter_config_sha256": file_sha256(config_path),
                "adapter_model_sha256": file_sha256(model_path),
                "training_metrics_sha256": file_sha256(metrics_path),
            }
            (root / "TRAIN_COMPLETE").write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            evidence = validated_adapter_evidence(root)
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence["run_name"], "logit-kd-fixture")

    def test_holm_and_paired_bootstrap_detect_positive_effect(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))
        left = {(f"p{index}", 17): True for index in range(20)}
        right = {(f"p{index}", 17): False for index in range(20)}
        effects = paired_problem_effects(left, right)
        result = paired_cluster_bootstrap(effects, samples=200, seed=9)
        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["ci_low"], 1.0)

    def test_shared_binomial_intervals_and_exact_mcnemar(self) -> None:
        lower, upper = wilson_interval(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p_value(5, 0), 0.0625)

    def test_local_end_to_end_generation_selection_and_dataset_build(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.jsonl"
            rows = [
                {"id": f"p{index}", "question": f"What is {index} + 1?", "answer": str(index + 1)}
                for index in range(4)
            ]
            data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config_path = root / "config.json"
            config = {
                "experiment_name": "fixture_factorial",
                "dataset": {
                    "source": "local_jsonl",
                    "path": str(data_path),
                    "question_field": "question",
                    "answer_field": "answer",
                    "max_examples": 4,
                },
                "generators": [
                    {
                        "name": "fixture_1p5b",
                        "size_b": 1.5,
                        "model_name": "fixture",
                        "backend": "local_rule",
                    }
                ],
                "generation": {"num_candidates": 3, "base_seed": 4, "batch_size": 2},
                "length_budgets": [
                    {"name": "short", "max_solution_tokens": 8, "style_hint": "short"},
                    {"name": "long", "max_solution_tokens": 64, "style_hint": "long"},
                ],
                "token_counter": {"backend": "whitespace"},
                "output": {"include_prompt_in_trace": True},
                "balancing": {
                    "smoke_min_common_problems": 4,
                    "formal_min_common_problems": 4,
                    "training_seeds": [17, 42, 73],
                    "max_equal_token_gap": 64,
                },
                "student": {"model_name": "fixture-student"},
                "training": {},
                "evaluation": {
                    "smoke_start_index": 0,
                    "smoke_limit": 4,
                    "formal_start_index": 0,
                    "formal_limit": 4,
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            raw_dir = root / "raw" / "fixture_1p5b"
            for shard_index in (0, 1):
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "5_1_generate_capacity_length_traces.py"),
                        "--config",
                        str(config_path),
                        "--generator-name",
                        "fixture_1p5b",
                        "--output-dir",
                        str(raw_dir),
                        "--num-shards",
                        "2",
                        "--shard-index",
                        str(shard_index),
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            selected_dir = root / "selected"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "5_2_merge_select_capacity_length_traces.py"),
                    "--config",
                    str(config_path),
                    "--input-glob",
                    str(raw_dir / "shard_*.jsonl"),
                    "--output-dir",
                    str(selected_dir),
                    "--stage",
                    "smoke",
                    "--expected-problems",
                    "4",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            sft_dir = root / "sft"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "5_3_build_capacity_length_sft_data.py"),
                    "--config",
                    str(config_path),
                    "--selected-traces",
                    str(selected_dir / "selected_traces.jsonl"),
                    "--output-dir",
                    str(sft_dir),
                    "--stage",
                    "smoke",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            audit = json.loads((selected_dir / "selection_audit.json").read_text(encoding="utf-8"))
            manifest = json.loads((sft_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["actual_total_candidates"], 4 * 2 * 3)
            self.assertEqual(audit["common_problem_count"], 4)
            self.assertEqual(audit["selected_sha256"], file_sha256(selected_dir / "selected_traces.jsonl"))
            self.assertEqual(manifest["expected_run_count"], 2 * 3 * 2 + 2 * 3)
            self.assertTrue(all(run.get("train_sha256") for run in manifest["runs"]))
            self.assertTrue((sft_dir / "DATASETS_COMPLETE").is_file())

            run_config_path = root / "run_seed17.json"
            run_config_path.write_text(
                json.dumps(
                    {
                        "parent_config_sha256": canonical_sha256(config),
                        "protocol_variant": "formal_single_seed_reduced",
                        "training_seeds": [17],
                    }
                ),
                encoding="utf-8",
            )
            reduced_sft_dir = root / "sft_seed17"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "5_3_build_capacity_length_sft_data.py"),
                    "--config",
                    str(config_path),
                    "--run-config",
                    str(run_config_path),
                    "--selected-traces",
                    str(selected_dir / "selected_traces.jsonl"),
                    "--output-dir",
                    str(reduced_sft_dir),
                    "--stage",
                    "smoke",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            reduced_manifest = json.loads(
                (reduced_sft_dir / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reduced_manifest["training_seeds"], [17])
            self.assertEqual(reduced_manifest["expected_run_count"], 2 * 1 * 2 + 2 * 1)
            self.assertEqual(reduced_manifest["protocol_variant"], "formal_single_seed_reduced")
            self.assertEqual(reduced_manifest["run_config"]["sha256"], file_sha256(run_config_path))


if __name__ == "__main__":
    unittest.main()
