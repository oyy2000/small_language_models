#!/usr/bin/env python3
"""Tests for relative-length repeated teacher sampling."""

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

from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.ranked_sampling import (
    LENGTH_LABELS,
    build_length_agnostic_teacher_prompt,
    load_bound_problem_ids,
    require_cohort_problems,
    select_relative_length_candidates,
)
from length_budget_distill.records import (
    ProblemRecord,
    TraceRecord,
    read_jsonl,
    trace_to_dict,
    write_jsonl,
)
from length_budget_distill.sft_format import trace_to_sft_record


def make_trace(
    problem_id: str,
    candidate_index: int,
    tokens: int,
    solution: str,
    correct: bool = True,
    config_hash: str = "fixture-hash",
) -> TraceRecord:
    return TraceRecord(
        trace_id=f"{problem_id}:qwen2p5_7b:unconstrained_sample_pool:candidate_{candidate_index:02d}",
        problem_id=problem_id,
        question="What is 1 + 1?",
        answer="2",
        budget_name="unconstrained_sample_pool",
        max_solution_tokens=512,
        teacher_backend="fixture",
        teacher_model="fixture-model",
        prompt="shared prompt",
        solution=solution,
        predicted_answer="2" if correct else "3",
        is_correct=correct,
        solution_token_count=tokens,
        metadata={"prompt_strategy": "length_agnostic_same_prompt_sampling"},
        generator_name="qwen2p5_7b",
        generator_size_b=7.0,
        candidate_index=candidate_index,
        generation_seed=17,
        budget_compliant=True,
        selected_for_sft=False,
        config_hash=config_hash,
        source_hash="fixture-source",
    )


class RankedSamplingTest(unittest.TestCase):
    def test_prompt_has_no_length_target_and_matches_answer_format(self) -> None:
        prompt = build_length_agnostic_teacher_prompt(
            ProblemRecord(problem_id="p1", question="What is 1 + 1?", answer="2")
        )
        self.assertNotIn("token", prompt.lower())
        self.assertNotIn("short", prompt.lower())
        self.assertIn("Answer: <final answer>", prompt)

    def test_selects_shortest_lower_median_and_longest_unique_correct(self) -> None:
        traces = [
            make_trace("p1", 0, 4, "First solution. Answer: 2"),
            make_trace("p1", 1, 2, "Wrong. Answer: 3", correct=False),
            make_trace("p1", 2, 5, " First   solution.\nAnswer: 2 "),
            make_trace("p1", 3, 8, "Second solution. Answer: 2"),
            make_trace("p1", 4, 10, "Third solution. Answer: 2"),
            make_trace("p1", 5, 20, "Fourth solution. Answer: 2"),
        ]
        selected = select_relative_length_candidates(traces)
        self.assertEqual(
            {label: selected[label].candidate_index for label in LENGTH_LABELS},
            {"short": 0, "medium": 3, "long": 5},
        )
        self.assertEqual(
            {label: selected[label].budget_name for label in LENGTH_LABELS},
            {"short": "relative_short", "medium": "relative_medium", "long": "relative_long"},
        )
        self.assertTrue(all(trace.selected_for_sft for trace in selected.values()))
        self.assertEqual(
            selected["medium"].metadata["relative_length_selection"]["eligible_unique_correct_count"],
            4,
        )

    def test_drops_problem_with_fewer_than_three_unique_correct_candidates(self) -> None:
        traces = [
            make_trace("p1", 0, 4, "First solution. Answer: 2"),
            make_trace("p1", 1, 5, "First solution. Answer: 2"),
            make_trace("p1", 2, 8, "Wrong. Answer: 3", correct=False),
        ]
        self.assertEqual(select_relative_length_candidates(traces), {})

    def test_loads_hash_bound_cohort_and_preserves_manifest_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort_path = root / "cohort.json"
            cohort_path.write_text(json.dumps({"problem_ids": ["p2", "p1"]}), encoding="utf-8")
            config = {
                "_config_path": str(root / "config.json"),
                "cohort": {
                    "problem_ids_path": str(cohort_path),
                    "problem_ids_field": "problem_ids",
                    "problem_ids_file_sha256": file_sha256(cohort_path),
                    "expected_problem_count": 2,
                },
            }
            problem_ids = load_bound_problem_ids(config)
            self.assertEqual(problem_ids, ["p2", "p1"])
            problems = [
                ProblemRecord(problem_id="p1", question="q1", answer="1"),
                ProblemRecord(problem_id="p2", question="q2", answer="2"),
            ]
            self.assertEqual(
                [problem.problem_id for problem in require_cohort_problems(problems, problem_ids)],
                ["p2", "p1"],
            )

    def test_merge_recomputes_selection_and_writes_training_ready_datasets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cohort_path = root / "cohort.json"
            cohort_path.write_text(json.dumps({"problem_ids": ["p0", "p1"]}), encoding="utf-8")
            config = {
                "experiment_name": "ranked_sampling_fixture",
                "dataset": {"verifier": "gsm8k_numeric"},
                "cohort": {
                    "problem_ids_path": str(cohort_path),
                    "problem_ids_field": "problem_ids",
                    "problem_ids_file_sha256": file_sha256(cohort_path),
                    "expected_problem_count": 2,
                },
                "teacher": {
                    "name": "fixture_teacher",
                    "model_name": "fixture-model",
                },
                "generation": {
                    "num_candidates": 4,
                    "num_shards": 2,
                    "max_new_tokens": 512,
                },
                "relative_length_selection": {
                    "labels": list(LENGTH_LABELS),
                    "method": "shortest_lower_median_longest",
                    "deduplication": "whitespace_normalized_exact_text",
                    "minimum_unique_correct": 3,
                    "insufficient_candidate_policy": "drop_problem_from_all_labels",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            config_hash = canonical_sha256(config)
            source_ids_hash = canonical_sha256(["p0", "p1"])
            input_dir = root / "generation"
            for shard_index, problem_id in enumerate(("p0", "p1")):
                traces = [
                    make_trace(problem_id, 0, 6, f"Short {problem_id}. Answer: 2", config_hash=config_hash),
                    make_trace(problem_id, 1, 10, f"Medium {problem_id}. Answer: 2", config_hash=config_hash),
                    make_trace(problem_id, 2, 20, f"Long {problem_id}. Answer: 2", config_hash=config_hash),
                    make_trace(
                        problem_id,
                        3,
                        4,
                        f"Wrong {problem_id}. Answer: 3",
                        correct=False,
                        config_hash=config_hash,
                    ),
                ]
                selected = select_relative_length_candidates(traces)
                suffix = f"shard_{shard_index:05d}_of_00002"
                raw_path = input_dir / "raw" / f"{suffix}.jsonl"
                selected_paths = {
                    label: input_dir / "selected_shards" / label / f"{suffix}.jsonl"
                    for label in LENGTH_LABELS
                }
                sft_paths = {
                    label: input_dir / "sft_shards" / label / f"{suffix}.jsonl"
                    for label in LENGTH_LABELS
                }
                write_jsonl(raw_path, (trace_to_dict(trace) for trace in traces))
                for label in LENGTH_LABELS:
                    write_jsonl(selected_paths[label], [trace_to_dict(selected[label])])
                    write_jsonl(sft_paths[label], [trace_to_sft_record(selected[label])])
                manifest = {
                    "status": "complete",
                    "config_hash": config_hash,
                    "source_problem_count": 2,
                    "source_problem_ids_sha256": source_ids_hash,
                    "shard_index": shard_index,
                    "num_shards": 2,
                    "num_candidates": 4,
                    "raw": _artifact(raw_path),
                    "selected": {label: _artifact(selected_paths[label]) for label in LENGTH_LABELS},
                    "sft": {label: _artifact(sft_paths[label]) for label in LENGTH_LABELS},
                }
                manifest_path = input_dir / "manifests" / f"{suffix}.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            output_dir = root / "merged"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "16_2_merge_ranked_length_samples.py"),
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
            self.assertTrue((output_dir / "GENERATION_COMPLETE").is_file())
            manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["eligible_problem_count"], 2)
            self.assertEqual([item["label"] for item in manifest["datasets"]], list(LENGTH_LABELS))
            for dataset in manifest["datasets"]:
                records = list(read_jsonl(Path(dataset["train_path"])))
                self.assertEqual(len(records), 2)
                self.assertTrue(all("prompt" in row and "completion" in row for row in records))

    def test_generation_entrypoint_writes_audited_local_rule_shard(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.jsonl"
            data_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"p{index}",
                            "question": f"What is {index} + 1?",
                            "answer": str(index + 1),
                        }
                    )
                    + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            cohort_path = root / "cohort.json"
            cohort_path.write_text(json.dumps({"problem_ids": ["p0", "p1"]}), encoding="utf-8")
            config = {
                "experiment_name": "ranked_sampling_generation_fixture",
                "dataset": {
                    "source": "local_jsonl",
                    "path": str(data_path),
                    "question_field": "question",
                    "answer_field": "answer",
                    "verifier": "gsm8k_numeric",
                },
                "cohort": {
                    "problem_ids_path": str(cohort_path),
                    "problem_ids_field": "problem_ids",
                    "problem_ids_file_sha256": file_sha256(cohort_path),
                    "expected_problem_count": 2,
                },
                "teacher": {
                    "name": "fixture_teacher",
                    "size_b": 1.0,
                    "model_name": "fixture-model",
                    "backend": "local_rule",
                },
                "generation": {
                    "num_candidates": 3,
                    "num_shards": 1,
                    "base_seed": 4,
                    "batch_size": 2,
                    "temperature": 0.0,
                    "max_new_tokens": 64,
                },
                "relative_length_selection": {
                    "labels": list(LENGTH_LABELS),
                    "method": "shortest_lower_median_longest",
                    "deduplication": "whitespace_normalized_exact_text",
                    "minimum_unique_correct": 3,
                    "insufficient_candidate_policy": "drop_problem_from_all_labels",
                },
                "token_counter": {"backend": "whitespace"},
                "output": {"include_prompt_in_trace": True},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output_dir = root / "generation"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "16_1_generate_ranked_length_samples.py"),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--shard-index",
                    "0",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest_path = output_dir / "manifests" / "shard_00000_of_00001.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["raw_record_count"], 6)
            self.assertEqual(manifest["eligible_problem_count"], 0)
            self.assertEqual(manifest["dropped_problem_count"], 2)
            self.assertEqual(manifest["raw"]["record_count"], 6)
            self.assertTrue(all(manifest["sft"][label]["record_count"] == 0 for label in LENGTH_LABELS))


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "record_count": sum(1 for _ in read_jsonl(path)),
    }


if __name__ == "__main__":
    unittest.main()
