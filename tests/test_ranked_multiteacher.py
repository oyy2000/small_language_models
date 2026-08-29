from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from length_budget_distill.factorial import (
    declared_launcher_wave_groups,
    select_launcher_shard_runs,
)
from length_budget_distill.ranked_multiteacher import (
    LAUNCHER_ASSIGNMENT_POLICY,
    LAUNCHER_SHARDS,
    LAUNCHER_WAVES,
    RANK_NAMES,
    TEACHER_NAMES,
    TRAINING_SEEDS,
    generation_config_for_teacher,
    ordered_matrix_runs,
    validate_launcher_assignment,
    validate_protocol,
)
from length_budget_distill.ranked_multiteacher_analysis import analyze_accuracy_contrasts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RankedMultiteacherProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        with (PROJECT_ROOT / "configs/capacity_length_ranked_sampling_multiteacher_v1.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.config = json.load(handle)

    def test_registered_matrix_has_exactly_36_runs(self) -> None:
        validate_protocol(self.config)
        runs = ordered_matrix_runs()
        summary = validate_launcher_assignment(runs)
        self.assertEqual(len(runs), 36)
        self.assertEqual(summary["run_count"], 36)
        self.assertEqual(len(summary["per_shard"]), 3)
        self.assertEqual(len({row["run_name"] for row in runs}), 36)
        self.assertEqual(sum(row["generator_name"] == "qwen2p5_1p5b" for row in runs), 9)

    def test_generation_materialization_excludes_sealed_7b(self) -> None:
        frozen = copy.deepcopy(self.config)
        frozen.pop("phase_a_parent_spec")
        frozen["phase_a_evidence"] = {"status": "passed"}
        derived = generation_config_for_teacher(
            frozen,
            "qwen2p5_3b",
            protocol_path="frozen.json",
            protocol_file_sha256="abc",
        )
        self.assertEqual(derived["teacher"]["name"], "qwen2p5_3b")
        self.assertEqual(derived["generation"]["num_candidates"], 16)
        with self.assertRaisesRegex(ValueError, "must be reused"):
            generation_config_for_teacher(
                frozen,
                "qwen2p5_7b",
                protocol_path="frozen.json",
                protocol_file_sha256="abc",
            )

    def test_matrix_launcher_assignment_balances_all_registered_factors(self) -> None:
        runs = ordered_matrix_runs()
        self.assertEqual(len(runs), 36)
        for index, run in enumerate(runs):
            self.assertEqual(run["launcher_shards"], LAUNCHER_SHARDS)
            self.assertEqual(run["launcher_shard_index"], index % LAUNCHER_SHARDS)
            self.assertEqual(run["launcher_assignment_policy"], LAUNCHER_ASSIGNMENT_POLICY)

        for shard_index in range(LAUNCHER_SHARDS):
            shard = select_launcher_shard_runs(
                runs,
                launcher_shards=LAUNCHER_SHARDS,
                launcher_shard_index=shard_index,
            )
            self.assertEqual(len(shard), 12)
            waves = declared_launcher_wave_groups(shard)
            self.assertEqual([len(wave) for wave in waves], [3, 3, 3, 3])
            self.assertEqual(
                Counter(run["generator_name"] for run in shard),
                Counter({teacher: 3 for teacher in TEACHER_NAMES}),
            )
            self.assertEqual(
                Counter(run["budget_name"] for run in shard),
                Counter({rank: 4 for rank in RANK_NAMES}),
            )
            self.assertEqual(
                Counter(int(run["seed"]) for run in shard),
                Counter({seed: 4 for seed in TRAINING_SEEDS}),
            )
            for wave_index in range(LAUNCHER_WAVES):
                wave = [
                    run for run in shard if run["launcher_wave_index"] == wave_index
                ]
                self.assertEqual(len(wave), 3)
                self.assertEqual(len({run["generator_name"] for run in wave}), 3)
                self.assertEqual(
                    Counter(run["budget_name"] for run in wave),
                    Counter({rank: 1 for rank in RANK_NAMES}),
                )

        for wave_index in range(LAUNCHER_WAVES):
            wave = [run for run in runs if run["launcher_wave_index"] == wave_index]
            self.assertEqual(len(wave), 9)
            self.assertEqual(
                Counter(run["budget_name"] for run in wave),
                Counter({rank: 3 for rank in RANK_NAMES}),
            )
            self.assertEqual(
                Counter(int(run["seed"]) for run in wave),
                Counter({seed: 3 for seed in TRAINING_SEEDS}),
            )

        for teacher in TEACHER_NAMES:
            for rank in RANK_NAMES:
                cell = [
                    run
                    for run in runs
                    if run["generator_name"] == teacher and run["budget_name"] == rank
                ]
                self.assertEqual(
                    {int(run["launcher_shard_index"]) for run in cell},
                    set(range(LAUNCHER_SHARDS)),
                )

    def test_launcher_selector_rejects_mixed_or_mismatched_assignments(self) -> None:
        implicit = [{"run_name": f"run_{index}"} for index in range(6)]
        selected = select_launcher_shard_runs(
            implicit, launcher_shards=3, launcher_shard_index=1
        )
        self.assertEqual([run["run_name"] for run in selected], ["run_1", "run_4"])

        mixed = copy.deepcopy(implicit)
        mixed[0].update({"launcher_shards": 3, "launcher_shard_index": 0})
        with self.assertRaisesRegex(ValueError, "mixes declared and implicit"):
            select_launcher_shard_runs(mixed, launcher_shards=3, launcher_shard_index=0)

        mismatched = ordered_matrix_runs()
        mismatched[0]["launcher_shards"] = 4
        with self.assertRaisesRegex(ValueError, "topology mismatch"):
            select_launcher_shard_runs(mismatched, launcher_shards=3, launcher_shard_index=0)

        mixed_waves = select_launcher_shard_runs(
            ordered_matrix_runs(), launcher_shards=3, launcher_shard_index=0
        )
        mixed_waves[0].pop("launcher_wave_index")
        with self.assertRaisesRegex(ValueError, "mix declared and implicit launcher waves"):
            declared_launcher_wave_groups(mixed_waves)

        reversed_waves = list(
            reversed(
                select_launcher_shard_runs(
                    ordered_matrix_runs(), launcher_shards=3, launcher_shard_index=0
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "not ordered by wave"):
            declared_launcher_wave_groups(reversed_waves)

    def test_training_launcher_wave_barrier_prevents_cross_wave_overlap(self) -> None:
        script_path = PROJECT_ROOT / "scripts/6_1_train_capacity_length_students.py"
        spec = importlib.util.spec_from_file_location("phase6_training_launcher", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FakeProcess:
            active: dict[int, int] = {}
            overlap_violations: list[tuple[int, tuple[int, ...]]] = []
            next_identity = 0

            def __init__(self, command, **kwargs):
                del kwargs
                self.identity = FakeProcess.next_identity
                FakeProcess.next_identity += 1
                self.wave = int(str(command[0]).split("_", maxsplit=1)[0].removeprefix("wave"))
                other_waves = tuple(sorted(set(FakeProcess.active.values()) - {self.wave}))
                if other_waves:
                    FakeProcess.overlap_violations.append((self.wave, other_waves))
                FakeProcess.active[self.identity] = self.wave
                self.remaining_polls = 1 if self.identity % 3 == 0 else 3

            def poll(self):
                self.remaining_polls -= 1
                if self.remaining_polls > 0:
                    return None
                FakeProcess.active.pop(self.identity, None)
                return 1

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            entries = []
            commands = []
            for wave_index in range(2):
                for position in range(3):
                    run_name = f"wave{wave_index}_run{position}"
                    entries.append(
                        {
                            "run_name": run_name,
                            "output_dir": str(root / "published" / run_name),
                            "log_path": str(root / "logs" / f"{run_name}.log"),
                            "launcher_wave_index": wave_index,
                        }
                    )
                    commands.append([run_name])
            with patch.dict(
                os.environ,
                {"LBD_RUNTIME_CHECKPOINT_ROOT": str(root / "runtime")},
                clear=False,
            ), patch.object(module.subprocess, "Popen", FakeProcess), patch.object(
                module.time, "sleep", lambda _: None
            ):
                failures = module._run_commands(
                    entries,
                    commands,
                    ["0", "1", "2"],
                    3,
                    False,
                    wave_barrier=True,
                )
            self.assertEqual(len(failures), 6)
            self.assertEqual(FakeProcess.overlap_violations, [])
            self.assertEqual(FakeProcess.active, {})

    def test_registered_contrast_families_recover_synthetic_interactions(self) -> None:
        problem_count = 40
        long_correct = 10
        short_gains = {
            "qwen2p5_1p5b": 0,
            "qwen2p5_3b": 4,
            "qwen2p5_7b": 8,
            "qwen2p5_14b": 12,
        }
        predictions = {}
        for teacher in TEACHER_NAMES:
            predictions[teacher] = {}
            for seed in TRAINING_SEEDS:
                thresholds = {
                    "relative_long": long_correct,
                    "relative_medium": long_correct + short_gains[teacher] // 2,
                    "relative_short": long_correct + short_gains[teacher],
                }
                predictions[teacher][seed] = {
                    rank: {
                        f"problem_{index:03d}": {"is_correct": index < threshold}
                        for index in range(problem_count)
                    }
                    for rank, threshold in thresholds.items()
                }

        analysis = self.config["analysis"]
        within, interactions = analyze_accuracy_contrasts(
            predictions,
            within_teacher_pairs=analysis["within_teacher_pairs"],
            interaction_rank_pair=analysis["interaction_rank_pair"],
            teacher_interaction_pairs=analysis["teacher_interaction_pairs"],
            bootstrap_samples=300,
            familywise_alpha=analysis["familywise_alpha"],
            config_hash="synthetic-main-matrix",
        )
        self.assertEqual(len(within), 12)
        self.assertEqual(len(interactions), 6)
        self.assertEqual({row["family"] for row in within}, {"within_teacher_rank_12"})
        self.assertEqual(
            {row["family"] for row in interactions},
            {"teacher_by_rank_interaction_6"},
        )
        for row in within + interactions:
            self.assertEqual(row["seed_count"], 3)
            self.assertEqual(row["problem_count"], problem_count)
            self.assertEqual(row["resampling_units"], ["training_seed", "paired_problem"])
            self.assertGreaterEqual(row["bootstrap_holm_p_value"], 0.0)
            self.assertLessEqual(row["bootstrap_holm_p_value"], 1.0)

        short_long = {
            row["teacher_name"]: row["estimate"]
            for row in within
            if row["left_rank"] == "relative_short"
            and row["right_rank"] == "relative_long"
        }
        self.assertEqual(set(short_long), set(TEACHER_NAMES))
        for teacher, gain in short_gains.items():
            self.assertAlmostEqual(short_long[teacher], gain / problem_count)

        expected_interactions = {
            (left, right): (short_gains[left] - short_gains[right]) / problem_count
            for left, right in analysis["teacher_interaction_pairs"]
        }
        observed_interactions = {
            (row["left_teacher"], row["right_teacher"]): row["estimate"]
            for row in interactions
        }
        self.assertEqual(set(observed_interactions), set(expected_interactions))
        for pair, expected in expected_interactions.items():
            self.assertAlmostEqual(observed_interactions[pair], expected)


if __name__ == "__main__":
    unittest.main()
