"""Registered protocol helpers for the teacher-capacity by length-rank matrix."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from .factorial import canonical_sha256


PROTOCOL_VARIANT = "registered_multiteacher_ranked_length_main_matrix"
TEACHER_NAMES = ("qwen2p5_1p5b", "qwen2p5_3b", "qwen2p5_7b", "qwen2p5_14b")
RANK_NAMES = ("relative_short", "relative_medium", "relative_long")
TRAINING_SEEDS = (17, 42, 73)
LAUNCHER_SHARDS = 3
LAUNCHER_ASSIGNMENT_POLICY = "balanced_teacher_rank_seed_latin_wave_v1"
LAUNCHER_WAVES = 4
RUNS_PER_WAVE = 3


def validate_protocol(config: Mapping[str, Any], *, require_frozen: bool = False) -> None:
    if config.get("protocol_variant") != PROTOCOL_VARIANT:
        raise ValueError(f"Unexpected main-matrix protocol: {config.get('protocol_variant')}")
    teachers = config.get("teachers", [])
    names = [str(row.get("name")) for row in teachers]
    if names != list(TEACHER_NAMES):
        raise ValueError(f"Teacher order must remain {TEACHER_NAMES}.")
    if str(teachers[0].get("role")) != "self_distillation_control":
        raise ValueError("The 1.5B teacher must be labeled self_distillation_control.")
    generation = dict(config.get("generation", {}))
    locked_generation = {
        "num_candidates": 16,
        "num_shards": 3,
        "base_seed": 20260825,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "cap_max_new_tokens_by_budget": False,
    }
    for key, expected in locked_generation.items():
        if generation.get(key) != expected:
            raise ValueError(
                f"Locked main-matrix generation field changed: {key}="
                f"{generation.get(key)!r}, expected={expected!r}"
            )
    selection = dict(config.get("relative_length_selection", {}))
    if selection.get("labels") != ["short", "medium", "long"]:
        raise ValueError("Rank labels must remain short, medium, and long.")
    if selection.get("method") != "shortest_lower_median_longest":
        raise ValueError("Rank-selection method changed.")
    if int(selection.get("minimum_unique_correct", 0)) != 3:
        raise ValueError("minimum_unique_correct must remain 3.")
    matrix = dict(config.get("matrix", {}))
    if [int(seed) for seed in matrix.get("training_seeds", [])] != list(TRAINING_SEEDS):
        raise ValueError(f"Training seeds must remain {TRAINING_SEEDS}.")
    if matrix.get("ranks") != list(RANK_NAMES):
        raise ValueError(f"Matrix ranks must remain {RANK_NAMES}.")
    if int(matrix.get("expected_adapter_count", 0)) != 36:
        raise ValueError("Main matrix must contain exactly 36 adapters.")
    if matrix.get("training_support") != "global_teacher_rank_common_problem_intersection":
        raise ValueError("Main matrix must train on the global teacher/rank common support.")
    if int(matrix.get("minimum_global_common_problems", 0)) != 500:
        raise ValueError("The registered global-support minimum must remain 500 problems.")
    student = dict(config.get("student", {}))
    if student.get("model_name") != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("The main-matrix student must remain Qwen2.5-1.5B-Instruct.")
    lora = dict(student.get("lora", {}))
    if (lora.get("r"), lora.get("alpha"), lora.get("dropout")) != (4, 16, 0.05):
        raise ValueError("The registered rank-4 LoRA recipe changed.")
    analysis = dict(config.get("analysis", {}))
    if analysis.get("within_teacher_pairs") != [
        ["relative_short", "relative_medium"],
        ["relative_medium", "relative_long"],
        ["relative_short", "relative_long"],
    ]:
        raise ValueError("Registered within-teacher contrast family changed.")
    if analysis.get("interaction_rank_pair") != ["relative_short", "relative_long"]:
        raise ValueError("Registered teacher-by-rank interaction endpoint changed.")
    if require_frozen:
        evidence = config.get("phase_a_evidence")
        if not isinstance(evidence, dict) or evidence.get("status") != "passed":
            raise ValueError("Frozen main matrix must bind passed Phase-A evidence.")


def matrix_run_name(teacher_name: str, rank_name: str, seed: int) -> str:
    if teacher_name not in TEACHER_NAMES:
        raise ValueError(f"Unexpected teacher: {teacher_name}")
    if rank_name not in RANK_NAMES:
        raise ValueError(f"Unexpected rank: {rank_name}")
    if int(seed) not in TRAINING_SEEDS:
        raise ValueError(f"Unexpected training seed: {seed}")
    return f"equal_example__{teacher_name}__{rank_name}__seed_{int(seed)}"


def ordered_matrix_runs() -> List[Dict[str, Any]]:
    """Return all cells in an order that balances ``index % 3`` launch sharding.

    The reusable training launcher historically assigns runs by list index
    modulo the launcher-shard count. A naive teacher/rank/seed product order
    therefore maps each training seed to a different node. The Latin-style
    assignment below distributes every teacher-rank cell's three seeds across
    all three nodes while balancing teacher, rank, and seed margins per node.
    """

    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(LAUNCHER_SHARDS)]
    for teacher_index, teacher in enumerate(TEACHER_NAMES):
        for rank_index, rank in enumerate(RANK_NAMES):
            for seed_index, seed in enumerate(TRAINING_SEEDS):
                shard_index = (teacher_index + rank_index + seed_index) % LAUNCHER_SHARDS
                wave_index = (teacher_index + rank_index) % LAUNCHER_WAVES
                buckets[shard_index].append(
                    {
                        "run_name": matrix_run_name(teacher, rank, seed),
                        "mode": "equal_example",
                        "generator_name": teacher,
                        "budget_name": rank,
                        "seed": seed,
                        "launcher_shards": LAUNCHER_SHARDS,
                        "launcher_shard_index": shard_index,
                        "launcher_wave_index": wave_index,
                        "launcher_assignment_policy": LAUNCHER_ASSIGNMENT_POLICY,
                    }
                )
    bucket_sizes = {len(bucket) for bucket in buckets}
    if bucket_sizes != {12}:
        raise ValueError(f"Unbalanced main-matrix launcher buckets: {sorted(bucket_sizes)}")
    for bucket in buckets:
        bucket.sort(
            key=lambda run: (
                int(run["launcher_wave_index"]),
                RANK_NAMES.index(str(run["budget_name"])),
                TEACHER_NAMES.index(str(run["generator_name"])),
                TRAINING_SEEDS.index(int(run["seed"])),
            )
        )
    ordered = [
        buckets[shard_index][position]
        for position in range(len(buckets[0]))
        for shard_index in range(LAUNCHER_SHARDS)
    ]
    validate_launcher_assignment(ordered)
    return ordered


def validate_launcher_assignment(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Validate and summarize the balanced three-node operational launch plan."""

    normalized = [dict(run) for run in runs]
    expected_names = {
        matrix_run_name(teacher, rank, seed)
        for teacher in TEACHER_NAMES
        for rank in RANK_NAMES
        for seed in TRAINING_SEEDS
    }
    observed_names = [str(run.get("run_name", "")) for run in normalized]
    if len(normalized) != 36 or set(observed_names) != expected_names:
        raise ValueError("Launcher assignment must cover the 36 unique registered runs.")
    if len(observed_names) != len(set(observed_names)):
        raise ValueError("Launcher assignment contains duplicate run identities.")
    for position, run in enumerate(normalized):
        shard_index = int(run.get("launcher_shard_index", -1))
        if int(run.get("launcher_shards", -1)) != LAUNCHER_SHARDS:
            raise ValueError(f"Launcher topology mismatch: {run.get('run_name')}")
        if shard_index != position % LAUNCHER_SHARDS:
            raise ValueError(f"Launcher order/assignment mismatch: {run.get('run_name')}")
        if run.get("launcher_assignment_policy") != LAUNCHER_ASSIGNMENT_POLICY:
            raise ValueError(f"Launcher assignment policy mismatch: {run.get('run_name')}")
    summaries: List[Dict[str, Any]] = []
    for shard_index in range(LAUNCHER_SHARDS):
        shard = [
            run for run in normalized if int(run["launcher_shard_index"]) == shard_index
        ]
        teacher_counts = Counter(str(run.get("generator_name")) for run in shard)
        rank_counts = Counter(str(run.get("budget_name")) for run in shard)
        seed_counts = Counter(int(run.get("seed")) for run in shard)
        if teacher_counts != Counter({teacher: 3 for teacher in TEACHER_NAMES}):
            raise ValueError(f"Teacher margin is unbalanced for launcher shard {shard_index}.")
        if rank_counts != Counter({rank: 4 for rank in RANK_NAMES}):
            raise ValueError(f"Rank margin is unbalanced for launcher shard {shard_index}.")
        if seed_counts != Counter({seed: 4 for seed in TRAINING_SEEDS}):
            raise ValueError(f"Seed margin is unbalanced for launcher shard {shard_index}.")
        wave_summaries: List[Dict[str, Any]] = []
        for local_position, run in enumerate(shard):
            expected_wave = local_position // RUNS_PER_WAVE
            if int(run.get("launcher_wave_index", -1)) != expected_wave:
                raise ValueError(
                    f"Launcher wave order mismatch for shard {shard_index}: {run.get('run_name')}"
                )
        for wave_index in range(LAUNCHER_WAVES):
            wave = [
                run for run in shard if int(run["launcher_wave_index"]) == wave_index
            ]
            wave_teachers = Counter(str(run["generator_name"]) for run in wave)
            wave_ranks = Counter(str(run["budget_name"]) for run in wave)
            if len(wave) != RUNS_PER_WAVE or set(wave_teachers.values()) != {1}:
                raise ValueError(
                    f"Teacher wave is unbalanced for shard {shard_index} wave {wave_index}."
                )
            if wave_ranks != Counter({rank: 1 for rank in RANK_NAMES}):
                raise ValueError(
                    f"Rank wave is unbalanced for shard {shard_index} wave {wave_index}."
                )
            wave_summaries.append(
                {
                    "launcher_wave_index": wave_index,
                    "run_count": len(wave),
                    "teacher_counts": dict(sorted(wave_teachers.items())),
                    "rank_counts": dict(sorted(wave_ranks.items())),
                    "seed_counts": {
                        str(key): value
                        for key, value in sorted(
                            Counter(int(run["seed"]) for run in wave).items()
                        )
                    },
                }
            )
        summaries.append(
            {
                "launcher_shard_index": shard_index,
                "run_count": len(shard),
                "teacher_counts": dict(sorted(teacher_counts.items())),
                "rank_counts": dict(sorted(rank_counts.items())),
                "seed_counts": {str(key): value for key, value in sorted(seed_counts.items())},
                "waves": wave_summaries,
            }
        )
    for teacher in TEACHER_NAMES:
        for rank in RANK_NAMES:
            cell_shards = {
                int(run["launcher_shard_index"])
                for run in normalized
                if run.get("generator_name") == teacher and run.get("budget_name") == rank
            }
            if cell_shards != set(range(LAUNCHER_SHARDS)):
                raise ValueError(f"Teacher-rank seeds are node-confounded: {teacher} {rank}")
    global_waves: List[Dict[str, Any]] = []
    for wave_index in range(LAUNCHER_WAVES):
        wave = [
            run for run in normalized if int(run["launcher_wave_index"]) == wave_index
        ]
        rank_counts = Counter(str(run["budget_name"]) for run in wave)
        seed_counts = Counter(int(run["seed"]) for run in wave)
        teacher_counts = Counter(str(run["generator_name"]) for run in wave)
        if rank_counts != Counter({rank: LAUNCHER_SHARDS for rank in RANK_NAMES}):
            raise ValueError(f"Global rank margin is unbalanced for wave {wave_index}.")
        if seed_counts != Counter({seed: LAUNCHER_SHARDS for seed in TRAINING_SEEDS}):
            raise ValueError(f"Global seed margin is unbalanced for wave {wave_index}.")
        if sorted(teacher_counts.values()) != [3, 3, 3]:
            raise ValueError(f"Global teacher spread is unbalanced for wave {wave_index}.")
        global_waves.append(
            {
                "launcher_wave_index": wave_index,
                "run_count": len(wave),
                "teacher_counts": dict(sorted(teacher_counts.items())),
                "rank_counts": dict(sorted(rank_counts.items())),
                "seed_counts": {str(key): value for key, value in sorted(seed_counts.items())},
            }
        )
    return {
        "launcher_shards": LAUNCHER_SHARDS,
        "launcher_assignment_policy": LAUNCHER_ASSIGNMENT_POLICY,
        "run_count": len(normalized),
        "launcher_waves": LAUNCHER_WAVES,
        "runs_per_shard_wave": RUNS_PER_WAVE,
        "per_shard": summaries,
        "global_waves": global_waves,
        "teacher_rank_seed_spread": "each teacher-rank cell spans all three launcher shards",
    }


def generation_config_for_teacher(
    frozen_protocol: Mapping[str, Any],
    teacher_name: str,
    *,
    protocol_path: str,
    protocol_file_sha256: str,
) -> Dict[str, Any]:
    """Materialize the single-teacher config consumed by phase-16 generation code."""

    validate_protocol(frozen_protocol, require_frozen=True)
    teacher_by_name = {str(row["name"]): dict(row) for row in frozen_protocol["teachers"]}
    if teacher_name not in teacher_by_name:
        raise ValueError(f"Unknown teacher: {teacher_name}")
    if teacher_name == "qwen2p5_7b":
        raise ValueError("The sealed 7B candidate pool must be reused, not regenerated.")
    parent_protocol = {
        key: value for key, value in frozen_protocol.items() if key != "_config_path"
    }
    return {
        "experiment_name": f"capacity_length_ranked_sampling_multiteacher_v1__{teacher_name}",
        "protocol_variant": "derived_single_teacher_ranked_generation",
        "parent_multiteacher_protocol": {
            "path": protocol_path,
            "canonical_sha256": canonical_sha256(parent_protocol),
            "file_sha256": protocol_file_sha256,
        },
        "dataset": dict(frozen_protocol["dataset"]),
        "cohort": dict(frozen_protocol["cohort"]),
        "teacher": teacher_by_name[teacher_name],
        "generation": dict(frozen_protocol["generation"]),
        "relative_length_selection": dict(frozen_protocol["relative_length_selection"]),
        "token_counter": dict(frozen_protocol["token_counter"]),
        "output": dict(frozen_protocol["generation_output"]),
    }
