"""Reusable statistics for the teacher-capacity by ranked-length main matrix."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .factorial import canonical_sha256
from .factorial_analysis import holm_adjust
from .ranked_multiseed_analysis import crossed_seed_problem_bootstrap, seed_problem_effects
from .ranked_multiteacher import TEACHER_NAMES, TRAINING_SEEDS


Prediction = Mapping[str, Mapping[str, Any]]
NestedPredictions = Mapping[str, Mapping[int, Mapping[str, Prediction]]]


def analyze_accuracy_contrasts(
    predictions: NestedPredictions,
    *,
    within_teacher_pairs: Sequence[Sequence[str]],
    interaction_rank_pair: Sequence[str],
    teacher_interaction_pairs: Sequence[Sequence[str]],
    bootstrap_samples: int,
    familywise_alpha: float,
    config_hash: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute the two preregistered Holm families on crossed seed/problem units."""

    _validate_nested_identities(predictions)
    within_contrasts: List[Dict[str, Any]] = []
    for teacher in TEACHER_NAMES:
        for pair in within_teacher_pairs:
            left_rank, right_rank = _pair(pair, "within-teacher rank")
            effects = seed_problem_effects(predictions[teacher], left_rank, right_rank)
            name = f"{teacher}__{left_rank}__vs__{right_rank}"
            within_contrasts.append(
                {
                    "contrast": name,
                    "teacher_name": teacher,
                    "left_rank": left_rank,
                    "right_rank": right_rank,
                    **crossed_seed_problem_bootstrap(
                        effects,
                        samples=int(bootstrap_samples),
                        seed=int(canonical_sha256([config_hash, "within", name])[:8], 16),
                    ),
                }
            )
    _apply_holm_family(
        within_contrasts,
        family_name="within_teacher_rank_12",
        familywise_alpha=familywise_alpha,
    )

    short_rank, long_rank = _pair(interaction_rank_pair, "interaction rank")
    rank_effects = {
        teacher: seed_problem_effects(predictions[teacher], short_rank, long_rank)
        for teacher in TEACHER_NAMES
    }
    interaction_rows: List[Dict[str, Any]] = []
    for pair in teacher_interaction_pairs:
        left_teacher, right_teacher = _pair(pair, "teacher interaction")
        if left_teacher not in rank_effects or right_teacher not in rank_effects:
            raise ValueError(f"Unknown teacher interaction pair: {pair}")
        effects = {
            seed: {
                problem_id: rank_effects[left_teacher][seed][problem_id]
                - rank_effects[right_teacher][seed][problem_id]
                for problem_id in rank_effects[left_teacher][seed]
            }
            for seed in TRAINING_SEEDS
        }
        name = f"{left_teacher}__vs__{right_teacher}__short_minus_long"
        interaction_rows.append(
            {
                "contrast": name,
                "left_teacher": left_teacher,
                "right_teacher": right_teacher,
                "rank_effect": f"{short_rank}_minus_{long_rank}",
                **crossed_seed_problem_bootstrap(
                    effects,
                    samples=int(bootstrap_samples),
                    seed=int(canonical_sha256([config_hash, "interaction", name])[:8], 16),
                ),
            }
        )
    _apply_holm_family(
        interaction_rows,
        family_name="teacher_by_rank_interaction_6",
        familywise_alpha=familywise_alpha,
    )
    return within_contrasts, interaction_rows


def _validate_nested_identities(predictions: NestedPredictions) -> None:
    if set(predictions) != set(TEACHER_NAMES):
        raise ValueError("Nested prediction teacher identities mismatch.")
    for teacher in TEACHER_NAMES:
        seeds = {int(seed) for seed in predictions[teacher]}
        if seeds != set(TRAINING_SEEDS):
            raise ValueError(f"Nested prediction seed identities mismatch: {teacher}")


def _apply_holm_family(
    rows: List[Dict[str, Any]], *, family_name: str, familywise_alpha: float
) -> None:
    adjusted = holm_adjust([float(row["bootstrap_p_value"]) for row in rows])
    for row, value in zip(rows, adjusted):
        row["family"] = family_name
        row["bootstrap_holm_p_value"] = value
        row["holm_significant"] = value < float(familywise_alpha)


def _pair(values: Sequence[str], label: str) -> Tuple[str, str]:
    if len(values) != 2:
        raise ValueError(f"{label} must contain exactly two identities: {values}")
    return str(values[0]), str(values[1])
