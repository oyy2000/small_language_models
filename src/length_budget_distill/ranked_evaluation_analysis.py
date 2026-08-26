"""Statistical summaries for ranked-length evaluation predictions."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, Mapping, Sequence

from .factorial_analysis import (
    exact_mcnemar_p_value,
    paired_cluster_bootstrap,
    wilson_interval,
)


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Prediction rows must be non-empty.")
    ids = [str(row["problem_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction rows contain duplicate problem IDs.")
    correct = sum(int(bool(row["is_correct"])) for row in rows)
    token_counts = [int(row["output_token_count"]) for row in rows]
    if any(value < 0 for value in token_counts):
        raise ValueError("Output token counts must be non-negative.")
    ci_low, ci_high = wilson_interval(correct, len(rows))
    extraction_failures = sum(
        1
        for row in rows
        if row.get("predicted_answer") is None or not str(row.get("predicted_answer", "")).strip()
    )
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "mean_output_tokens": mean(token_counts),
        "median_output_tokens": median(token_counts),
        "max_output_tokens": max(token_counts),
        "max_token_hit_count": sum(value >= 511 for value in token_counts),
        "max_token_hit_rate": sum(value >= 511 for value in token_counts) / len(rows),
        "answer_extraction_failures": extraction_failures,
        "answer_extraction_failure_rate": extraction_failures / len(rows),
    }


def paired_contrast(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("Paired predictions have non-identical problem support.")
    left_only = 0
    right_only = 0
    both_correct = 0
    both_wrong = 0
    effects: Dict[str, float] = {}
    for problem_id in sorted(left):
        left_correct = bool(left[problem_id]["is_correct"])
        right_correct = bool(right[problem_id]["is_correct"])
        effects[problem_id] = float(left_correct) - float(right_correct)
        if left_correct and right_correct:
            both_correct += 1
        elif left_correct:
            left_only += 1
        elif right_correct:
            right_only += 1
        else:
            both_wrong += 1
    bootstrap = paired_cluster_bootstrap(
        effects,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "n": len(effects),
        "accuracy_difference": bootstrap["estimate"],
        "accuracy_difference_ci_low": bootstrap["ci_low"],
        "accuracy_difference_ci_high": bootstrap["ci_high"],
        "bootstrap_p_value": bootstrap["p_value"],
        "mcnemar_p_value": exact_mcnemar_p_value(left_only, right_only),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }
