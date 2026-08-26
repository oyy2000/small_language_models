"""Pure selection and advancement logic for the paired-rewrite pilot."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple


def gradient_clip_rate(
    log_history: Sequence[Mapping[str, Any]],
    *,
    through_epoch: float,
    max_grad_norm: float,
) -> Tuple[float, int]:
    norms = [
        float(row["grad_norm"])
        for row in log_history
        if row.get("grad_norm") is not None
        and row.get("epoch") is not None
        and float(row["epoch"]) <= through_epoch + 1e-9
        and math.isfinite(float(row["grad_norm"]))
    ]
    if not norms:
        return 0.0, 0
    return sum(value > max_grad_norm for value in norms) / len(norms), len(norms)


def select_shared_recipe(
    rows: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    accuracy_tie_pp: float,
) -> Dict[str, Any]:
    """Select one LR/epoch shared across conditions, then apply stability ties."""

    expected = set(conditions)
    grouped: Dict[Tuple[float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["learning_rate"]), float(row["epoch"]))].append(row)
    candidates = []
    for (learning_rate, epoch), group in sorted(grouped.items()):
        observed = {str(row["condition"]) for row in group}
        if observed != expected or len(group) != len(expected):
            continue
        accuracies = [float(row["accuracy"]) for row in group]
        clip_rates = [float(row["clip_rate"]) for row in group]
        candidates.append(
            {
                "learning_rate": learning_rate,
                "epoch": epoch,
                "macro_accuracy": sum(accuracies) / len(accuracies),
                "mean_clip_rate": sum(clip_rates) / len(clip_rates),
                "conditions": {str(row["condition"]): dict(row) for row in group},
            }
        )
    if not candidates:
        raise ValueError("No complete shared LR/epoch recipe matrix was found")
    best_accuracy = max(row["macro_accuracy"] for row in candidates)
    tolerance = float(accuracy_tie_pp) / 100.0
    tied = [row for row in candidates if row["macro_accuracy"] >= best_accuracy - tolerance]
    selected = min(
        tied,
        key=lambda row: (
            row["mean_clip_rate"],
            row["epoch"],
            row["learning_rate"],
        ),
    )
    return {
        **selected,
        "best_macro_accuracy": best_accuracy,
        "accuracy_tolerance": tolerance,
        "candidate_count": len(candidates),
        "tied_candidate_count": len(tied),
    }


def advancement_gate(
    *,
    baseline_accuracy: float,
    candidate_accuracy: float,
    baseline_output_tokens: float,
    candidate_output_tokens: float,
    max_accuracy_drop_pp: float,
    max_output_token_ratio: float,
) -> Dict[str, Any]:
    accuracy_drop_pp = 100.0 * (baseline_accuracy - candidate_accuracy)
    token_ratio = (
        candidate_output_tokens / baseline_output_tokens
        if baseline_output_tokens > 0
        else float("inf")
    )
    accuracy_pass = accuracy_drop_pp <= max_accuracy_drop_pp
    length_pass = token_ratio <= max_output_token_ratio
    return {
        "status": "pass" if accuracy_pass and length_pass else "fail",
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_drop_pp": accuracy_drop_pp,
        "max_accuracy_drop_pp": max_accuracy_drop_pp,
        "baseline_mean_output_tokens": baseline_output_tokens,
        "candidate_mean_output_tokens": candidate_output_tokens,
        "output_token_ratio": token_ratio,
        "max_output_token_ratio": max_output_token_ratio,
        "accuracy_pass": accuracy_pass,
        "length_pass": length_pass,
    }
