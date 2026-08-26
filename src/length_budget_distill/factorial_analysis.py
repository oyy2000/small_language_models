"""Statistical helpers for paired capacity-length comparisons."""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def wilson_interval(
    correct: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar_p_value(left_only: int, right_only: int) -> float:
    """Return the two-sided exact McNemar p-value for paired binary outcomes."""

    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    smaller = min(int(left_only), int(right_only))
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def paired_cluster_bootstrap(
    effects_by_problem: Mapping[str, float],
    samples: int = 10_000,
    seed: int = 20260820,
) -> Dict[str, float]:
    """Bootstrap a paired effect while clustering repeated seeds by problem."""

    if samples <= 0:
        raise ValueError("samples must be positive.")
    effects = [float(effects_by_problem[key]) for key in sorted(effects_by_problem)]
    if not effects:
        raise ValueError("At least one paired problem effect is required.")
    rng = random.Random(seed)
    bootstrap = []
    count = len(effects)
    for _ in range(samples):
        bootstrap.append(mean(effects[rng.randrange(count)] for _ in range(count)))
    bootstrap.sort()
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    non_positive = sum(value <= 0.0 for value in bootstrap) / samples
    non_negative = sum(value >= 0.0 for value in bootstrap) / samples
    return {
        "estimate": mean(effects),
        "ci_low": bootstrap[lower_index],
        "ci_high": bootstrap[upper_index],
        "p_value": min(1.0, 2.0 * min(non_positive, non_negative)),
        "problem_count": float(count),
        "bootstrap_samples": float(samples),
    }


def paired_problem_effects(
    left: Mapping[tuple[str, int], bool],
    right: Mapping[tuple[str, int], bool],
) -> Dict[str, float]:
    """Average paired correctness differences over seeds for each problem."""

    shared = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("Planned paired contrast has non-identical problem/seed support.")
    grouped: Dict[str, List[float]] = {}
    for problem_id, seed in shared:
        grouped.setdefault(problem_id, []).append(float(left[(problem_id, seed)]) - float(right[(problem_id, seed)]))
    return {problem_id: mean(values) for problem_id, values in grouped.items()}


def difference_in_differences_effects(
    large_short: Mapping[tuple[str, int], bool],
    large_long: Mapping[tuple[str, int], bool],
    small_short: Mapping[tuple[str, int], bool],
    small_long: Mapping[tuple[str, int], bool],
) -> Dict[str, float]:
    supports = [set(mapping) for mapping in (large_short, large_long, small_short, small_long)]
    if not all(support == supports[0] for support in supports[1:]):
        raise ValueError("Difference-in-differences inputs have non-identical support.")
    grouped: Dict[str, List[float]] = {}
    for problem_id, seed in sorted(supports[0]):
        effect = (
            float(large_short[(problem_id, seed)])
            - float(large_long[(problem_id, seed)])
            - float(small_short[(problem_id, seed)])
            + float(small_long[(problem_id, seed)])
        )
        grouped.setdefault(problem_id, []).append(effect)
    return {problem_id: mean(values) for problem_id, values in grouped.items()}
