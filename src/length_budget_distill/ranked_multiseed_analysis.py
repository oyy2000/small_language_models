"""Statistics for paired ranked-length comparisons across training seeds."""

from __future__ import annotations

import random
from statistics import mean, stdev
from typing import Any, Dict, Mapping


def seed_problem_effects(
    predictions_by_seed: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    left_rank: str,
    right_rank: str,
) -> Dict[int, Dict[str, float]]:
    """Return paired correctness effects with identical problem support."""

    effects: Dict[int, Dict[str, float]] = {}
    for seed, by_rank in sorted(predictions_by_seed.items()):
        if left_rank not in by_rank or right_rank not in by_rank:
            raise ValueError(f"Missing rank for seed {seed}: {left_rank} or {right_rank}")
        left = by_rank[left_rank]
        right = by_rank[right_rank]
        if set(left) != set(right):
            raise ValueError(f"Paired predictions have non-identical support for seed {seed}.")
        effects[int(seed)] = {
            problem_id: float(bool(left[problem_id]["is_correct"]))
            - float(bool(right[problem_id]["is_correct"]))
            for problem_id in sorted(left)
        }
    _validate_effect_support(effects)
    return effects


def crossed_seed_problem_bootstrap(
    effects_by_seed: Mapping[int, Mapping[str, float]],
    *,
    samples: int = 10_000,
    seed: int = 20260826,
) -> Dict[str, Any]:
    """Bootstrap training seeds and shared paired problems as crossed units.

    The same resampled problem identities are used for every sampled training
    seed. This preserves the paired evaluation cohort while incorporating the
    observed training-seed variability. With only three seeds, the percentile
    interval is descriptive and should not be treated as a precise population
    estimate.
    """

    if samples <= 0:
        raise ValueError("samples must be positive.")
    seeds, problem_ids = _validate_effect_support(effects_by_seed)
    if len(seeds) < 2:
        raise ValueError("At least two training seeds are required.")
    matrix = {
        seed_value: [float(effects_by_seed[seed_value][problem_id]) for problem_id in problem_ids]
        for seed_value in seeds
    }
    per_seed = {seed_value: mean(matrix[seed_value]) for seed_value in seeds}
    estimate = mean(per_seed.values())
    rng = random.Random(seed)
    seed_count = len(seeds)
    problem_count = len(problem_ids)
    bootstrap = []
    for _ in range(samples):
        problem_indices = [rng.randrange(problem_count) for _ in range(problem_count)]
        problem_resampled_seed_means = {
            seed_value: mean(matrix[seed_value][index] for index in problem_indices)
            for seed_value in seeds
        }
        sampled_seeds = [seeds[rng.randrange(seed_count)] for _ in range(seed_count)]
        bootstrap.append(mean(problem_resampled_seed_means[value] for value in sampled_seeds))
    bootstrap.sort()
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    non_positive = sum(value <= 0.0 for value in bootstrap) / samples
    non_negative = sum(value >= 0.0 for value in bootstrap) / samples
    return {
        "estimate": estimate,
        "ci_low": bootstrap[lower_index],
        "ci_high": bootstrap[upper_index],
        "bootstrap_p_value": min(1.0, 2.0 * min(non_positive, non_negative)),
        "seed_count": seed_count,
        "problem_count": problem_count,
        "bootstrap_samples": samples,
        "per_seed_effects": {str(key): value for key, value in per_seed.items()},
        "per_seed_sample_sd": stdev(per_seed.values()),
        "resampling_units": ["training_seed", "paired_problem"],
    }


def _validate_effect_support(
    effects_by_seed: Mapping[int, Mapping[str, float]],
) -> tuple[list[int], list[str]]:
    if not effects_by_seed:
        raise ValueError("At least one training seed is required.")
    seeds = sorted(int(seed) for seed in effects_by_seed)
    if len(seeds) != len(effects_by_seed):
        raise ValueError("Training seed identities must be unique integers.")
    support: set[str] | None = None
    for seed in seeds:
        current = set(effects_by_seed[seed])
        if not current:
            raise ValueError(f"Seed {seed} has no paired problem effects.")
        if support is None:
            support = current
        elif current != support:
            raise ValueError("Training seeds have non-identical problem support.")
    assert support is not None
    return seeds, sorted(support)
