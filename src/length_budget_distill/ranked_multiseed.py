"""Shared protocol helpers for ranked-length multi-seed extensions."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Set


LENGTH_LABELS = ("short", "medium", "long")
PROTOCOL_VARIANT = "comparative_multiseed_ranked_length_sft_extension"


def manifest_field_equal(field_name: str, left: Any, right: Any) -> bool:
    """Compare prepared and launch fields while normalizing marker seed text."""

    if field_name == "seed":
        try:
            return int(left) == int(right)
        except (TypeError, ValueError):
            return False
    return left == right


def validate_training_scope(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a multi-seed ranked-length training scope."""

    if str(config.get("protocol_variant")) != PROTOCOL_VARIANT:
        raise ValueError(f"Unexpected ranked multi-seed protocol: {config.get('protocol_variant')}")
    scope = dict(config.get("training_scope", {}))
    if tuple(scope.get("labels", [])) != LENGTH_LABELS:
        raise ValueError(f"Training labels must equal {LENGTH_LABELS}.")
    if str(scope.get("mode")) != "equal_example":
        raise ValueError("Ranked multi-seed extension must use equal_example supervision.")
    seeds = [int(seed) for seed in scope.get("training_seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be a non-empty unique list.")
    if seeds != sorted(seeds):
        raise ValueError("Training seeds must be sorted for deterministic run ordering.")
    if 17 in seeds:
        raise ValueError("Seed 17 belongs to the sealed parent experiment and must not be retrained.")
    expected_run_count = len(LENGTH_LABELS) * len(seeds)
    if int(scope.get("run_count", -1)) != expected_run_count:
        raise ValueError(
            f"Run count mismatch: expected={expected_run_count} actual={scope.get('run_count')}"
        )
    scope["training_seeds"] = seeds
    return scope


def ranked_run_name(generator_name: str, label: str, seed: int) -> str:
    if label not in LENGTH_LABELS:
        raise ValueError(f"Unknown ranked-length label: {label}")
    if not generator_name:
        raise ValueError("Generator name must be non-empty.")
    return f"equal_example__{generator_name}__relative_{label}__seed_{int(seed)}"


def expected_run_names(
    generator_name: str,
    seeds: Sequence[int],
) -> Set[str]:
    return {
        ranked_run_name(generator_name, label, int(seed))
        for seed in seeds
        for label in LENGTH_LABELS
    }


def ordered_run_specs(
    generator_name: str,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    return [
        {
            "run_name": ranked_run_name(generator_name, label, int(seed)),
            "label": label,
            "budget_name": f"relative_{label}",
            "seed": int(seed),
        }
        for seed in seeds
        for label in LENGTH_LABELS
    ]
