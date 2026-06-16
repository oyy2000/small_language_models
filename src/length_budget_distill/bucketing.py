"""Length-budget validation and shard helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def get_length_budgets(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    budgets = list(config.get("length_budgets", []))
    if not budgets:
        raise ValueError("Config must define at least one length budget.")

    names = set()
    for budget in budgets:
        name = budget.get("name")
        max_tokens = budget.get("max_solution_tokens")
        if not name:
            raise ValueError(f"Length budget is missing a name: {budget}")
        if name in names:
            raise ValueError(f"Duplicate length budget name: {name}")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"Budget {name} must define a positive integer max_solution_tokens.")
        names.add(name)
    return budgets


def iter_shard(items: Sequence[T], num_shards: int, shard_index: int) -> Iterator[Tuple[int, T]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    for index, item in enumerate(items):
        if index % num_shards == shard_index:
            yield index, item

