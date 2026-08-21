"""Helpers for materializing JSONL files by length budget."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .records import read_jsonl, write_jsonl

DEFAULT_BUDGET_NAMES = ("small", "medium", "large")


def parse_budget_names(value: str) -> Sequence[str]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("At least one budget name is required.")
    return names


def _get_budget_name(record: Mapping[str, Any], input_path: Path) -> str:
    top_level_budget = record.get("budget_name")
    if isinstance(top_level_budget, str) and top_level_budget:
        return top_level_budget

    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_budget = metadata.get("budget_name")
        if isinstance(metadata_budget, str) and metadata_budget:
            return metadata_budget

    row_id = record.get("id", record.get("trace_id", "<unknown>"))
    raise ValueError(f"JSONL row {row_id!r} in {input_path} is missing budget_name.")


def split_records_by_budget(
    input_path: Path,
    output_dir: Path,
    budget_names: Sequence[str] = DEFAULT_BUDGET_NAMES,
    output_prefix: str = "sft",
    ignore_unknown: bool = False,
) -> Dict[str, int]:
    """Split one merged JSONL file into one JSONL file per budget name."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_budget: Dict[str, List[Dict[str, Any]]] = {name: [] for name in budget_names}
    unknown_counts: Counter[str] = Counter()

    for record in read_jsonl(input_path):
        budget_name = _get_budget_name(record, input_path)
        if budget_name not in records_by_budget:
            unknown_counts[budget_name] += 1
            continue
        records_by_budget[budget_name].append(record)

    if unknown_counts and not ignore_unknown:
        formatted = ", ".join(f"{name}={count}" for name, count in sorted(unknown_counts.items()))
        expected = ", ".join(budget_names)
        raise ValueError(f"Unexpected budget names in {input_path}: {formatted}; expected one of: {expected}")

    counts: Dict[str, int] = {}
    for budget_name, records in records_by_budget.items():
        output_path = output_dir / f"{output_prefix}_{budget_name}.jsonl"
        counts[budget_name] = write_jsonl(output_path, records)
    return counts
