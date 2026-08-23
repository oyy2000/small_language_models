"""Reusable helpers for deterministic mixed-domain math SFT pilots."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def stable_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_question_sha256(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip()
    return stable_text_sha256(normalized)


def stable_rank(seed: int, *parts: object) -> str:
    payload = json.dumps([int(seed), *parts], ensure_ascii=False, separators=(",", ":"))
    return stable_text_sha256(payload)


def parse_math_level(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Could not parse MATH difficulty level from {value!r}")
    level = int(match.group(1))
    if level not in {1, 2, 3, 4, 5}:
        raise ValueError(f"MATH difficulty level must be in 1..5, got {level}")
    return level


def proportional_stratified_sample(
    rows: Sequence[Mapping[str, Any]],
    sample_count: int,
    *,
    stratum_fields: Sequence[str],
    seed: int,
    identity_field: str = "id",
    minimum_per_stratum: int = 1,
) -> List[Dict[str, Any]]:
    """Select a stable proportional sample with deterministic stratum coverage."""

    if sample_count <= 0 or sample_count > len(rows):
        raise ValueError(f"sample_count must be in 1..{len(rows)}, got {sample_count}")
    if minimum_per_stratum < 0:
        raise ValueError("minimum_per_stratum must be non-negative")

    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    identities = set()
    for row in rows:
        identity = str(row[identity_field])
        if identity in identities:
            raise ValueError(f"Duplicate sampling identity: {identity}")
        identities.add(identity)
        key = tuple(str(row[field]) for field in stratum_fields)
        groups[key].append(row)

    ordered_keys = sorted(groups)
    allocations = {
        key: min(minimum_per_stratum, len(groups[key]))
        for key in ordered_keys
    }
    allocated = sum(allocations.values())
    if allocated > sample_count:
        raise ValueError(
            f"minimum_per_stratum requires {allocated} rows, exceeding sample_count={sample_count}"
        )

    while allocated < sample_count:
        remaining = sample_count - allocated
        capacities = {key: len(groups[key]) - allocations[key] for key in ordered_keys}
        total_capacity = sum(capacities.values())
        if total_capacity < remaining:
            raise RuntimeError("Stratified allocation ran out of capacity")
        ideals = {
            key: (remaining * capacities[key] / total_capacity if total_capacity else 0.0)
            for key in ordered_keys
        }
        additions = {
            key: min(capacities[key], int(math.floor(ideals[key])))
            for key in ordered_keys
        }
        added = sum(additions.values())
        for key, value in additions.items():
            allocations[key] += value
        allocated += added
        if allocated == sample_count:
            break
        candidates = [
            key
            for key in ordered_keys
            if allocations[key] < len(groups[key])
        ]
        candidates.sort(
            key=lambda key: (
                -(ideals[key] - math.floor(ideals[key])),
                stable_rank(seed, "allocation", *key),
            )
        )
        if not candidates:
            raise RuntimeError("Stratified allocation could not assign the remainder")
        for key in candidates[: sample_count - allocated]:
            allocations[key] += 1
            allocated += 1

    selected: List[Dict[str, Any]] = []
    for key in ordered_keys:
        ranked = sorted(
            groups[key],
            key=lambda row: stable_rank(seed, "row", str(row[identity_field])),
        )
        selected.extend(dict(row) for row in ranked[: allocations[key]])
    selected.sort(key=lambda row: str(row[identity_field]))
    if len(selected) != sample_count:
        raise RuntimeError(f"Sample cardinality mismatch: expected={sample_count} actual={len(selected)}")
    return selected


def count_by_fields(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        counts[tuple(str(row[field]) for field in fields)] += 1
    return [
        {**{field: key[index] for index, field in enumerate(fields)}, "count": counts[key]}
        for key in sorted(counts)
    ]
