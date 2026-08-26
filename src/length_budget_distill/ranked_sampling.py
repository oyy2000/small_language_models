"""Selection helpers for relative-length teacher sampling."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .config import resolve_path
from .factorial import file_sha256
from .records import ProblemRecord, TraceRecord


LENGTH_LABELS = ("short", "medium", "long")
SELECTION_METHOD = "shortest_lower_median_longest"
DEDUPLICATION_METHOD = "whitespace_normalized_exact_text"


def build_length_agnostic_teacher_prompt(problem: ProblemRecord) -> str:
    """Build one shared prompt whose candidates are not assigned length targets."""

    return (
        "You are a careful math teacher. Solve the problem correctly with visible "
        "step-by-step reasoning. Do not target a particular response length; use the "
        "amount of detail that follows naturally from your solution. End with a line "
        "in the form: Answer: <final answer>.\n\n"
        f"Problem:\n{problem.question}"
    )


def normalized_completion_key(text: str) -> str:
    """Normalize inconsequential whitespace before exact-completion deduplication."""

    return re.sub(r"\s+", " ", text.strip())


def unique_correct_candidates(traces: Iterable[TraceRecord]) -> List[TraceRecord]:
    """Return deterministic, exact-text-deduplicated answer-correct candidates."""

    ordered = sorted(traces, key=lambda item: (item.candidate_index, item.trace_id))
    seen: set[str] = set()
    eligible: List[TraceRecord] = []
    for trace in ordered:
        if not trace.is_correct:
            continue
        key = normalized_completion_key(trace.solution)
        if not key or key in seen:
            continue
        seen.add(key)
        eligible.append(trace)
    return eligible


def select_relative_length_candidates(
    traces: Sequence[TraceRecord],
    minimum_unique_correct: int = 3,
) -> Dict[str, TraceRecord]:
    """Select the shortest, lower-median, and longest unique correct candidates.

    The function returns an empty mapping when the problem does not have enough
    unique answer-correct candidates. Selection is based on completion token count;
    candidate index and trace ID provide deterministic tie-breaking.
    """

    if minimum_unique_correct < 3:
        raise ValueError("minimum_unique_correct must be at least 3.")
    problem_ids = {trace.problem_id for trace in traces}
    if len(problem_ids) > 1:
        raise ValueError(f"Expected candidates for one problem, observed={sorted(problem_ids)}")

    eligible = unique_correct_candidates(traces)
    if len(eligible) < minimum_unique_correct:
        return {}
    ranked = sorted(
        eligible,
        key=lambda item: (item.solution_token_count, item.candidate_index, item.trace_id),
    )
    selected_indices = {
        "short": 0,
        "medium": (len(ranked) - 1) // 2,
        "long": len(ranked) - 1,
    }
    selected: Dict[str, TraceRecord] = {}
    for label in LENGTH_LABELS:
        rank = selected_indices[label]
        winner = ranked[rank]
        metadata = dict(winner.metadata)
        metadata["relative_length_selection"] = {
            "label": label,
            "method": SELECTION_METHOD,
            "deduplication": DEDUPLICATION_METHOD,
            "rank_zero_based": rank,
            "eligible_unique_correct_count": len(ranked),
            "candidate_pool_count": len(traces),
        }
        selected[label] = replace(
            winner,
            budget_name=f"relative_{label}",
            selected_for_sft=True,
            metadata=metadata,
        )
    if len({trace.trace_id for trace in selected.values()}) != len(LENGTH_LABELS):
        raise AssertionError("Relative-length selection must choose three distinct candidates.")
    return selected


def select_relative_lengths_by_problem(
    traces: Iterable[TraceRecord],
    minimum_unique_correct: int = 3,
) -> Dict[str, Dict[str, TraceRecord]]:
    """Group a candidate pool by problem and select its three relative lengths."""

    grouped: Dict[str, List[TraceRecord]] = defaultdict(list)
    for trace in traces:
        grouped[trace.problem_id].append(trace)
    return {
        problem_id: selected
        for problem_id in sorted(grouped)
        if (
            selected := select_relative_length_candidates(
                grouped[problem_id],
                minimum_unique_correct=minimum_unique_correct,
            )
        )
    }


def load_bound_problem_ids(config: Mapping[str, Any]) -> List[str]:
    """Load and hash-check the configured JSON cohort manifest."""

    cohort = config.get("cohort", {})
    path_value = cohort.get("problem_ids_path")
    field = str(cohort.get("problem_ids_field", "problem_ids"))
    if not path_value:
        raise ValueError("cohort.problem_ids_path is required.")
    path = resolve_path(str(path_value), dict(config))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Cohort manifest does not exist: {path_value}")

    expected_sha256 = cohort.get("problem_ids_file_sha256")
    actual_sha256 = file_sha256(path)
    if expected_sha256 and str(expected_sha256) != actual_sha256:
        raise ValueError(
            "Cohort manifest hash mismatch: "
            f"expected={expected_sha256} actual={actual_sha256} path={path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if field not in payload or not isinstance(payload[field], list):
        raise ValueError(f"Cohort manifest field {field!r} must be a list: {path}")
    problem_ids = [str(item) for item in payload[field]]
    if not problem_ids:
        raise ValueError("Configured cohort is empty.")
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError("Configured cohort contains duplicate problem IDs.")
    expected_count = cohort.get("expected_problem_count")
    if expected_count is not None and len(problem_ids) != int(expected_count):
        raise ValueError(
            f"Cohort cardinality mismatch: expected={expected_count} actual={len(problem_ids)}"
        )
    return problem_ids


def validate_ranked_sampling_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported protocol variants instead of silently changing selection."""

    generation = config.get("generation", {})
    selection = config.get("relative_length_selection", {})
    if int(generation.get("num_candidates", 0)) < 3:
        raise ValueError("generation.num_candidates must be at least 3.")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive.")
    if int(generation.get("num_shards", 1)) <= 0:
        raise ValueError("generation.num_shards must be positive.")
    labels = tuple(selection.get("labels", LENGTH_LABELS))
    if labels != LENGTH_LABELS:
        raise ValueError(f"relative_length_selection.labels must equal {LENGTH_LABELS}.")
    if selection.get("method", SELECTION_METHOD) != SELECTION_METHOD:
        raise ValueError(f"Only selection method {SELECTION_METHOD!r} is supported.")
    if selection.get("deduplication", DEDUPLICATION_METHOD) != DEDUPLICATION_METHOD:
        raise ValueError(f"Only deduplication method {DEDUPLICATION_METHOD!r} is supported.")
    if int(selection.get("minimum_unique_correct", 3)) < 3:
        raise ValueError("relative_length_selection.minimum_unique_correct must be at least 3.")
    if selection.get("insufficient_candidate_policy", "drop_problem_from_all_labels") != (
        "drop_problem_from_all_labels"
    ):
        raise ValueError(
            "Only insufficient-candidate policy 'drop_problem_from_all_labels' is supported."
        )


def require_cohort_problems(
    problems: Sequence[ProblemRecord],
    problem_ids: Sequence[str],
) -> List[ProblemRecord]:
    """Return the bound cohort in manifest order and fail on missing source rows."""

    by_id = {problem.problem_id: problem for problem in problems}
    if len(by_id) != len(problems):
        raise ValueError("Loaded source dataset contains duplicate problem IDs.")
    missing = [problem_id for problem_id in problem_ids if problem_id not in by_id]
    if missing:
        raise ValueError(f"Cohort problem IDs are missing from the source dataset: {missing[:10]}")
    return [by_id[problem_id] for problem_id in problem_ids]
