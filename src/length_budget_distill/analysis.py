"""Analysis helpers for length-budget trace outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from .records import TraceRecord


def summarize_by_budget(traces: Iterable[TraceRecord]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[TraceRecord]] = defaultdict(list)
    budget_order: List[str] = []
    for trace in traces:
        if trace.budget_name not in grouped:
            budget_order.append(trace.budget_name)
        grouped[trace.budget_name].append(trace)

    summary = []
    for budget_name in budget_order:
        bucket = grouped[budget_name]
        token_counts = [trace.solution_token_count for trace in bucket]
        correct_count = sum(1 for trace in bucket if trace.is_correct)
        summary.append(
            {
                "budget_name": budget_name,
                "n": len(bucket),
                "correct": correct_count,
                "correct_rate": correct_count / len(bucket) if bucket else 0.0,
                "avg_solution_tokens": mean(token_counts) if token_counts else 0.0,
                "min_solution_tokens": min(token_counts) if token_counts else 0,
                "max_solution_tokens": max(token_counts) if token_counts else 0,
                "declared_max_solution_tokens": bucket[0].max_solution_tokens if bucket else None,
            }
        )
    return summary


def write_summary_json(path: Path, summary: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_summary_csv(path: Path, summary: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget_name",
        "n",
        "correct",
        "correct_rate",
        "avg_solution_tokens",
        "min_solution_tokens",
        "max_solution_tokens",
        "declared_max_solution_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def maybe_plot_summary(path: Path, summary: List[Dict[str, Any]]) -> bool:
    if path is None:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["budget_name"] for row in summary]
    avg_tokens = [row["avg_solution_tokens"] for row in summary]
    correct_rates = [row["correct_rate"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].bar(labels, avg_tokens, color="#4c78a8")
    axes[0].set_title("Average solution tokens")
    axes[0].set_ylabel("Tokens")
    axes[1].bar(labels, correct_rates, color="#59a14f")
    axes[1].set_title("Verified-correct rate")
    axes[1].set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True
