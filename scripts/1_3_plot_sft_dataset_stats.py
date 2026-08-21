#!/usr/bin/env python3
"""Summarize and plot SFT dataset statistics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_DATASETS = {
    "small": "results/real_length_budget/sft_small.jsonl",
    "medium": "results/real_length_budget/sft_medium.jsonl",
    "large": "results/real_length_budget/sft_large.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset spec in name=path format. May be repeated. Defaults to small/medium/large SFT files.",
    )
    parser.add_argument("--output-json", default="data/sft_dataset_stats.json", help="Output JSON path.")
    parser.add_argument("--output-csv", default="data/sft_dataset_stats.csv", help="Output CSV path.")
    parser.add_argument("--figure", default="figures/sft_dataset_stats.png", help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset_specs = _parse_dataset_specs(args.dataset)
    summaries = []
    token_distributions: Dict[str, List[int]] = {}
    for name, path in dataset_specs.items():
        records = _read_jsonl(path)
        summary, token_counts = _summarize_dataset(name, path, records)
        summaries.append(summary)
        token_distributions[name] = token_counts

    _write_json(Path(args.output_json), {"datasets": summaries})
    _write_csv(Path(args.output_csv), summaries)
    figure_written = _maybe_plot(Path(args.figure), summaries, token_distributions)

    logging.info("wrote_json=%s", args.output_json)
    logging.info("wrote_csv=%s", args.output_csv)
    if figure_written:
        logging.info("wrote_figure=%s", args.figure)

    print("dataset,n,unique_problems,correct_rate,mean_solution_tokens,median_solution_tokens,p90_solution_tokens")
    for row in summaries:
        print(
            f"{row['dataset']},{row['n_examples']},{row['n_unique_problems']},"
            f"{row['correct_rate']:.4f},{row['mean_solution_tokens']:.2f},"
            f"{row['median_solution_tokens']:.2f},{row['p90_solution_tokens']:.2f}"
        )


def _parse_dataset_specs(raw_specs: Sequence[str] | None) -> Dict[str, Path]:
    if not raw_specs:
        return {name: Path(path) for name, path in DEFAULT_DATASETS.items()}
    specs: Dict[str, Path] = {}
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Dataset spec must use name=path format: {raw_spec}")
        name, path = raw_spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Dataset name is empty: {raw_spec}")
        specs[name] = Path(path.strip())
    return specs


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _summarize_dataset(
    name: str,
    path: Path,
    records: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[int]]:
    metadata = [record.get("metadata", {}) for record in records]
    problem_ids = [str(item.get("problem_id", "")) for item in metadata if item.get("problem_id")]
    token_counts = [_to_int(item.get("solution_token_count")) for item in metadata]
    token_counts = [value for value in token_counts if value is not None]
    prompt_chars = [len(str(record.get("prompt", ""))) for record in records]
    completion_chars = [len(str(record.get("completion", ""))) for record in records]
    correct_values = [bool(item.get("is_correct")) for item in metadata if "is_correct" in item]
    max_budget_values = [_to_int(item.get("max_solution_tokens")) for item in metadata]
    max_budget_values = [value for value in max_budget_values if value is not None]

    summary = {
        "dataset": name,
        "path": str(path),
        "n_examples": len(records),
        "n_unique_problems": len(set(problem_ids)),
        "correct_count": sum(correct_values),
        "correct_rate": _safe_mean([1.0 if value else 0.0 for value in correct_values]),
        "max_solution_tokens_budget": _most_common(max_budget_values),
        "mean_solution_tokens": _safe_mean(token_counts),
        "median_solution_tokens": _safe_median(token_counts),
        "p10_solution_tokens": _percentile(token_counts, 10),
        "p90_solution_tokens": _percentile(token_counts, 90),
        "min_solution_tokens": min(token_counts) if token_counts else None,
        "max_solution_tokens": max(token_counts) if token_counts else None,
        "mean_prompt_chars": _safe_mean(prompt_chars),
        "mean_completion_chars": _safe_mean(completion_chars),
    }
    return summary, token_counts


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(values: Sequence[float | int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_median(values: Sequence[float | int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _most_common(values: Sequence[int]) -> int | None:
    if not values:
        return None
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_plot(path: Path, summaries: List[Dict[str, Any]], token_distributions: Dict[str, List[int]]) -> bool:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipped figure=%s", path)
        return False

    labels = [row["dataset"] for row in summaries]
    colors = ["#4c78a8", "#f58518", "#54a24b"][: len(labels)]
    x_positions = list(range(len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    axes[0, 0].bar(x_positions, [row["n_examples"] for row in summaries], color=colors)
    axes[0, 0].set_title("Training examples")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].bar(x_positions, [row["correct_rate"] for row in summaries], color=colors)
    axes[0, 1].set_title("Teacher-correct examples")
    axes[0, 1].set_ylabel("Rate")
    axes[0, 1].set_ylim(0.0, 1.05)

    width = 0.36
    mean_positions = [x - width / 2 for x in x_positions]
    median_positions = [x + width / 2 for x in x_positions]
    axes[1, 0].bar(
        mean_positions,
        [row["mean_solution_tokens"] for row in summaries],
        width=width,
        color="#72b7b2",
        label="Mean",
    )
    axes[1, 0].bar(
        median_positions,
        [row["median_solution_tokens"] for row in summaries],
        width=width,
        color="#e45756",
        label="Median",
    )
    axes[1, 0].set_title("Solution token length")
    axes[1, 0].set_ylabel("Tokens")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].boxplot([token_distributions[label] for label in labels], tick_labels=labels, showfliers=False)
    axes[1, 1].set_title("Solution token distribution")
    axes[1, 1].set_ylabel("Tokens")

    for axis in axes.flat:
        if axis is not axes[1, 1]:
            axis.set_xticks(x_positions)
            axis.set_xticklabels(labels)
        axis.grid(axis="y", color="#d7dce0", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    fig.suptitle("SFT Training Dataset Statistics", y=0.99)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


if __name__ == "__main__":
    main()
