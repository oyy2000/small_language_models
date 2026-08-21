#!/usr/bin/env python3
"""Summarize existing student SFT grid eval results without launching eval jobs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="results/student_sft_grid/manifest.json",
        help="Manifest produced by 2_2_grid_search_student_sft.py.",
    )
    parser.add_argument(
        "--eval-dir",
        default="results/student_sft_grid/eval",
        help="Directory containing summaries/, predictions/, and aggregate outputs.",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Directory containing per-run summary JSON files. Defaults to <eval-dir>/summaries.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Aggregate JSON path. Defaults to <eval-dir>/grid_eval_summary.json.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Aggregate CSV path. Defaults to <eval-dir>/grid_eval_summary.csv.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include manifest runs without summary JSON as rows with status=missing.",
    )
    parser.add_argument(
        "--figure",
        default="figures/student_sft_grid_eval_accuracy.png",
        help="Optional accuracy figure path. Use an empty string to skip figure generation.",
    )
    parser.add_argument(
        "--base-accuracy",
        type=float,
        default=0.34,
        help="Base model accuracy reference line to draw on the figure.",
    )
    parser.add_argument("--base-label", default="Base model", help="Label for the base accuracy reference line.")
    parser.add_argument("--figure-top-k", type=int, default=20, help="Number of top evaluated runs to plot.")
    parser.add_argument("--top-k", type=int, default=10, help="Print this many top rows after writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest_path = Path(args.manifest)
    eval_dir = Path(args.eval_dir)
    summary_dir = Path(args.summary_dir) if args.summary_dir else eval_dir / "summaries"
    output_json = Path(args.output_json) if args.output_json else eval_dir / "grid_eval_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else eval_dir / "grid_eval_summary.csv"
    figure_path = Path(args.figure) if args.figure else None

    manifest_entries = _read_manifest(manifest_path)
    rows, missing = _collect_existing_summaries(manifest_entries, summary_dir, eval_dir, args.include_missing)
    evaluated_rows = [row for row in rows if row.get("status") == "evaluated"]
    figure_written = _maybe_write_accuracy_figure(
        figure_path,
        evaluated_rows,
        args.figure_top_k,
        args.base_accuracy,
        args.base_label,
    )

    payload = {
        "manifest": str(manifest_path),
        "summary_dir": str(summary_dir),
        "figure": str(figure_path) if figure_written and figure_path else None,
        "base_accuracy": args.base_accuracy,
        "base_label": args.base_label,
        "total_manifest_runs": len(manifest_entries),
        "evaluated_runs": len(evaluated_rows),
        "missing_runs": len(missing),
        "missing": missing,
        "runs": rows,
    }
    _write_json(output_json, payload)
    _write_csv(output_csv, rows)

    logging.info("wrote_json=%s", output_json)
    logging.info("wrote_csv=%s", output_csv)
    if figure_written:
        logging.info("wrote_figure=%s", figure_path)
    logging.info(
        "total_manifest_runs=%d evaluated_runs=%d missing_runs=%d",
        len(manifest_entries),
        len(evaluated_rows),
        len(missing),
    )
    _print_top_rows(evaluated_rows, args.top_k)


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"Manifest has no runs: {path}")
    return runs


def _collect_existing_summaries(
    manifest_entries: List[Dict[str, Any]],
    summary_dir: Path,
    eval_dir: Path,
    include_missing: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for entry in manifest_entries:
        run_name = entry["run_name"]
        summary_path = summary_dir / f"{run_name}.json"
        prediction_path = eval_dir / "predictions" / f"{run_name}.jsonl"
        if not summary_path.exists():
            missing_entry = {
                "run_name": run_name,
                "summary_json": str(summary_path),
                "adapter_path": entry.get("output_dir", ""),
            }
            missing.append(missing_entry)
            logging.warning("missing_summary run=%s path=%s", run_name, summary_path)
            if include_missing:
                rows.append(_missing_row(entry, summary_path, prediction_path))
            continue

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row = {
            "run_name": run_name,
            "status": "evaluated",
            "accuracy": summary.get("accuracy"),
            "correct": summary.get("correct"),
            "n": summary.get("n"),
            "model_name": summary.get("model_name"),
            "split": summary.get("split"),
            "adapter_path": summary.get("adapter_path", entry.get("output_dir", "")),
            "summary_json": str(summary_path),
            "output_jsonl": summary.get("output_jsonl", str(prediction_path)),
            "train_log_path": entry.get("log_path", ""),
        }
        row.update({f"override.{key}": value for key, value in entry.get("overrides", {}).items()})
        rows.append(row)

    return _sort_rows(rows), missing


def _missing_row(entry: Dict[str, Any], summary_path: Path, prediction_path: Path) -> Dict[str, Any]:
    row = {
        "run_name": entry["run_name"],
        "status": "missing",
        "accuracy": None,
        "correct": None,
        "n": None,
        "model_name": None,
        "split": None,
        "adapter_path": entry.get("output_dir", ""),
        "summary_json": str(summary_path),
        "output_jsonl": str(prediction_path),
        "train_log_path": entry.get("log_path", ""),
    }
    row.update({f"override.{key}": value for key, value in entry.get("overrides", {}).items()})
    return row


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> tuple[bool, float, str]:
        accuracy = row.get("accuracy")
        return (accuracy is None, -(accuracy or 0.0), row.get("run_name", ""))

    return sorted(rows, key=sort_key)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _maybe_write_accuracy_figure(
    path: Path | None,
    rows: List[Dict[str, Any]],
    top_k: int,
    base_accuracy: float | None,
    base_label: str,
) -> bool:
    if path is None or top_k <= 0 or not rows:
        return False
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        logging.warning("matplotlib is not installed; skipped figure=%s", path)
        return False

    plot_rows = rows[:top_k]
    labels = [_compact_run_label(row) for row in plot_rows]
    accuracies = [float(row.get("accuracy") or 0.0) for row in plot_rows]
    colors = [_train_size_color(row) for row in plot_rows]
    x_max_source = max(accuracies + ([base_accuracy] if base_accuracy is not None else []))

    height = max(4.8, 0.34 * len(plot_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, height))
    y_positions = list(range(len(plot_rows)))
    ax.barh(y_positions, accuracies, color=colors, edgecolor="#2f3437", linewidth=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(0.6, min(1.0, x_max_source + 0.08)))
    ax.set_xlabel("Accuracy")
    ax.set_title("Student SFT Grid Evaluation Accuracy")
    ax.grid(axis="x", color="#d7dce0", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if base_accuracy is not None:
        ax.axvline(base_accuracy, color="#d62728", linestyle="--", linewidth=1.6)
        ax.text(
            base_accuracy + 0.008,
            -0.7,
            f"{base_label}: {base_accuracy:.4f}",
            color="#9d1f1f",
            fontsize=9,
            va="center",
        )

    for y_position, row, accuracy in zip(y_positions, plot_rows, accuracies):
        correct = row.get("correct")
        n = row.get("n")
        label = f"{accuracy:.2f}"
        if correct is not None and n is not None:
            label = f"{label} ({correct}/{n})"
        ax.text(accuracy + 0.01, y_position, label, va="center", fontsize=8)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#2f3437", label=label)
        for label, color in _TRAIN_SIZE_COLORS.items()
    ]
    if base_accuracy is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle="--",
                linewidth=1.6,
                label=f"{base_label} ({base_accuracy:.4f})",
            )
        )
    ax.legend(handles=legend_handles, title="Training set", loc="lower right", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


def _compact_run_label(row: Dict[str, Any]) -> str:
    run_name = row.get("run_name", "")
    run_id = run_name.split("_")[1] if "_" in run_name else run_name
    train_size = _train_size(row)
    learning_rate = _format_value(row.get("override.training.learning_rate"))
    rank = _format_value(row.get("override.student.lora.r"))
    alpha = _format_value(row.get("override.student.lora.alpha"))
    return f"{run_id} {train_size} lr={learning_rate} r={rank} a={alpha}"


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


_TRAIN_SIZE_COLORS = {
    "small": "#4c78a8",
    "medium": "#f58518",
    "large": "#54a24b",
    "standard": "#72b7b2",
    "cod": "#e45756",
    "unknown": "#8c8c8c",
}


def _train_size(row: Dict[str, Any]) -> str:
    train_path = str(row.get("override.data.train_path", ""))
    for size in ("small", "medium", "large"):
        if f"sft_{size}" in train_path:
            return size
    if "standard_prompt" in train_path:
        return "standard"
    if "chain_of_draft" in train_path:
        return "cod"
    return "unknown"


def _train_size_color(row: Dict[str, Any]) -> str:
    return _TRAIN_SIZE_COLORS.get(_train_size(row), _TRAIN_SIZE_COLORS["unknown"])


def _print_top_rows(rows: List[Dict[str, Any]], top_k: int) -> None:
    if top_k <= 0 or not rows:
        return
    print("rank,accuracy,correct,n,run_name")
    for index, row in enumerate(rows[:top_k], start=1):
        print(
            f"{index},{row.get('accuracy')},{row.get('correct')},"
            f"{row.get('n')},{row.get('run_name')}"
        )


if __name__ == "__main__":
    main()
