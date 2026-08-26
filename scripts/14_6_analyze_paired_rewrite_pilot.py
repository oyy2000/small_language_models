#!/usr/bin/env python3
"""Select the training recipe or analyze the paired-rewrite budget frontier."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.paired_rewrite_analysis import (
    advancement_gate,
    gradient_clip_rate,
    select_shared_recipe,
)
from length_budget_distill.records import write_jsonl


CONDITION_LABELS = {
    "standard_original": "Standard 7B-long",
    "rewrite_80": "Paired rewrite 80%",
    "rewrite_65": "Paired rewrite 65%",
    "direct_short": "Direct short",
    "progressive_standard_to_rewrite_65": "Standard then rewrite 65%",
    "matched_rewrite_65_extra": "Rewrite 65% plus matched stage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_paired_rewrite_7b_pilot_v1.json")
    parser.add_argument("--mode", choices=["selection", "final"], required=True)
    parser.add_argument("--eval-manifest-glob", required=True)
    parser.add_argument("--training-manifest-glob", default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    eval_tasks = _load_eval_tasks(args.eval_manifest_glob, config_hash)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "selection":
        if not args.training_manifest_glob or not args.dataset_manifest:
            raise ValueError("Selection analysis requires training and dataset manifests")
        _run_selection(
            config,
            eval_tasks,
            args.training_manifest_glob,
            Path(args.dataset_manifest),
            output_dir,
            figure_dir,
        )
    else:
        if not args.selection_json:
            raise ValueError("Final analysis requires --selection-json")
        _run_final(config, eval_tasks, Path(args.selection_json), output_dir, figure_dir)


def _run_selection(
    config: Mapping[str, Any],
    eval_tasks: Sequence[Mapping[str, Any]],
    training_pattern: str,
    dataset_manifest_path: Path,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    training_manifests = _training_by_run(training_pattern, expected_stage="grid")
    dataset_manifest = _read_json(dataset_manifest_path)
    rows: List[Dict[str, Any]] = []
    max_grad_norm = float(config["training"]["max_grad_norm"])
    for task in eval_tasks:
        if task.get("decoding_mode") != "greedy":
            continue
        summary = _read_json(Path(str(task["summary_path"])))
        training = training_manifests[str(task["run_name"])]
        clip_rate, grad_observations = gradient_clip_rate(
            training.get("training_log_history", []),
            through_epoch=float(task["epoch"]),
            max_grad_norm=max_grad_norm,
        )
        loss_rows = [
            row
            for row in training.get("training_log_history", [])
            if row.get("loss") is not None
            and row.get("epoch") is not None
            and float(row["epoch"]) <= float(task["epoch"]) + 1e-9
        ]
        rows.append(
            {
                "model_id": task["model_id"],
                "run_name": task["run_name"],
                "condition": task["condition"],
                "learning_rate": float(task["learning_rate"]),
                "epoch": float(task["epoch"]),
                "snapshot_path": task["adapter_path"],
                "accuracy": float(summary["greedy_accuracy"]),
                "mean_output_tokens": float(summary["mean_output_tokens"]),
                "max_token_hit_rate": float(summary["max_token_hit_rate"]),
                "eos_finish_rate": float(summary["eos_finish_rate"]),
                "clip_rate": clip_rate,
                "grad_norm_observations": grad_observations,
                "last_logged_loss": float(loss_rows[-1]["loss"]) if loss_rows else None,
                "min_logged_loss": min(float(row["loss"]) for row in loss_rows) if loss_rows else None,
            }
        )
    selected = select_shared_recipe(
        rows,
        conditions=[str(value) for value in config["selection"]["conditions"]],
        accuracy_tie_pp=float(config["selection"]["accuracy_tie_pp"]),
    )
    selection = {
        "status": "complete",
        "stage": "recipe_selection",
        "evidence_level": "exploratory_single_seed_pilot",
        "selection_rule": "macro accuracy; within tie tolerance use lower clip rate, fewer epochs, lower LR",
        "selected_learning_rate": selected["learning_rate"],
        "selected_epoch": selected["epoch"],
        "selected_macro_accuracy": selected["macro_accuracy"],
        "best_macro_accuracy": selected["best_macro_accuracy"],
        "selected_mean_clip_rate": selected["mean_clip_rate"],
        "candidate_count": selected["candidate_count"],
        "tied_candidate_count": selected["tied_candidate_count"],
        "selected_snapshots": {
            condition: evidence["snapshot_path"]
            for condition, evidence in selected["conditions"].items()
        },
        "selected_condition_metrics": selected["conditions"],
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
    }
    metrics_path = output_dir / "selection_grid_metrics.jsonl"
    selection_path = output_dir / "recipe_selection.json"
    if metrics_path.exists() or selection_path.exists():
        raise FileExistsError("Refusing to overwrite selection analysis")
    write_jsonl(metrics_path, sorted(rows, key=lambda row: (row["condition"], row["learning_rate"], row["epoch"])))
    _write_json(selection_path, selection)
    _plot_dataset_lengths(dataset_manifest, figure_dir)
    _plot_selection_grid(rows, figure_dir)
    _plot_training_diagnostics(training_manifests, max_grad_norm, figure_dir)
    logging.info("selected_lr=%g selected_epoch=%g", selected["learning_rate"], selected["epoch"])


def _run_final(
    config: Mapping[str, Any],
    eval_tasks: Sequence[Mapping[str, Any]],
    selection_path: Path,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    selection = _read_json(selection_path)
    rows: List[Dict[str, Any]] = []
    for task in eval_tasks:
        summary = _read_json(Path(str(task["summary_path"])))
        rows.append(
            {
                "model_id": task["model_id"],
                "condition": task["condition"],
                "schedule": task.get("schedule"),
                "max_new_tokens": int(task["max_new_tokens"]),
                "decoding_mode": task["decoding_mode"],
                "num_samples": int(task["num_samples"]),
                "accuracy": float(summary["greedy_accuracy"]),
                "pass_at_k": float(summary["pass_at_k"]),
                "mean_output_tokens": float(summary["mean_output_tokens"]),
                "max_token_hit_rate": float(summary["max_token_hit_rate"]),
                "eos_finish_rate": float(summary["eos_finish_rate"]),
                "answer_extraction_failures": int(summary["answer_extraction_failures"]),
            }
        )
    primary_cap = int(config["evaluation"]["primary_max_new_tokens"])
    primary = {
        row["model_id"]: row
        for row in rows
        if row["decoding_mode"] == "greedy" and row["max_new_tokens"] == primary_cap
    }
    baseline = primary.get("selected__standard_original")
    candidate = primary.get("selected__rewrite_65")
    if baseline is None or candidate is None:
        raise ValueError("Final evaluation lacks selected standard and rewrite-65 primary summaries")
    gate = advancement_gate(
        baseline_accuracy=baseline["accuracy"],
        candidate_accuracy=candidate["accuracy"],
        baseline_output_tokens=baseline["mean_output_tokens"],
        candidate_output_tokens=candidate["mean_output_tokens"],
        max_accuracy_drop_pp=float(config["advancement_gate"]["max_accuracy_drop_pp"]),
        max_output_token_ratio=float(config["advancement_gate"]["max_output_token_ratio"]),
    )
    progressive = primary.get("progressive_standard_to_rewrite_65")
    progressive_gate = None
    if progressive is not None:
        progressive_gate = advancement_gate(
            baseline_accuracy=baseline["accuracy"],
            candidate_accuracy=progressive["accuracy"],
            baseline_output_tokens=baseline["mean_output_tokens"],
            candidate_output_tokens=progressive["mean_output_tokens"],
            max_accuracy_drop_pp=float(config["advancement_gate"]["max_accuracy_drop_pp"]),
            max_output_token_ratio=float(config["advancement_gate"]["max_output_token_ratio"]),
        )
    metrics_path = output_dir / "confirmatory_metrics.jsonl"
    summary_path = output_dir / "paired_rewrite_pilot_summary.json"
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite final paired-rewrite analysis")
    write_jsonl(metrics_path, sorted(rows, key=lambda row: (row["model_id"], row["decoding_mode"], row["max_new_tokens"])))
    _write_json(
        summary_path,
        {
            "status": "complete",
            "evidence_level": "exploratory_single_seed_pilot",
            "scope": "GSM8K only",
            "recipe_selection": str(selection_path),
            "recipe_selection_sha256": file_sha256(selection_path),
            "primary_max_new_tokens": primary_cap,
            "paired_rewrite_65_advancement_gate": gate,
            "progressive_advancement_gate": progressive_gate,
            "primary_results": primary,
        },
    )
    _plot_budget_accuracy(rows, figure_dir)
    _plot_accuracy_length_frontier(rows, primary_cap, figure_dir)


def _load_eval_tasks(pattern: str, config_hash: str) -> List[Dict[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No evaluation manifests matched {pattern!r}")
    tasks: List[Dict[str, Any]] = []
    seen = set()
    for path in paths:
        manifest = _read_json(path)
        if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
            raise ValueError(f"Incomplete or config-mismatched evaluation manifest: {path}")
        for task in manifest["tasks"]:
            if task.get("eval_status") not in {"complete", "skipped_complete"}:
                raise ValueError(f"Incomplete evaluation task: {task.get('task_id')}")
            if task["task_id"] in seen:
                raise ValueError(f"Duplicate evaluation task: {task['task_id']}")
            seen.add(task["task_id"])
            tasks.append(dict(task))
    return tasks


def _training_by_run(pattern: str, *, expected_stage: str) -> Dict[str, Dict[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No training manifests matched {pattern!r}")
    by_run: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        manifest = _read_json(path)
        if manifest.get("status") != "complete" or manifest.get("stage") != expected_stage:
            continue
        run_name = str(manifest["run_name"])
        if run_name in by_run:
            raise ValueError(f"Duplicate training run: {run_name}")
        by_run[run_name] = manifest
    if not by_run:
        raise ValueError(f"No complete {expected_stage} training manifests")
    return by_run


def _plot_dataset_lengths(manifest: Mapping[str, Any], figure_dir: Path) -> None:
    plt = _pyplot()
    rows = sorted(manifest["runs"], key=lambda row: ["standard_original", "rewrite_80", "rewrite_65", "direct_short"].index(row["condition"]))
    labels = [CONDITION_LABELS[row["condition"]] for row in rows]
    values = [float(row["mean_completion_tokens"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    bars = axis.bar(labels, values, color=["#4c78a8", "#72b7b2", "#54a24b", "#e45756"])
    axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set_ylabel("Mean supervised completion tokens")
    axis.set_title("Paired training data length on identical problem support")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figure_dir / "paired_training_data_lengths")


def _plot_selection_grid(rows: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _pyplot()
    conditions = ["standard_original", "rewrite_80", "rewrite_65"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for axis, condition in zip(axes, conditions):
        subset = [row for row in rows if row["condition"] == condition]
        for learning_rate in sorted({float(row["learning_rate"]) for row in subset}):
            line = sorted(
                [row for row in subset if float(row["learning_rate"]) == learning_rate],
                key=lambda row: float(row["epoch"]),
            )
            axis.plot(
                [row["epoch"] for row in line],
                [row["accuracy"] for row in line],
                marker="o",
                label=f"LR {learning_rate:g}",
            )
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xlabel("Training epochs")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Development accuracy")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Shared recipe selection across paired SFT targets")
    figure.tight_layout()
    _save_figure(figure, figure_dir / "paired_rewrite_recipe_selection")


def _plot_training_diagnostics(
    manifests: Mapping[str, Mapping[str, Any]],
    max_grad_norm: float,
    figure_dir: Path,
) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    colors = {
        "standard_original": "#4c78a8",
        "rewrite_80": "#54a24b",
        "rewrite_65": "#72b7b2",
    }
    for manifest in sorted(
        manifests.values(),
        key=lambda row: (str(row["condition"]), float(row["learning_rate"])),
    ):
        history = manifest.get("training_log_history", [])
        condition = str(manifest["condition"])
        label = f"{CONDITION_LABELS[condition]}, LR {float(manifest['learning_rate']):g}"
        loss_rows = [row for row in history if row.get("epoch") is not None and row.get("loss") is not None]
        grad_rows = [row for row in history if row.get("epoch") is not None and row.get("grad_norm") is not None]
        axes[0].plot(
            [float(row["epoch"]) for row in loss_rows],
            [float(row["loss"]) for row in loss_rows],
            color=colors[condition],
            alpha=0.72,
            linewidth=1.3,
            label=label,
        )
        axes[1].plot(
            [float(row["epoch"]) for row in grad_rows],
            [float(row["grad_norm"]) for row in grad_rows],
            color=colors[condition],
            alpha=0.72,
            linewidth=1.3,
            label=label,
        )
    axes[0].set_xlabel("Training epochs")
    axes[0].set_ylabel("Completion-only training loss")
    axes[0].set_title("Loss trajectory")
    axes[1].axhline(max_grad_norm, color="#d62728", linestyle="--", linewidth=1.2, label="Clip threshold")
    axes[1].set_xlabel("Training epochs")
    axes[1].set_ylabel("Logged gradient norm")
    axes[1].set_title("Gradient stability")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    figure.suptitle("Paired-rewrite SFT optimization diagnostics")
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    _save_figure(figure, figure_dir / "paired_rewrite_training_diagnostics")


def _plot_budget_accuracy(rows: Sequence[Mapping[str, Any]], figure_dir: Path) -> None:
    plt = _pyplot()
    greedy = [row for row in rows if row["decoding_mode"] == "greedy"]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for model_id in sorted({str(row["model_id"]) for row in greedy}):
        line = sorted([row for row in greedy if row["model_id"] == model_id], key=lambda row: row["max_new_tokens"])
        axis.plot(
            [row["max_new_tokens"] for row in line],
            [row["accuracy"] for row in line],
            marker="o",
            label=_model_label(model_id),
        )
    axis.set_xlabel("Generation budget (max new tokens)")
    axis.set_ylabel("GSM8K accuracy")
    axis.set_title("Accuracy under explicit generation budgets")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    _save_figure(figure, figure_dir / "paired_rewrite_budget_accuracy")


def _plot_accuracy_length_frontier(
    rows: Sequence[Mapping[str, Any]],
    primary_cap: int,
    figure_dir: Path,
) -> None:
    plt = _pyplot()
    primary = [
        row
        for row in rows
        if row["decoding_mode"] == "greedy" and row["max_new_tokens"] == primary_cap
    ]
    figure, axis = plt.subplots(figsize=(8.0, 5.4))
    for row in primary:
        axis.scatter(row["mean_output_tokens"], row["accuracy"], s=65)
        axis.annotate(_model_label(str(row["model_id"])), (row["mean_output_tokens"], row["accuracy"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Mean generated tokens")
    axis.set_ylabel("GSM8K accuracy")
    axis.set_title("Paired-rewrite accuracy-length frontier")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, figure_dir / "paired_rewrite_accuracy_length_frontier")


def _model_label(model_id: str) -> str:
    if model_id.startswith("selected__"):
        return CONDITION_LABELS.get(model_id.removeprefix("selected__"), model_id)
    return CONDITION_LABELS.get(model_id, model_id.replace("_", " "))


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure: Any, path_without_suffix: Path) -> None:
    for suffix in (".png", ".pdf"):
        figure.savefig(path_without_suffix.with_suffix(suffix), dpi=220, bbox_inches="tight")
    figure.clf()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
