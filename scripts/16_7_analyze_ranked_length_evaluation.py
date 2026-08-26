#!/usr/bin/env python3
"""Analyze and report the locked ranked-length GSM8K evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.factorial_analysis import holm_adjust
from length_budget_distill.records import read_jsonl
from length_budget_distill.ranked_evaluation import (
    completed_evaluation_evidence,
    protocol_hash,
    validate_evaluation_protocol,
    validate_parent_training,
)
from length_budget_distill.ranked_evaluation_analysis import (
    paired_contrast,
    summarize_predictions,
)


MODEL_ORDER = ("base", "relative_short", "relative_medium", "relative_long")
MODEL_LABELS = {
    "base": "Base 1.5B",
    "relative_short": "Short-ranked SFT",
    "relative_medium": "Medium-ranked SFT",
    "relative_long": "Long-ranked SFT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_ranked_sampling_7b_eval_seed17_v1.json",
    )
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    validate_evaluation_protocol(config)
    validate_parent_training(config, PROJECT_ROOT)
    eval_hash = protocol_hash(config)
    evaluation = dict(config["evaluation"])
    analysis_config = dict(config["analysis"])
    manifest_path = _resolve(args.eval_manifest)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("config_hash") != eval_hash:
        raise ValueError(f"Evaluation manifest is incomplete or mismatched: {manifest_path}")
    if int(manifest.get("run_count", -1)) != int(evaluation["expected_run_count"]):
        raise ValueError("Evaluation manifest run count mismatch.")

    by_model: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metrics: List[Dict[str, Any]] = []
    for run in manifest.get("runs", []):
        model_id = str(run["model_id"])
        if model_id in by_model:
            raise ValueError(f"Duplicate evaluation model: {model_id}")
        if run.get("eval_status") not in {"complete", "skipped_complete"}:
            raise ValueError(f"Incomplete evaluation model: {model_id}")
        evidence = completed_evaluation_evidence(
            run["prediction_path"],
            run["summary_path"],
            expected_n=int(evaluation["limit"]),
            expected_start_index=int(evaluation["start_index"]),
            expected_split=str(evaluation["dataset_split"]),
        )
        if evidence is None:
            raise ValueError(f"Invalid evaluation artifacts for {model_id}")
        for field in ("prediction_sha256", "summary_sha256"):
            if run.get(field) != evidence[field]:
                raise ValueError(f"Evaluation hash mismatch for {model_id}: {field}")
        rows = list(read_jsonl(Path(str(run["prediction_path"]))))
        mapping = {str(row["problem_id"]): dict(row) for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate prediction identity for {model_id}")
        summary = summarize_predictions(rows)
        training_examples = run.get("training_examples")
        supervised_tokens = run.get("supervised_tokens")
        metrics.append(
            {
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "budget_name": run.get("budget_name"),
                "seed": run.get("seed"),
                "training_examples": training_examples,
                "supervised_tokens": supervised_tokens,
                "mean_supervised_tokens": (
                    float(supervised_tokens) / int(training_examples)
                    if training_examples and supervised_tokens is not None
                    else None
                ),
                **summary,
            }
        )
        predictions[model_id] = mapping
        by_model[model_id] = run
    if set(by_model) != set(MODEL_ORDER):
        raise ValueError(f"Evaluation model identities mismatch: {sorted(by_model)}")
    support = [set(predictions[model_id]) for model_id in MODEL_ORDER]
    if not all(ids == support[0] for ids in support[1:]):
        raise ValueError("Evaluation models have non-identical problem support.")

    contrast_rows = []
    for family_name, field_name in (
        ("primary", "primary_family"),
        ("secondary", "secondary_family"),
    ):
        family = []
        for contrast_name in analysis_config[field_name]:
            left, separator, right = str(contrast_name).partition("__vs__")
            if not separator or left not in predictions or right not in predictions:
                raise ValueError(f"Invalid registered contrast: {contrast_name}")
            row = {
                "family": family_name,
                "contrast": contrast_name,
                "left_model": left,
                "right_model": right,
                **paired_contrast(
                    predictions[left],
                    predictions[right],
                    bootstrap_samples=int(analysis_config["bootstrap_samples"]),
                    bootstrap_seed=int(canonical_sha256([eval_hash, contrast_name])[:8], 16),
                ),
            }
            family.append(row)
        adjusted = holm_adjust([float(row["mcnemar_p_value"]) for row in family])
        for row, adjusted_p in zip(family, adjusted):
            row["mcnemar_holm_p_value"] = adjusted_p
            row["holm_significant"] = adjusted_p < float(analysis_config["familywise_alpha"])
            contrast_rows.append(row)

    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics.sort(key=lambda row: MODEL_ORDER.index(str(row["model_id"])))
    analysis_path = output_dir / "ranked_length_analysis.json"
    metrics_path = output_dir / "run_metrics.csv"
    contrasts_path = output_dir / "paired_contrasts.csv"
    report_path = output_dir / "experiment_report.md"
    figure_prefix = figure_dir / "ranked_length_accuracy_and_output_length"
    _write_json(
        analysis_path,
        {
            "status": "complete",
            "experiment_name": config["experiment_name"],
            "protocol_variant": config["protocol_variant"],
            "evidence_level": analysis_config["evidence_level"],
            "scope": analysis_config["scope"],
            "config_path": str(config_path),
            "config_hash": eval_hash,
            "config_file_sha256": file_sha256(config_path),
            "eval_manifest": str(manifest_path),
            "eval_manifest_sha256": file_sha256(manifest_path),
            "run_count": len(metrics),
            "problem_count": len(support[0]),
            "metrics": metrics,
            "contrasts": contrast_rows,
            "limitations": [
                "Only training seed 17 was evaluated; training-seed variability is not estimated.",
                "The scope is limited to the locked GSM8K test[50:1319] cohort.",
                "Training sets have equal example counts but unequal supervised-token totals.",
            ],
        },
    )
    _write_csv(metrics_path, metrics)
    _write_csv(contrasts_path, contrast_rows)
    _plot_results(metrics, figure_prefix)
    _write_report(report_path, config, metrics, contrast_rows)
    artifacts = [
        analysis_path,
        metrics_path,
        contrasts_path,
        report_path,
        figure_prefix.with_suffix(".png"),
        figure_prefix.with_suffix(".pdf"),
    ]
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {
            "status": "complete",
            "config_hash": eval_hash,
            "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
            "analysis_library_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/ranked_evaluation_analysis.py"
            ),
            "artifacts": [
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
    )
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={eval_hash}\n"
        f"eval_manifest_sha256={file_sha256(manifest_path)}\n"
        f"artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n"
        f"run_count={len(metrics)}\nproblem_count={len(support[0])}\n",
        encoding="utf-8",
    )
    logging.info("ranked_length_analysis_complete report=%s", report_path)


def _plot_results(metrics: List[Mapping[str, Any]], output_prefix: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["model_label"]) for row in metrics]
    colors = ["#8c8c8c", "#4c78a8", "#f2a541", "#c44e52"]
    accuracies = [float(row["accuracy"]) for row in metrics]
    lower = [value - float(row["wilson_ci_low"]) for value, row in zip(accuracies, metrics)]
    upper = [float(row["wilson_ci_high"]) - value for value, row in zip(accuracies, metrics)]
    output_tokens = [float(row["mean_output_tokens"]) for row in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    bars = axes[0].bar(labels, accuracies, color=colors, yerr=[lower, upper], capsize=4)
    axes[0].bar_label(bars, labels=[f"{100 * value:.2f}%" for value in accuracies], padding=4)
    axes[0].set_ylabel("GSM8K accuracy")
    axes[0].set_ylim(0, max(float(row["wilson_ci_high"]) for row in metrics) + 0.08)
    axes[0].set_title("Locked GSM8K test[50:1319], n=1269")
    axes[0].grid(axis="y", alpha=0.25)
    length_bars = axes[1].bar(labels, output_tokens, color=colors)
    axes[1].bar_label(length_bars, labels=[f"{value:.1f}" for value in output_tokens], padding=4)
    axes[1].set_ylabel("Mean generated tokens")
    axes[1].set_title("Greedy output length, max 512 tokens")
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
    figure.suptitle("Rank-selected teacher supervision: student accuracy and response length")
    figure.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    config: Mapping[str, Any],
    metrics: List[Mapping[str, Any]],
    contrasts: List[Mapping[str, Any]],
) -> None:
    by_id = {str(row["model_id"]): row for row in metrics}
    best = max(metrics, key=lambda row: float(row["accuracy"]))
    lines = [
        "# Ranked-length SFT evaluation report",
        "",
        "## Protocol and evidence scope",
        "",
        "- Student: Qwen2.5-1.5B-Instruct with rank-4 LoRA.",
        "- Evaluation: greedy decoding on locked GSM8K `test[50:1319]` (`n=1269`), max 512 new tokens.",
        "- Training supervision: 881 equal-example problems per adapter, selected from 16 correct-candidate samples per problem where available.",
        "- Evidence level: revised formal single-seed evaluation (seed 17); training-seed variability is not estimated.",
        "",
        "## Accuracy and generated length",
        "",
        "| Model | Correct / n | Accuracy | 95% Wilson CI | Mean output tokens | Max-token hit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model_label']} | {row['correct']} / {row['n']} | "
            f"{100 * float(row['accuracy']):.2f}% | "
            f"[{100 * float(row['wilson_ci_low']):.2f}%, {100 * float(row['wilson_ci_high']):.2f}%] | "
            f"{float(row['mean_output_tokens']):.1f} | {100 * float(row['max_token_hit_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Positive differences favor the left-hand model. McNemar p-values are Holm-adjusted within the registered primary and secondary families.",
            "",
            "| Family | Contrast | Accuracy difference | Paired 95% bootstrap CI | Holm-adjusted McNemar p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['family']} | {MODEL_LABELS[row['left_model']]} vs {MODEL_LABELS[row['right_model']]} | "
            f"{100 * float(row['accuracy_difference']):+.2f} pp | "
            f"[{100 * float(row['accuracy_difference_ci_low']):+.2f}, "
            f"{100 * float(row['accuracy_difference_ci_high']):+.2f}] pp | "
            f"{float(row['mcnemar_holm_p_value']):.4g} |"
        )
    short = by_id["relative_short"]
    medium = by_id["relative_medium"]
    long = by_id["relative_long"]
    lines.extend(
        [
            "",
            "## Numerical takeaway",
            "",
            f"The highest observed accuracy is **{best['model_label']}** at {100 * float(best['accuracy']):.2f}%.",
            f"Across ranked supervision, mean generated length changes from {float(short['mean_output_tokens']):.1f} tokens (short) "
            f"to {float(medium['mean_output_tokens']):.1f} (medium) and {float(long['mean_output_tokens']):.1f} (long).",
            "Statistical interpretation must follow the registered paired contrasts above; point-estimate ordering alone is not confirmatory evidence.",
            "",
            "## Limitations",
            "",
            "- Only seed 17 was trained, so seed-to-seed training variability is unknown.",
            "- Results are limited to GSM8K and should not be generalized to MATH-500 or other OOD benchmarks.",
            "- Equal-example training does not equalize supervised tokens: short, medium, and long runs expose different token totals.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
