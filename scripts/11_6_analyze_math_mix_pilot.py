#!/usr/bin/env python3
"""Analyze and plot GSM-only versus MATH-mixed 7B pilot adapters."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.factorial_analysis import holm_adjust, paired_cluster_bootstrap
from length_budget_distill.records import read_jsonl


BUDGETS = ("short_128", "medium_256", "long_512")
MODES = ("equal_example", "equal_token")
DATASETS = ("gsm8k", "math500", "aime2025")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eval_dir = Path(args.eval_output_dir)
    model_manifests = [Path(path) for path in sorted(glob.glob(str(eval_dir / "model_manifests" / "*.json")))]
    if len(model_manifests) != 13:
        raise ValueError(f"Expected 13 completed model manifests, got {len(model_manifests)}")

    metric_rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, str], Dict[str, bool]] = {}
    model_metadata: Dict[str, Dict[str, Any]] = {}
    input_evidence = []
    for manifest_path in model_manifests:
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or len(manifest.get("artifacts", [])) != 3:
            raise ValueError(f"Incomplete model evaluation manifest: {manifest_path}")
        model_id = str(manifest["model_id"])
        metadata = dict(manifest["model_metadata"])
        model_metadata[model_id] = metadata
        input_evidence.append({"path": str(manifest_path), "sha256": file_sha256(manifest_path)})
        for artifact in manifest["artifacts"]:
            dataset_name = str(artifact["dataset_name"])
            prediction_path = Path(artifact["prediction_path"])
            summary_path = Path(artifact["summary_path"])
            if file_sha256(prediction_path) != artifact["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {prediction_path}")
            if file_sha256(summary_path) != artifact["summary_sha256"]:
                raise ValueError(f"Summary hash mismatch: {summary_path}")
            summary = _read_json(summary_path)
            rows = list(read_jsonl(prediction_path))
            expected_n = {"gsm8k": 200, "math500": 100, "aime2025": 30}[dataset_name]
            if len(rows) != expected_n or int(summary["n"]) != expected_n:
                raise ValueError(f"Evaluation cardinality mismatch: {model_id}/{dataset_name}")
            mapping = {str(row["problem_id"]): bool(row["is_correct"]) for row in rows}
            if len(mapping) != expected_n:
                raise ValueError(f"Duplicate prediction IDs: {model_id}/{dataset_name}")
            predictions[(model_id, dataset_name)] = mapping
            ci_low, ci_high = _wilson_interval(int(summary["correct"]), expected_n)
            metric_rows.append(
                {
                    "model_id": model_id,
                    "training_variant": metadata["training_variant"],
                    "mode": metadata["mode"],
                    "budget_name": metadata.get("budget_name"),
                    "seed": metadata.get("seed"),
                    "dataset_name": dataset_name,
                    "n": expected_n,
                    "correct": int(summary["correct"]),
                    "accuracy": float(summary["accuracy"]),
                    "wilson_ci_low": ci_low,
                    "wilson_ci_high": ci_high,
                    "mean_output_tokens": float(summary["mean_output_tokens"]),
                    "extraction_failures": int(summary["extraction_failures"]),
                    "max_token_hits": int(summary["max_token_hits"]),
                }
            )

    registry = _comparison_registry(model_metadata)
    comparison_rows = []
    for dataset_name in DATASETS:
        dataset_rows = []
        for mode in MODES:
            for budget_name in BUDGETS:
                reference_id = registry[("gsm_only", mode, budget_name)]
                mixed_id = registry[("math_mix", mode, budget_name)]
                reference = predictions[(reference_id, dataset_name)]
                mixed = predictions[(mixed_id, dataset_name)]
                if set(reference) != set(mixed):
                    raise ValueError(f"Paired support mismatch: {dataset_name}/{mode}/{budget_name}")
                effects = {
                    problem_id: float(mixed[problem_id]) - float(reference[problem_id])
                    for problem_id in sorted(reference)
                }
                bootstrap = paired_cluster_bootstrap(
                    effects,
                    samples=args.bootstrap_samples,
                    seed=int(canonical_sha256([dataset_name, mode, budget_name])[:8], 16),
                )
                mixed_wins = sum(mixed[key] and not reference[key] for key in reference)
                reference_wins = sum(reference[key] and not mixed[key] for key in reference)
                dataset_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "mode": mode,
                        "budget_name": budget_name,
                        "reference_model_id": reference_id,
                        "mixed_model_id": mixed_id,
                        "n": len(reference),
                        "reference_correct": sum(reference.values()),
                        "mixed_correct": sum(mixed.values()),
                        "accuracy_delta": bootstrap["estimate"],
                        "bootstrap_ci_low": bootstrap["ci_low"],
                        "bootstrap_ci_high": bootstrap["ci_high"],
                        "mixed_only_correct": mixed_wins,
                        "reference_only_correct": reference_wins,
                        "mcnemar_p_value": _exact_mcnemar_p_value(mixed_wins, reference_wins),
                    }
                )
        adjusted = holm_adjust([float(row["mcnemar_p_value"]) for row in dataset_rows])
        for row, adjusted_p in zip(dataset_rows, adjusted):
            row["holm_p_value"] = adjusted_p
        comparison_rows.extend(dataset_rows)

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "run_metrics.csv"
    comparisons_path = output_dir / "paired_comparisons.csv"
    report_path = output_dir / "experiment_report.md"
    analysis_path = output_dir / "math_mix_pilot_analysis.json"
    png_path = figure_dir / "gsm_math500_aime25_accuracy.png"
    pdf_path = figure_dir / "gsm_math500_aime25_accuracy.pdf"
    _write_csv(metrics_path, metric_rows)
    _write_csv(comparisons_path, comparison_rows)
    _write_accuracy_figure(metric_rows, png_path, pdf_path)
    _write_report(report_path, metric_rows, comparison_rows)
    _write_json(
        analysis_path,
        {
            "status": "complete",
            "evidence_level": "exploratory_single_seed_pilot",
            "scope": "Qwen2.5-7B teacher length pilot with mixed GSM8K and MATH-train supervision",
            "limitations": [
                "Single training seed 17 does not estimate training-seed variability.",
                "GSM8K official test was previously observed; GSM8K-200 is an adaptive diagnostic.",
                "MATH-500-100 and AIME-2025-30 are small pilot evaluations.",
            ],
            "metric_rows": metric_rows,
            "paired_comparisons": comparison_rows,
            "inputs": input_evidence,
        },
    )
    artifacts = [metrics_path, comparisons_path, report_path, analysis_path, png_path, pdf_path]
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {
            "status": "complete",
            "artifacts": [
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
    )
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        "evidence_level=exploratory_single_seed_pilot\n"
        f"model_count=13\ndataset_count=3\n"
        f"artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n",
        encoding="utf-8",
    )
    logging.info("math_mix_analysis_complete output=%s figure=%s", output_dir, png_path)


def _comparison_registry(metadata_by_model: Mapping[str, Mapping[str, Any]]) -> Dict[Tuple[str, str, str], str]:
    registry = {}
    for model_id, metadata in metadata_by_model.items():
        variant = str(metadata["training_variant"])
        if variant not in {"gsm_only", "math_mix"}:
            continue
        key = (variant, str(metadata["mode"]), str(metadata["budget_name"]))
        if key in registry:
            raise ValueError(f"Duplicate comparison model identity: {key}")
        registry[key] = model_id
    expected = {
        (variant, mode, budget)
        for variant in ("gsm_only", "math_mix")
        for mode in MODES
        for budget in BUDGETS
    }
    if set(registry) != expected:
        raise ValueError(f"Incomplete comparison registry: observed={sorted(registry)}")
    return registry


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _exact_mcnemar_p_value(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _write_accuracy_figure(rows: Sequence[Mapping[str, Any]], png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {"short_128": "Short (128)", "medium_256": "Medium (256)", "long_512": "Long (512)"}
    dataset_titles = {"gsm8k": "GSM8K test subset (n=200)", "math500": "MATH-500 subset (n=100)", "aime2025": "AIME 2025 (n=30)"}
    colors = {"gsm_only": "#4C78A8", "math_mix": "#E45756"}
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharex=True)
    for row_index, mode in enumerate(MODES):
        for column_index, dataset_name in enumerate(DATASETS):
            axis = axes[row_index][column_index]
            base_rows = [
                row for row in rows
                if row["training_variant"] == "base" and row["dataset_name"] == dataset_name
            ]
            if len(base_rows) != 1:
                raise ValueError(f"Missing base metric for {dataset_name}")
            axis.axhline(
                100.0 * float(base_rows[0]["accuracy"]),
                color="#777777",
                linestyle="--",
                linewidth=1.3,
                label="Base" if row_index == 0 and column_index == 0 else None,
            )
            for variant in ("gsm_only", "math_mix"):
                values = []
                lower = []
                upper = []
                for budget in BUDGETS:
                    matches = [
                        row for row in rows
                        if row["training_variant"] == variant
                        and row["mode"] == mode
                        and row["budget_name"] == budget
                        and row["dataset_name"] == dataset_name
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"Missing metric for {variant}/{mode}/{budget}/{dataset_name}")
                    match = matches[0]
                    accuracy = 100.0 * float(match["accuracy"])
                    values.append(accuracy)
                    lower.append(accuracy - 100.0 * float(match["wilson_ci_low"]))
                    upper.append(100.0 * float(match["wilson_ci_high"]) - accuracy)
                axis.errorbar(
                    range(3),
                    values,
                    yerr=[lower, upper],
                    marker="o",
                    capsize=3,
                    linewidth=1.8,
                    color=colors[variant],
                    label=("GSM8K-only" if variant == "gsm_only" else "GSM8K + MATH")
                    if row_index == 0 and column_index == 0
                    else None,
                )
            axis.set_xticks(range(3), [labels[budget] for budget in BUDGETS])
            axis.grid(axis="y", alpha=0.25)
            if row_index == 0:
                axis.set_title(dataset_titles[dataset_name])
            if column_index == 0:
                axis.set_ylabel(("Equal-example\n" if mode == "equal_example" else "Equal-token\n") + "Accuracy (%)")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure.suptitle("Qwen2.5-7B length pilot: GSM8K-only versus MATH-mixed SFT", y=0.985)
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def _write_report(path: Path, metrics: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# MATH-mixed SFT pilot report",
        "",
        "Evidence level: exploratory single-seed pilot. This report does not replace the frozen seed-17 GSM8K formal result.",
        "",
        "## Paired GSM8K-only versus MATH-mixed comparisons",
        "",
        "| Dataset | Mode | Length | GSM-only correct | Mixed correct | Delta (pp) | 95% paired bootstrap CI (pp) | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['dataset_name']} | {row['mode']} | {row['budget_name']} | "
            f"{row['reference_correct']}/{row['n']} | {row['mixed_correct']}/{row['n']} | "
            f"{100 * row['accuracy_delta']:.2f} | "
            f"[{100 * row['bootstrap_ci_low']:.2f}, {100 * row['bootstrap_ci_high']:.2f}] | "
            f"{row['holm_p_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- AIME 2025 contains only 30 items and is especially susceptible to floor effects.",
            "- The 200 GSM8K questions come from the previously observed official test and are adaptive diagnostics.",
            "- All adapters use only training seed 17, so training-seed variability is not estimated.",
            "- Expansion to all teacher capacities requires a separate approval after reviewing this pilot.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
