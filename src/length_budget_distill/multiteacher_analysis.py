"""Analysis and figures for matched SFT versus multi-teacher logit-KD pilots."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .factorial import canonical_sha256, file_sha256
from .factorial_analysis import (
    exact_mcnemar_p_value,
    holm_adjust,
    paired_cluster_bootstrap,
    wilson_interval,
)
from .records import read_jsonl


TEACHERS = ("qwen2p5_1p5b", "qwen2p5_3b", "qwen2p5_7b", "qwen2p5_14b")
BUDGETS = ("short_128", "medium_256", "long_512")
DATASETS = ("gsm8k", "math500", "aime2025")
METHODS = ("sft", "logit_kd")


def analyze_multiteacher_kd(
    frozen: Mapping[str, Any],
    *,
    eval_dir: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    bootstrap_samples: int = 10_000,
) -> Dict[str, Any]:
    """Validate evaluation artifacts, compute paired effects, and render two figures."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    resolved_eval = Path(eval_dir)
    manifest_paths = [
        Path(path)
        for path in sorted(glob.glob(str(resolved_eval / "model_manifests" / "*.json")))
    ]
    if len(manifest_paths) != 25:
        raise ValueError(f"Expected 25 completed model manifests, got {len(manifest_paths)}")

    eval_suite_path = Path(str(frozen["evaluation_suite_manifest"]["path"]))
    eval_suite = _read_json(eval_suite_path)
    expected_n = {
        str(dataset["dataset_name"]): int(dataset["n"])
        for dataset in eval_suite["datasets"]
    }
    if set(expected_n) != set(DATASETS):
        raise ValueError(f"Unexpected evaluation suite datasets: {sorted(expected_n)}")

    metric_rows: List[Dict[str, Any]] = []
    predictions: Dict[Tuple[str, str], Dict[str, bool]] = {}
    metadata_by_model: Dict[str, Dict[str, Any]] = {}
    input_evidence = []
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or len(manifest.get("artifacts", [])) != 3:
            raise ValueError(f"Incomplete model evaluation manifest: {manifest_path}")
        model_id = str(manifest["model_id"])
        metadata = dict(manifest["model_metadata"])
        metadata_by_model[model_id] = metadata
        input_evidence.append(_file_evidence(manifest_path))
        for artifact in manifest["artifacts"]:
            dataset_name = str(artifact["dataset_name"])
            prediction_path = Path(str(artifact["prediction_path"]))
            summary_path = Path(str(artifact["summary_path"]))
            if file_sha256(prediction_path) != artifact["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {prediction_path}")
            if file_sha256(summary_path) != artifact["summary_sha256"]:
                raise ValueError(f"Summary hash mismatch: {summary_path}")
            rows = list(read_jsonl(prediction_path))
            summary = _read_json(summary_path)
            n = expected_n[dataset_name]
            if len(rows) != n or int(summary["n"]) != n:
                raise ValueError(f"Evaluation cardinality mismatch: {model_id}/{dataset_name}")
            mapping = {str(row["problem_id"]): bool(row["is_correct"]) for row in rows}
            if len(mapping) != n:
                raise ValueError(f"Duplicate prediction IDs: {model_id}/{dataset_name}")
            predictions[(model_id, dataset_name)] = mapping
            correct = int(summary["correct"])
            ci_low, ci_high = wilson_interval(correct, n)
            metric_rows.append(
                {
                    "model_id": model_id,
                    "method": metadata["method"],
                    "generator_name": metadata.get("generator_name"),
                    "generator_size_b": metadata.get("generator_size_b"),
                    "self_distillation_control": metadata.get(
                        "self_distillation_control"
                    ),
                    "budget_name": metadata.get("budget_name"),
                    "seed": metadata.get("seed"),
                    "dataset_name": dataset_name,
                    "n": n,
                    "correct": correct,
                    "accuracy": float(summary["accuracy"]),
                    "wilson_ci_low": ci_low,
                    "wilson_ci_high": ci_high,
                    "mean_output_tokens": float(summary["mean_output_tokens"]),
                    "extraction_failures": int(summary["extraction_failures"]),
                    "max_token_hits": int(summary["max_token_hits"]),
                }
            )

    registry, base_id = _comparison_registry(metadata_by_model)
    comparison_rows: List[Dict[str, Any]] = []
    for dataset_name in DATASETS:
        family = []
        for generator_name in TEACHERS:
            for budget_name in BUDGETS:
                sft_id = registry[("sft", generator_name, budget_name)]
                kd_id = registry[("logit_kd", generator_name, budget_name)]
                sft = predictions[(sft_id, dataset_name)]
                kd = predictions[(kd_id, dataset_name)]
                if set(sft) != set(kd):
                    raise ValueError(
                        f"Paired support mismatch: {dataset_name}/{generator_name}/{budget_name}"
                    )
                effects = {
                    problem_id: float(kd[problem_id]) - float(sft[problem_id])
                    for problem_id in sorted(sft)
                }
                bootstrap = paired_cluster_bootstrap(
                    effects,
                    samples=bootstrap_samples,
                    seed=int(
                        canonical_sha256(
                            [dataset_name, generator_name, budget_name, "kd_minus_sft"]
                        )[:8],
                        16,
                    ),
                )
                kd_only = sum(kd[key] and not sft[key] for key in sft)
                sft_only = sum(sft[key] and not kd[key] for key in sft)
                family.append(
                    {
                        "dataset_name": dataset_name,
                        "generator_name": generator_name,
                        "budget_name": budget_name,
                        "sft_model_id": sft_id,
                        "kd_model_id": kd_id,
                        "n": len(sft),
                        "sft_correct": sum(sft.values()),
                        "kd_correct": sum(kd.values()),
                        "accuracy_delta": bootstrap["estimate"],
                        "bootstrap_ci_low": bootstrap["ci_low"],
                        "bootstrap_ci_high": bootstrap["ci_high"],
                        "kd_only_correct": kd_only,
                        "sft_only_correct": sft_only,
                        "mcnemar_p_value": exact_mcnemar_p_value(kd_only, sft_only),
                    }
                )
        adjusted = holm_adjust([float(row["mcnemar_p_value"]) for row in family])
        for row, p_value in zip(family, adjusted):
            row["holm_p_value"] = p_value
        comparison_rows.extend(family)

    resolved_output = Path(output_dir)
    resolved_figures = Path(figure_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_figures.mkdir(parents=True, exist_ok=True)
    metrics_path = resolved_output / "run_metrics.csv"
    comparisons_path = resolved_output / "paired_kd_vs_sft.csv"
    report_path = resolved_output / "experiment_report.md"
    analysis_path = resolved_output / "multiteacher_kd_analysis.json"
    accuracy_png = resolved_figures / "multibench_accuracy_by_teacher_length.png"
    accuracy_pdf = resolved_figures / "multibench_accuracy_by_teacher_length.pdf"
    delta_png = resolved_figures / "kd_minus_sft_accuracy_heatmap.png"
    delta_pdf = resolved_figures / "kd_minus_sft_accuracy_heatmap.pdf"
    _write_csv(metrics_path, metric_rows)
    _write_csv(comparisons_path, comparison_rows)
    _plot_accuracy_grid(metric_rows, accuracy_png, accuracy_pdf)
    _plot_delta_heatmap(comparison_rows, delta_png, delta_pdf)
    _write_report(report_path, comparison_rows, frozen)
    payload = {
        "status": "complete",
        "evidence_level": str(frozen["evidence_level"]),
        "model_count": 25,
        "dataset_count": 3,
        "paired_comparison_count": len(comparison_rows),
        "base_model_id": base_id,
        "limitations": [
            "All adapters use training seed 17; training-seed variability is not estimated.",
            "Teacher common support is matched across lengths within each teacher, not across teachers.",
            "GSM8K-200 is adaptive because the official test was previously observed.",
            "MATH-500-100 and AIME-2025-30 remain small exploratory cohorts.",
            "The inherited alpha and temperature were selected on 7B GSM8K equal-token data.",
        ],
        "inputs": input_evidence,
        "metric_rows": metric_rows,
        "paired_comparisons": comparison_rows,
    }
    _write_json(analysis_path, payload)
    artifacts = [
        metrics_path,
        comparisons_path,
        report_path,
        analysis_path,
        accuracy_png,
        accuracy_pdf,
        delta_png,
        delta_pdf,
    ]
    artifact_manifest_path = resolved_output / "analysis_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {"status": "complete", "artifacts": [_file_evidence(path) for path in artifacts]},
    )
    (resolved_output / "ANALYSIS_COMPLETE").write_text(
        f"evidence_level={frozen['evidence_level']}\n"
        "model_count=25\ndataset_count=3\npaired_comparisons=36\n"
        f"artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n",
        encoding="utf-8",
    )
    return payload


def _comparison_registry(
    metadata_by_model: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[Tuple[str, str, str], str], str]:
    registry = {}
    base_ids = []
    for model_id, metadata in metadata_by_model.items():
        method = str(metadata["method"])
        if method == "base":
            base_ids.append(model_id)
            continue
        key = (method, str(metadata["generator_name"]), str(metadata["budget_name"]))
        if key in registry:
            raise ValueError(f"Duplicate comparison model identity: {key}")
        registry[key] = model_id
    expected = {
        (method, generator, budget)
        for method in METHODS
        for generator in TEACHERS
        for budget in BUDGETS
    }
    if set(registry) != expected or len(base_ids) != 1:
        raise ValueError("Evaluation metadata does not cover base plus the 24 planned adapters")
    return registry, base_ids[0]


def _plot_accuracy_grid(
    rows: Sequence[Mapping[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    lookup = {
        (
            str(row["method"]),
            str(row["generator_name"]),
            str(row["budget_name"]),
            str(row["dataset_name"]),
        ): row
        for row in rows
        if row["method"] in METHODS
    }
    base = {
        str(row["dataset_name"]): float(row["accuracy"])
        for row in rows
        if row["method"] == "base"
    }
    x_values = [128, 256, 512]
    fig, axes = plt.subplots(4, 3, figsize=(12.4, 12.8), sharex=True)
    method_style = {
        "sft": {"label": "Hard-target SFT", "color": "#3b6fb6", "marker": "o"},
        "logit_kd": {"label": "Logit KD", "color": "#d97924", "marker": "s"},
    }
    teacher_labels = {
        "qwen2p5_1p5b": "Teacher 1.5B (self-distillation control)",
        "qwen2p5_3b": "Teacher 3B",
        "qwen2p5_7b": "Teacher 7B",
        "qwen2p5_14b": "Teacher 14B",
    }
    dataset_labels = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime2025": "AIME 2025"}
    for row_index, teacher in enumerate(TEACHERS):
        for column_index, dataset in enumerate(DATASETS):
            axis = axes[row_index][column_index]
            for method in METHODS:
                values = [
                    float(lookup[(method, teacher, budget, dataset)]["accuracy"])
                    for budget in BUDGETS
                ]
                lower = [
                    values[index]
                    - float(lookup[(method, teacher, budget, dataset)]["wilson_ci_low"])
                    for index, budget in enumerate(BUDGETS)
                ]
                upper = [
                    float(lookup[(method, teacher, budget, dataset)]["wilson_ci_high"])
                    - values[index]
                    for index, budget in enumerate(BUDGETS)
                ]
                style = method_style[method]
                axis.errorbar(
                    x_values,
                    values,
                    yerr=[lower, upper],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    capsize=2.5,
                    label=style["label"],
                )
            axis.axhline(base[dataset], color="#666666", linestyle="--", linewidth=1.1)
            axis.grid(alpha=0.2)
            axis.set_xticks(x_values)
            if row_index == 0:
                axis.set_title(dataset_labels[dataset])
            if column_index == 0:
                axis.set_ylabel(f"{teacher_labels[teacher]}\nAccuracy")
            if row_index == len(TEACHERS) - 1:
                axis.set_xlabel("Teacher trajectory budget (tokens)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Matched SFT versus logit KD across teachers, lengths, and benchmarks", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_heatmap(
    rows: Sequence[Mapping[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {
        (str(row["dataset_name"]), str(row["generator_name"]), str(row["budget_name"])): float(
            row["accuracy_delta"]
        )
        for row in rows
    }
    matrices = [
        np.array(
            [[lookup[(dataset, teacher, budget)] for budget in BUDGETS] for teacher in TEACHERS]
        )
        for dataset in DATASETS
    ]
    limit = max(0.01, max(abs(float(value)) for matrix in matrices for value in matrix.flat))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), sharey=True)
    image = None
    for axis, dataset, matrix in zip(axes, DATASETS, matrices):
        image = axis.imshow(matrix * 100.0, cmap="coolwarm", vmin=-limit * 100, vmax=limit * 100)
        axis.set_title({"gsm8k": "GSM8K", "math500": "MATH-500", "aime2025": "AIME 2025"}[dataset])
        axis.set_xticks(range(3), ["128", "256", "512"])
        axis.set_xlabel("Trajectory budget")
        for row_index in range(4):
            for column_index in range(3):
                value = matrix[row_index, column_index] * 100.0
                axis.text(column_index, row_index, f"{value:+.1f}", ha="center", va="center", fontsize=9)
    axes[0].set_yticks(
        range(4),
        ["1.5B control", "3B", "7B", "14B"],
    )
    if image is not None:
        fig.colorbar(image, ax=axes, label="Logit KD minus SFT accuracy (percentage points)", shrink=0.88)
    fig.suptitle("Paired accuracy effect of logit KD", y=0.99)
    fig.subplots_adjust(left=0.09, right=0.92, bottom=0.14, top=0.86, wspace=0.18)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> None:
    lines = [
        "# Multi-benchmark, multi-teacher logit-KD pilot report",
        "",
        "Evidence level: exploratory single-seed pilot. The experiment does not replace the frozen GSM8K formal result.",
        "",
        f"Inherited KD hyperparameters: alpha={frozen['kd']['alpha']}, temperature={frozen['kd']['temperature']}.",
        "",
        "## Paired logit-KD versus matched SFT comparisons",
        "",
        "| Dataset | Teacher | Length | SFT correct | KD correct | Delta (pp) | 95% paired bootstrap CI (pp) | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_name']} | {row['generator_name']} | {row['budget_name']} | "
            f"{row['sft_correct']}/{row['n']} | {row['kd_correct']}/{row['n']} | "
            f"{100 * row['accuracy_delta']:.2f} | "
            f"[{100 * row['bootstrap_ci_low']:.2f}, {100 * row['bootstrap_ci_high']:.2f}] | "
            f"{row['holm_p_value']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- All adapters use seed 17, so the experiment does not estimate training-seed variability.",
            "- Common problem support is matched across lengths within a teacher but differs across teachers.",
            "- GSM8K-200 is adaptive; MATH-500-100 and AIME-2025-30 are exploratory pilot cohorts.",
            "- Alpha and temperature were inherited from the completed 7B GSM8K equal-token KD experiment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_evidence(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
