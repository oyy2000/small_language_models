#!/usr/bin/env python3
"""Analyze formal SFT-versus-logit-KD accuracy, length, and matched logits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial_analysis import holm_adjust, paired_cluster_bootstrap
from length_budget_distill.logit_kd import (
    file_sha256,
    kd_run_name,
    load_protocol,
    protocol_hash,
    read_json,
    read_jsonl,
    resolve_project_path,
    supervision_mode,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    return parser.parse_args()


def _prediction_metrics(path: Path, budget_tokens: int | None) -> Dict[str, Any]:
    rows = read_jsonl(path)
    lengths = [int(row["output_token_count"]) for row in rows]
    ordered_lengths = sorted(lengths)
    payload = {
        "n": len(rows),
        "accuracy": sum(bool(row["is_correct"]) for row in rows) / len(rows),
        "mean_output_tokens": statistics.mean(lengths),
        "median_output_tokens": statistics.median(lengths),
        "p95_output_tokens": ordered_lengths[int(0.95 * (len(ordered_lengths) - 1))],
        "max_output_tokens": max(lengths),
    }
    if budget_tokens is not None:
        payload["budget_compliance"] = sum(length <= budget_tokens for length in lengths) / len(lengths)
    return payload


def _prediction_map(path: Path) -> Dict[str, bool]:
    rows = read_jsonl(path)
    mapping = {str(row["problem_id"]): bool(row["is_correct"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"Duplicate prediction problem IDs: {path}")
    return mapping


def _mcnemar_exact(kd: Mapping[str, bool], sft: Mapping[str, bool]) -> Dict[str, Any]:
    if set(kd) != set(sft):
        raise ValueError("KD and SFT predictions do not have identical problem support.")
    kd_only = sum(kd[key] and not sft[key] for key in kd)
    sft_only = sum(sft[key] and not kd[key] for key in kd)
    discordant = kd_only + sft_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(kd_only, sft_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "kd_correct_sft_wrong": kd_only,
        "sft_correct_kd_wrong": sft_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def _paired_accuracy_bootstrap(kd: Mapping[str, bool], sft: Mapping[str, bool], samples: int) -> Dict[str, float]:
    effects = {key: float(kd[key]) - float(sft[key]) for key in kd}
    return paired_cluster_bootstrap(effects, samples=samples, seed=20260822)


def _logit_metrics(snapshot_dir: Path, num_shards: int, method: str, expected_records: int) -> Dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    token_count = 0
    weighted: Dict[str, float] = {}
    record_indices = set()
    expected_fields = ["entropy", "invalid_vocab_mass", "target_rank"]
    if method != "teacher":
        expected_fields.extend(["teacher_to_student_kl", "jensen_shannon", "topk_overlap_count"])
    shard_evidence = []
    for shard_index in range(num_shards):
        stem = f"shard_{shard_index:02d}_of_{num_shards:02d}"
        marker_path = snapshot_dir / f"{stem}.complete.json"
        marker = read_json(marker_path)
        tensor_path = Path(marker["tensor_path"])
        metadata_path = Path(marker["metadata_path"])
        if marker.get("tensor_sha256") != file_sha256(tensor_path):
            raise ValueError(f"Logit tensor hash mismatch: {tensor_path}")
        if marker.get("metadata_sha256") != file_sha256(metadata_path):
            raise ValueError(f"Logit metadata hash mismatch: {metadata_path}")
        metadata = read_json(metadata_path)
        tensors = load_file(tensor_path, device="cpu")
        current_tokens = int(marker["completion_tokens"])
        if tensors["target_token_ids"].shape[0] != current_tokens:
            raise ValueError(f"Logit token count mismatch: {tensor_path}")
        for field in expected_fields:
            if field not in tensors:
                raise ValueError(f"Logit tensor is missing {field}: {tensor_path}")
            values = tensors[field].to(torch.float64)
            weighted[field] = weighted.get(field, 0.0) + float(values.sum())
        for record in metadata["record_metadata"]:
            source_index = int(record["source_index"])
            if source_index in record_indices:
                raise ValueError(f"Duplicate logit source index: {source_index}")
            record_indices.add(source_index)
        token_count += current_tokens
        shard_evidence.append(
            {
                "marker_path": str(marker_path),
                "marker_sha256": file_sha256(marker_path),
                "tensor_path": str(tensor_path),
                "tensor_sha256": file_sha256(tensor_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": file_sha256(metadata_path),
            }
        )
    if record_indices != set(range(expected_records)):
        raise ValueError(f"Logit snapshot record coverage mismatch: {snapshot_dir}")
    metrics = {f"mean_{field}": total / token_count for field, total in weighted.items()}
    metrics.update({"completion_tokens": token_count, "records": len(record_indices), "shards": shard_evidence})
    return metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: Any, png_path: Path) -> List[Path]:
    pdf_path = png_path.with_suffix(".pdf")
    figure.tight_layout()
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    return [png_path, pdf_path]


def _figures(
    output_dir: Path,
    run_rows: List[Dict[str, Any]],
    logit_rows: List[Dict[str, Any]],
    kd_label: str,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = ["short_128", "medium_256", "long_512"]
    labels = ["Short (128)", "Medium (256)", "Long (512)"]
    budget_methods = ["SFT"]
    if any(row["method"] == "Same-prompt KD" for row in run_rows):
        budget_methods.append("Same-prompt KD")
    budget_methods.append(kd_label)
    colors = {
        "SFT": "#4C78A8",
        "Same-prompt KD": "#B9A0D3",
        kd_label: "#F58518",
    }
    artifacts: List[Path] = []
    accuracy_path = output_dir / "accuracy_by_budget.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    x = list(range(len(budgets)))
    width = 0.8 / len(budget_methods)
    for method_index, method in enumerate(budget_methods):
        offset = (method_index - (len(budget_methods) - 1) / 2) * width
        values = [next(row["accuracy"] for row in run_rows if row["method"] == method and row["budget_name"] == budget) for budget in budgets]
        axis.bar([value + offset for value in x], values, width=width, label=method, color=colors[method])
    base_accuracy = next(row["accuracy"] for row in run_rows if row["method"] == "Base")
    axis.axhline(base_accuracy, color="#777777", linestyle="--", label="Base")
    axis.set_xticks(x, labels)
    axis.set_ylabel("GSM8K accuracy")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    artifacts.extend(_save_figure(figure, accuracy_path))
    plt.close(figure)

    compliance_path = output_dir / "budget_compliance_by_budget.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for method_index, method in enumerate(budget_methods):
        offset = (method_index - (len(budget_methods) - 1) / 2) * width
        values = [
            next(
                row["budget_compliance"]
                for row in run_rows
                if row["method"] == method and row["budget_name"] == budget
            )
            for budget in budgets
        ]
        axis.bar(
            [value + offset for value in x],
            values,
            width=width,
            label=method,
            color=colors[method],
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Budget compliance")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    artifacts.extend(_save_figure(figure, compliance_path))
    plt.close(figure)

    pareto_path = output_dir / "accuracy_length_pareto.png"
    figure, axis = plt.subplots(figsize=(6.5, 4.8))
    for method in ("Base", *budget_methods):
        selected = [row for row in run_rows if row["method"] == method]
        axis.scatter(
            [row["mean_output_tokens"] for row in selected],
            [row["accuracy"] for row in selected],
            label=method,
            s=55,
        )
        for row in selected:
            if row["budget_name"]:
                axis.annotate(row["budget_name"].split("_")[0], (row["mean_output_tokens"], row["accuracy"]), fontsize=8)
    axis.set_xlabel("Mean generated tokens")
    axis.set_ylabel("GSM8K accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    artifacts.extend(_save_figure(figure, pareto_path))
    plt.close(figure)

    kl_path = output_dir / "matched_teacher_student_kl.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    methods = ["Base", "SFT", kd_label]
    width = 0.24
    for method_index, method in enumerate(methods):
        values = [next(row["mean_teacher_to_student_kl"] for row in logit_rows if row["method"] == method and row["budget_name"] == budget) for budget in budgets]
        axis.bar(
            [value + (method_index - 1) * width for value in x],
            values,
            width=width,
            label=method,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Exact teacher→student KL (nats/token)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    artifacts.extend(_save_figure(figure, kl_path))
    plt.close(figure)
    return artifacts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load_protocol(args.config)
    context_mode = str(protocol["kd"].get("context_mode", "same_prompt_teacher_forced"))
    kd_label = "Teacher-prompt KD" if context_mode == "dual_prompt_teacher_forced" else "Logit KD"
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    preflight = read_json(result_root / "preflight" / "parent_evidence.json")
    selection = read_json(result_root / "validation" / "selection.json")
    alpha = float(selection["selected_alpha"])
    temperature = float(selection["selected_temperature"])
    output_dir = result_root / "formal" / "analysis"
    if (output_dir / "ANALYSIS_COMPLETE").exists():
        raise FileExistsError(f"Formal analysis is already complete: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_prediction = Path(preflight["base_evaluation"]["prediction_path"])
    run_rows = [
        {
            "method": "Base",
            "budget_name": None,
            "budget_tokens": None,
            "prediction_path": str(base_prediction),
            **_prediction_metrics(base_prediction, None),
        }
    ]
    contrasts = []
    mcnemar_p_values = []
    for budget_name, budget in protocol["budgets"].items():
        budget_tokens = int(budget["max_solution_tokens"])
        sft_prediction = Path(preflight["budgets"][budget_name]["baseline_evaluation"]["prediction_path"])
        kd_eval_name = f"kd__{kd_run_name(budget_name, alpha, temperature)}"
        kd_marker_path = result_root / "formal" / "eval" / "markers" / f"{kd_eval_name}.json"
        kd_marker = read_json(kd_marker_path)
        kd_prediction = Path(kd_marker["prediction_path"])
        if kd_marker.get("prediction_sha256") != file_sha256(kd_prediction):
            raise ValueError(f"Formal KD prediction hash mismatch: {budget_name}")
        sft_metrics = _prediction_metrics(sft_prediction, budget_tokens)
        kd_metrics = _prediction_metrics(kd_prediction, budget_tokens)
        run_rows.extend(
            [
                {
                    "method": "SFT",
                    "budget_name": budget_name,
                    "budget_tokens": budget_tokens,
                    "prediction_path": str(sft_prediction),
                    **sft_metrics,
                },
                {
                    "method": kd_label,
                    "budget_name": budget_name,
                    "budget_tokens": budget_tokens,
                    "prediction_path": str(kd_prediction),
                    **kd_metrics,
                },
            ]
        )
        sft_map = _prediction_map(sft_prediction)
        kd_map = _prediction_map(kd_prediction)
        paired = _paired_accuracy_bootstrap(kd_map, sft_map, int(protocol["formal"]["bootstrap_samples"]))
        mcnemar = _mcnemar_exact(kd_map, sft_map)
        mcnemar_p_values.append(mcnemar["p_value"])
        contrasts.append(
            {
                "budget_name": budget_name,
                "accuracy_delta": kd_metrics["accuracy"] - sft_metrics["accuracy"],
                "budget_compliance_delta": kd_metrics["budget_compliance"] - sft_metrics["budget_compliance"],
                "mean_output_token_delta": kd_metrics["mean_output_tokens"] - sft_metrics["mean_output_tokens"],
                **{f"bootstrap_{key}": value for key, value in paired.items()},
                **{f"mcnemar_{key}": value for key, value in mcnemar.items()},
            }
        )
    adjusted = holm_adjust(mcnemar_p_values)
    for row, adjusted_p in zip(contrasts, adjusted):
        row["mcnemar_holm_p_value"] = adjusted_p

    prompt_context_contrasts: List[Dict[str, Any]] = []
    comparison_evidence = preflight.get("comparison")
    if comparison_evidence:
        comparison_analysis_path = Path(comparison_evidence["analysis"])
        if comparison_evidence.get("analysis_sha256") != file_sha256(comparison_analysis_path):
            raise ValueError("Same-prompt comparison analysis hash mismatch.")
        comparison_analysis = read_json(comparison_analysis_path)
        context_p_values = []
        for budget_name, budget in protocol["budgets"].items():
            budget_tokens = int(budget["max_solution_tokens"])
            old_row = next(
                row
                for row in comparison_analysis["run_metrics"]
                if row["method"] == "Logit KD" and row["budget_name"] == budget_name
            )
            old_prediction = Path(old_row["prediction_path"])
            old_metrics = _prediction_metrics(old_prediction, budget_tokens)
            run_rows.append(
                {
                    "method": "Same-prompt KD",
                    "budget_name": budget_name,
                    "budget_tokens": budget_tokens,
                    "prediction_path": str(old_prediction),
                    **old_metrics,
                }
            )
            new_row = next(
                row
                for row in run_rows
                if row["method"] == kd_label and row["budget_name"] == budget_name
            )
            new_prediction = Path(new_row["prediction_path"])
            old_map = _prediction_map(old_prediction)
            new_map = _prediction_map(new_prediction)
            paired = _paired_accuracy_bootstrap(
                new_map,
                old_map,
                int(protocol["formal"]["bootstrap_samples"]),
            )
            mcnemar = _mcnemar_exact(new_map, old_map)
            context_p_values.append(mcnemar["p_value"])
            prompt_context_contrasts.append(
                {
                    "budget_name": budget_name,
                    "accuracy_delta_teacher_prompt_vs_same_prompt": (
                        new_row["accuracy"] - old_metrics["accuracy"]
                    ),
                    "budget_compliance_delta_teacher_prompt_vs_same_prompt": (
                        new_row["budget_compliance"] - old_metrics["budget_compliance"]
                    ),
                    "mean_output_token_delta_teacher_prompt_vs_same_prompt": (
                        new_row["mean_output_tokens"] - old_metrics["mean_output_tokens"]
                    ),
                    **{f"bootstrap_{key}": value for key, value in paired.items()},
                    **{f"mcnemar_{key}": value for key, value in mcnemar.items()},
                }
            )
        for row, adjusted_p in zip(prompt_context_contrasts, holm_adjust(context_p_values)):
            row["mcnemar_holm_p_value"] = adjusted_p

    num_shards = int(protocol["outputs"]["logit_shards_per_snapshot"])
    logit_rows = []
    logit_evidence: Dict[str, Any] = {}
    for budget_name in protocol["budgets"]:
        expected_records = int(protocol["budgets"][budget_name]["expected_records"])
        logit_evidence[budget_name] = {}
        teacher_metrics = _logit_metrics(
            result_root / "formal" / "logits" / budget_name / "teacher",
            num_shards,
            "teacher",
            expected_records,
        )
        logit_evidence[budget_name]["teacher"] = teacher_metrics
        for method_dir, method_label in (("base", "Base"), ("sft", "SFT"), ("kd", kd_label)):
            metrics = _logit_metrics(
                result_root / "formal" / "logits" / budget_name / method_dir,
                num_shards,
                method_dir,
                expected_records,
            )
            logit_evidence[budget_name][method_dir] = metrics
            logit_rows.append(
                {
                    "budget_name": budget_name,
                    "method": method_label,
                    **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                }
            )
    max_compliance_drop = float(protocol["validation"].get("max_compliance_drop", 0.0))
    improved_all = all(
        row["accuracy_delta"] > 0
        and row["budget_compliance_delta"] >= -max_compliance_drop
        for row in contrasts
    )
    statistically_supported_all = improved_all and all(
        row["mcnemar_holm_p_value"] < float(protocol["formal"]["familywise_alpha"])
        and row["bootstrap_ci_low"] > 0
        for row in contrasts
    )
    conclusion = {
        "classification": (
            "all_budgets_improved_within_compliance_margin_and_statistical_support"
            if statistically_supported_all
            else "all_budgets_directionally_improved_within_compliance_margin"
            if improved_all
            else "mixed_or_negative_formal_result"
        ),
        "all_budgets_accuracy_improved_within_compliance_margin": improved_all,
        "registered_max_compliance_drop": max_compliance_drop,
        "all_budgets_holm_significant": statistically_supported_all,
        "formal_test_used_for_retuning": False,
        "formal_test_is_adaptive_diagnostic": bool(protocol.get("evidence_level")),
        "training_seed_variability_estimated": False,
    }
    run_csv = output_dir / "run_metrics.csv"
    contrast_csv = output_dir / "sft_vs_kd_contrasts.csv"
    context_contrast_csv = output_dir / "same_prompt_vs_teacher_prompt_contrasts.csv"
    logit_csv = output_dir / "matched_logit_metrics.csv"
    _write_csv(run_csv, run_rows)
    _write_csv(contrast_csv, contrasts)
    if prompt_context_contrasts:
        _write_csv(context_contrast_csv, prompt_context_contrasts)
    _write_csv(logit_csv, logit_rows)
    figure_paths = _figures(output_dir, run_rows, logit_rows, kd_label)
    analysis_path = output_dir / "logit_kd_analysis.json"
    analysis = {
        "status": "complete",
        "scope": "GSM8K only",
        "protocol_hash": protocol_hash(protocol),
        "protocol_variant": protocol["protocol_variant"],
        "supervision_mode": supervision_mode(protocol),
        "method": "online exact logit-level KD with completion-only hard CE plus forward KL",
        "context_mode": context_mode,
        "teacher_context_field": protocol["kd"].get("teacher_context_field", "prompt"),
        "student_context_field": protocol["kd"].get("student_context_field", "prompt"),
        "selected_alpha": alpha,
        "selected_temperature": temperature,
        "run_metrics": run_rows,
        "contrasts": contrasts,
        "prompt_context_contrasts": prompt_context_contrasts,
        "logit_metrics": logit_rows,
        "conclusion": conclusion,
        "inputs": {
            "parent_evidence_sha256": file_sha256(result_root / "preflight" / "parent_evidence.json"),
            "selection_sha256": file_sha256(result_root / "validation" / "selection.json"),
        },
        "source_sha256": {
            "scripts/11_2_analyze_logit_kd_experiment.py": file_sha256(Path(__file__).resolve()),
            "src/length_budget_distill/factorial_analysis.py": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/factorial_analysis.py"
            ),
        },
    }
    write_json(analysis_path, analysis)
    report_path = output_dir / "experiment_report.md"
    lines = [
        "# 7B-to-1.5B Teacher-Prompt Logit Distillation Experiment",
        "",
        "This is a single-seed adaptive GSM8K prompt-context ablation and does not estimate training-seed variability.",
        "",
        f"Training supervision mode: {supervision_mode(protocol)}.",
        "",
        f"KD context mode: {context_mode}.",
        "",
        f"Selected shared hyperparameters: alpha={alpha:g}, temperature={temperature:g}.",
        "",
        "## Formal results",
        "",
        "| Budget | SFT accuracy | KD accuracy | Delta | SFT compliance | KD compliance | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for contrast in contrasts:
        budget_name = contrast["budget_name"]
        sft = next(row for row in run_rows if row["method"] == "SFT" and row["budget_name"] == budget_name)
        kd = next(row for row in run_rows if row["method"] == kd_label and row["budget_name"] == budget_name)
        lines.append(
            f"| {budget_name} | {sft['accuracy']:.4f} | {kd['accuracy']:.4f} | "
            f"{contrast['accuracy_delta']:+.4f} | {sft['budget_compliance']:.4f} | "
            f"{kd['budget_compliance']:.4f} | {contrast['mcnemar_holm_p_value']:.4g} |"
        )
    if prompt_context_contrasts:
        lines.extend(
            [
                "",
                "## Teacher-prompt versus same-prompt KD",
                "",
                "| Budget | Accuracy delta | Compliance delta | Mean-token delta | Holm p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for contrast in prompt_context_contrasts:
            lines.append(
                f"| {contrast['budget_name']} | "
                f"{contrast['accuracy_delta_teacher_prompt_vs_same_prompt']:+.4f} | "
                f"{contrast['budget_compliance_delta_teacher_prompt_vs_same_prompt']:+.4f} | "
                f"{contrast['mean_output_token_delta_teacher_prompt_vs_same_prompt']:+.2f} | "
                f"{contrast['mcnemar_holm_p_value']:.4g} |"
            )
    lines.extend(["", "## Conclusion", "", conclusion["classification"], ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifacts = [analysis_path, run_csv, contrast_csv, logit_csv, report_path, *figure_paths]
    if prompt_context_contrasts:
        artifacts.append(context_contrast_csv)
    artifact_manifest = output_dir / "analysis_artifact_manifest.json"
    write_json(
        artifact_manifest,
        {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "artifacts": [
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
    )
    write_json(
        output_dir / "ANALYSIS_COMPLETE",
        {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "classification": conclusion["classification"],
            "artifact_manifest_sha256": file_sha256(artifact_manifest),
        },
    )
    logging.info("logit_kd_analysis_complete classification=%s", conclusion["classification"])


if __name__ == "__main__":
    main()
