#!/usr/bin/env python3
"""Analyze the locked seed-17/42/73 ranked-length GSM8K evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
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
from length_budget_distill.ranked_evaluation import completed_evaluation_evidence
from length_budget_distill.ranked_evaluation_analysis import paired_contrast, summarize_predictions
from length_budget_distill.ranked_multiseed_analysis import (
    crossed_seed_problem_bootstrap,
    seed_problem_effects,
)
from length_budget_distill.ranked_multiseed_evaluation import (
    LENGTH_BUDGETS,
    model_id,
    validate_all_parent_trainings,
)


RANK_LABELS = {
    "relative_short": "Short-ranked SFT",
    "relative_medium": "Medium-ranked SFT",
    "relative_long": "Long-ranked SFT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    training_runs = validate_all_parent_trainings(config, PROJECT_ROOT)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    evaluation = dict(config["evaluation"])
    analysis_config = dict(config["analysis"])
    seeds = [int(seed) for seed in analysis_config["training_seeds"]]
    ranks = [str(rank) for rank in analysis_config["ranks"]]

    manifest_path = _resolve(args.eval_manifest)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
        raise ValueError(f"Evaluation manifest is incomplete or mismatched: {manifest_path}")
    if int(manifest.get("run_count", -1)) != int(evaluation["expected_run_count"]):
        raise ValueError("Evaluation manifest run count mismatch.")

    expected_ids = {"base"}
    expected_ids.update(model_id("qwen2p5_7b", rank, seed) for seed in seeds for rank in ranks)
    by_model: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metrics: List[Dict[str, Any]] = []
    common_support: set[str] | None = None
    for run in manifest.get("runs", []):
        current_id = str(run.get("model_id", ""))
        if current_id in by_model:
            raise ValueError(f"Duplicate evaluation model: {current_id}")
        if run.get("eval_status") not in {"complete", "skipped_complete"}:
            raise ValueError(f"Incomplete evaluation model: {current_id}")
        evidence = completed_evaluation_evidence(
            run["prediction_path"],
            run["summary_path"],
            expected_n=int(evaluation["limit"]),
            expected_start_index=int(evaluation["start_index"]),
            expected_split=str(evaluation["dataset_split"]),
        )
        if evidence is None:
            raise ValueError(f"Invalid evaluation artifacts for {current_id}")
        for field in ("prediction_sha256", "summary_sha256"):
            if run.get(field) != evidence[field]:
                raise ValueError(f"Evaluation hash mismatch for {current_id}: {field}")
        rows = list(read_jsonl(Path(str(run["prediction_path"]))))
        mapping = {str(row["problem_id"]): dict(row) for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate prediction identity for {current_id}")
        if common_support is None:
            common_support = set(mapping)
        elif set(mapping) != common_support:
            raise ValueError(f"Evaluation problem support mismatch for {current_id}")
        training_examples = run.get("training_examples")
        supervised_tokens = run.get("supervised_tokens")
        metrics.append(
            {
                "model_id": current_id,
                "model_label": (
                    "Base 1.5B"
                    if current_id == "base"
                    else f"{RANK_LABELS[str(run['budget_name'])]} (seed {int(run['seed'])})"
                ),
                "budget_name": run.get("budget_name"),
                "seed": run.get("seed"),
                "training_examples": training_examples,
                "supervised_tokens": supervised_tokens,
                "mean_supervised_tokens": (
                    float(supervised_tokens) / int(training_examples)
                    if training_examples and supervised_tokens is not None
                    else None
                ),
                **summarize_predictions(rows),
            }
        )
        predictions[current_id] = mapping
        by_model[current_id] = dict(run)
    if set(by_model) != expected_ids:
        raise ValueError(
            f"Evaluation model identities mismatch: expected={sorted(expected_ids)} "
            f"actual={sorted(by_model)}"
        )
    assert common_support is not None

    per_seed_predictions = {
        seed: {
            rank: predictions[model_id("qwen2p5_7b", rank, seed)]
            for rank in ranks
        }
        for seed in seeds
    }
    per_seed_contrasts: List[Dict[str, Any]] = []
    aggregate_contrasts: List[Dict[str, Any]] = []
    pairs = [(str(pair[0]), str(pair[1])) for pair in analysis_config["primary_pairs"]]
    for seed_value in seeds:
        family: List[Dict[str, Any]] = []
        for left_rank, right_rank in pairs:
            contrast_name = f"{left_rank}__vs__{right_rank}"
            family.append(
                {
                    "seed": seed_value,
                    "contrast": contrast_name,
                    "left_rank": left_rank,
                    "right_rank": right_rank,
                    **paired_contrast(
                        per_seed_predictions[seed_value][left_rank],
                        per_seed_predictions[seed_value][right_rank],
                        bootstrap_samples=int(analysis_config["bootstrap_samples"]),
                        bootstrap_seed=int(
                            canonical_sha256([config_hash, seed_value, contrast_name])[:8], 16
                        ),
                    ),
                }
            )
        adjusted = holm_adjust([float(row["mcnemar_p_value"]) for row in family])
        for row, adjusted_p in zip(family, adjusted):
            row["within_seed_mcnemar_holm_p_value"] = adjusted_p
            row["within_seed_holm_significant"] = adjusted_p < float(
                analysis_config["familywise_alpha"]
            )
            per_seed_contrasts.append(row)

    for left_rank, right_rank in pairs:
        contrast_name = f"{left_rank}__vs__{right_rank}"
        effects = seed_problem_effects(per_seed_predictions, left_rank, right_rank)
        aggregate_contrasts.append(
            {
                "contrast": contrast_name,
                "left_rank": left_rank,
                "right_rank": right_rank,
                **crossed_seed_problem_bootstrap(
                    effects,
                    samples=int(analysis_config["bootstrap_samples"]),
                    seed=int(canonical_sha256([config_hash, "crossed", contrast_name])[:8], 16),
                ),
            }
        )
    adjusted = holm_adjust(
        [float(row["bootstrap_p_value"]) for row in aggregate_contrasts]
    )
    for row, adjusted_p in zip(aggregate_contrasts, adjusted):
        row["bootstrap_holm_p_value"] = adjusted_p
        row["holm_significant"] = adjusted_p < float(analysis_config["familywise_alpha"])

    metrics.sort(key=lambda row: _metric_sort_key(row, seeds, ranks))
    aggregate_metrics = _aggregate_rank_metrics(metrics, ranks)
    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "ranked_multiseed_analysis.json"
    metrics_path = output_dir / "run_metrics.csv"
    per_seed_path = output_dir / "per_seed_paired_contrasts.csv"
    aggregate_path = output_dir / "aggregate_paired_contrasts.csv"
    report_path = output_dir / "experiment_report.md"
    figure_prefix = figure_dir / "ranked_multiseed_accuracy_and_output_length"
    _write_json(
        analysis_path,
        {
            "status": "complete",
            "experiment_name": config["experiment_name"],
            "protocol_variant": config["protocol_variant"],
            "evidence_level": analysis_config["evidence_level"],
            "scope": analysis_config["scope"],
            "config_path": str(config_path),
            "config_hash": config_hash,
            "config_file_sha256": file_sha256(config_path),
            "eval_manifest": str(manifest_path),
            "eval_manifest_sha256": file_sha256(manifest_path),
            "training_parent_count": len(config["parent_trainings"]),
            "validated_adapter_count": len(training_runs),
            "evaluated_model_count": len(metrics),
            "training_seed_count": len(seeds),
            "problem_count": len(common_support),
            "metrics": metrics,
            "aggregate_metrics": aggregate_metrics,
            "per_seed_contrasts": per_seed_contrasts,
            "aggregate_contrasts": aggregate_contrasts,
            "limitations": [
                "Only three training seeds are available, so seed-level uncertainty remains imprecise.",
                "All seeds reuse the same locked and previously observed GSM8K test[50:1319] cohort; this does not add independent evaluation questions.",
                "Training sets have equal example counts but unequal supervised-token totals across ranks.",
                "The evidence scope is GSM8K only.",
            ],
        },
    )
    _write_csv(metrics_path, metrics)
    _write_csv(per_seed_path, per_seed_contrasts)
    _write_csv(aggregate_path, aggregate_contrasts)
    _plot_results(metrics, aggregate_metrics, seeds, ranks, figure_prefix)
    _write_report(report_path, config, metrics, aggregate_metrics, aggregate_contrasts)
    artifacts = [
        analysis_path,
        metrics_path,
        per_seed_path,
        aggregate_path,
        report_path,
        figure_prefix.with_suffix(".png"),
        figure_prefix.with_suffix(".pdf"),
    ]
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {
            "status": "complete",
            "config_hash": config_hash,
            "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
            "analysis_library_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/ranked_multiseed_analysis.py"
            ),
            "shared_analysis_library_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/ranked_evaluation_analysis.py"
            ),
            "artifact_count": len(artifacts),
            "artifacts": [
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in artifacts
            ],
        },
    )
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        f"status=complete\nconfig_hash={config_hash}\n"
        f"eval_manifest_sha256={file_sha256(manifest_path)}\n"
        f"artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n"
        f"evaluated_model_count={len(metrics)}\ntraining_seed_count={len(seeds)}\n"
        f"problem_count={len(common_support)}\n",
        encoding="utf-8",
    )
    logging.info("ranked_multiseed_analysis_complete report=%s", report_path)


def _aggregate_rank_metrics(
    metrics: List[Mapping[str, Any]], ranks: List[str]
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for rank in ranks:
        rows = [row for row in metrics if row.get("budget_name") == rank]
        if len(rows) != 3:
            raise ValueError(f"Expected three seed metrics for {rank}, found {len(rows)}")
        accuracies = [float(row["accuracy"]) for row in rows]
        lengths = [float(row["mean_output_tokens"]) for row in rows]
        result.append(
            {
                "budget_name": rank,
                "model_label": RANK_LABELS[rank],
                "seed_count": len(rows),
                "mean_accuracy": statistics.mean(accuracies),
                "accuracy_sample_sd": statistics.stdev(accuracies),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "mean_output_tokens": statistics.mean(lengths),
                "output_tokens_sample_sd": statistics.stdev(lengths),
            }
        )
    return result


def _metric_sort_key(
    row: Mapping[str, Any], seeds: List[int], ranks: List[str]
) -> tuple[int, int]:
    if row["model_id"] == "base":
        return (-1, -1)
    return (seeds.index(int(row["seed"])), ranks.index(str(row["budget_name"])))


def _plot_results(
    metrics: List[Mapping[str, Any]],
    aggregate_metrics: List[Mapping[str, Any]],
    seeds: List[int],
    ranks: List[str],
    output_prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    base = next(row for row in metrics if row["model_id"] == "base")
    by_seed_rank = {
        (int(row["seed"]), str(row["budget_name"])): row
        for row in metrics
        if row["model_id"] != "base"
    }
    x_values = list(range(len(ranks)))
    labels = ["Short", "Medium", "Long"]
    colors = ["#4c78a8", "#f2a541", "#c44e52"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    for seed_value, color in zip(seeds, colors):
        axes[0].plot(
            x_values,
            [100.0 * float(by_seed_rank[(seed_value, rank)]["accuracy"]) for rank in ranks],
            marker="o",
            color=color,
            alpha=0.82,
            label=f"Train seed {seed_value}",
        )
        axes[1].plot(
            x_values,
            [float(by_seed_rank[(seed_value, rank)]["mean_output_tokens"]) for rank in ranks],
            marker="o",
            color=color,
            alpha=0.82,
            label=f"Train seed {seed_value}",
        )
    means = [100.0 * float(row["mean_accuracy"]) for row in aggregate_metrics]
    sds = [100.0 * float(row["accuracy_sample_sd"]) for row in aggregate_metrics]
    axes[0].errorbar(
        x_values, means, yerr=sds, color="black", marker="D", linestyle="--", capsize=4,
        linewidth=1.8, label="3-seed mean +/- seed SD",
    )
    mean_lengths = [float(row["mean_output_tokens"]) for row in aggregate_metrics]
    length_sds = [float(row["output_tokens_sample_sd"]) for row in aggregate_metrics]
    axes[1].errorbar(
        x_values, mean_lengths, yerr=length_sds, color="black", marker="D", linestyle="--",
        capsize=4, linewidth=1.8, label="3-seed mean +/- seed SD",
    )
    axes[0].axhline(
        100.0 * float(base["accuracy"]), color="#777777", linestyle=":", label="Base 1.5B"
    )
    axes[1].axhline(
        float(base["mean_output_tokens"]), color="#777777", linestyle=":", label="Base 1.5B"
    )
    axes[0].set_ylabel("GSM8K accuracy (%)")
    axes[0].set_title("Accuracy across training seeds")
    axes[1].set_ylabel("Mean generated tokens")
    axes[1].set_title("Student response length across training seeds")
    for axis in axes:
        axis.set_xticks(x_values, labels)
        axis.set_xlabel("Teacher-response rank used for SFT")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Rank-selected 7B teacher supervision on locked GSM8K test[50:1319]")
    figure.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    config: Mapping[str, Any],
    metrics: List[Mapping[str, Any]],
    aggregate_metrics: List[Mapping[str, Any]],
    aggregate_contrasts: List[Mapping[str, Any]],
) -> None:
    base = next(row for row in metrics if row["model_id"] == "base")
    lines = [
        "# Ranked-length SFT multi-seed evaluation report",
        "",
        "## Protocol and evidence scope",
        "",
        "- Student: Qwen2.5-1.5B-Instruct with the unchanged registered LoRA recipe.",
        "- Teacher traces: sealed Qwen2.5-7B-Instruct ranked short/medium/long datasets, 881 equal-example problems per adapter.",
        "- Training seeds: 17, 42, and 73; nine adapters in total.",
        "- Evaluation: greedy decoding on locked GSM8K `test[50:1319]` (`n=1269`) with 512 maximum new tokens.",
        "- This reuses an already observed test cohort; it estimates training-seed variability but does not increase the number of independent evaluation questions.",
        "",
        "## Three-seed summary",
        "",
        "| Rank | Mean accuracy | Seed SD | Seed range | Mean output tokens | Token-length seed SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_metrics:
        lines.append(
            f"| {row['model_label']} | {100 * float(row['mean_accuracy']):.2f}% | "
            f"{100 * float(row['accuracy_sample_sd']):.2f} pp | "
            f"{100 * float(row['min_accuracy']):.2f}% to {100 * float(row['max_accuracy']):.2f}% | "
            f"{float(row['mean_output_tokens']):.1f} | {float(row['output_tokens_sample_sd']):.1f} |"
        )
    lines.extend(
        [
            f"| Base 1.5B (single deterministic evaluation) | {100 * float(base['accuracy']):.2f}% | n/a | n/a | {float(base['mean_output_tokens']):.1f} | n/a |",
            "",
            "## Crossed seed/problem paired contrasts",
            "",
            "Positive differences favor the left-hand rank. The interval resamples both three training seeds and the shared paired questions. Holm adjustment covers the three registered rank comparisons.",
            "",
            "| Contrast | Mean difference | Crossed-bootstrap 95% interval | Seed-specific differences | Holm-adjusted bootstrap p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_contrasts:
        seed_effects = ", ".join(
            f"{seed}: {100 * float(value):+.2f} pp"
            for seed, value in sorted(row["per_seed_effects"].items(), key=lambda item: int(item[0]))
        )
        lines.append(
            f"| {RANK_LABELS[row['left_rank']]} vs {RANK_LABELS[row['right_rank']]} | "
            f"{100 * float(row['estimate']):+.2f} pp | "
            f"[{100 * float(row['ci_low']):+.2f}, {100 * float(row['ci_high']):+.2f}] pp | "
            f"{seed_effects} | {float(row['bootstrap_holm_p_value']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Three seeds materially improve on a single-seed point estimate, but remain too few for a precise estimate of the training-seed distribution.",
            "- The same 1,269 paired GSM8K questions are reused for all seeds; the effective evaluation-question count remains 1,269, not 3,807.",
            "- The test cohort was previously inspected, so this is comparative replication evidence rather than a fresh confirmatory test.",
            "- Equal-example training does not equalize supervised tokens across short, medium, and long ranks.",
            "- No claim is made beyond GSM8K.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
