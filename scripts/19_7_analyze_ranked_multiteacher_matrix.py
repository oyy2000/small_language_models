#!/usr/bin/env python3
"""Analyze teacher-capacity by ranked-length effects across three training seeds."""

from __future__ import annotations

import argparse
import csv
import json
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
from length_budget_distill.records import read_jsonl
from length_budget_distill.ranked_evaluation import completed_evaluation_evidence
from length_budget_distill.ranked_evaluation_analysis import summarize_predictions
from length_budget_distill.ranked_multiteacher_analysis import analyze_accuracy_contrasts
from length_budget_distill.ranked_multiteacher import (
    LAUNCHER_ASSIGNMENT_POLICY,
    RANK_NAMES,
    TEACHER_NAMES,
    TRAINING_SEEDS,
    validate_launcher_assignment,
)
from length_budget_distill.ranked_multiteacher_evaluation import (
    matrix_model_id,
    validated_matrix_training_runs,
)


TEACHER_LABELS = {
    "qwen2p5_1p5b": "1.5B self-distillation control",
    "qwen2p5_3b": "3B teacher",
    "qwen2p5_7b": "7B teacher",
    "qwen2p5_14b": "14B teacher",
}
RANK_LABELS = {
    "relative_short": "Short",
    "relative_medium": "Medium",
    "relative_long": "Long",
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
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    training_runs = validated_matrix_training_runs(config, PROJECT_ROOT)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    evaluation = dict(config["evaluation"])
    analysis_config = dict(config["analysis"])
    launcher_plan_path = config_path.parent / "launcher_assignment_plan.json"
    launcher_plan = _read_json(launcher_plan_path)
    launcher_runs = [dict(run) for run in launcher_plan.get("runs", [])]
    validate_launcher_assignment(launcher_runs)
    if launcher_plan.get("status") != "complete":
        raise ValueError("Launcher assignment plan is incomplete.")
    if launcher_plan.get("config_hash") != config_hash:
        raise ValueError("Launcher assignment plan config mismatch.")
    if launcher_plan.get("assignment_sha256") != canonical_sha256(launcher_runs):
        raise ValueError("Launcher assignment hash mismatch.")
    operational_assignment = {
        "policy": LAUNCHER_ASSIGNMENT_POLICY,
        "launcher_plan": str(launcher_plan_path),
        "launcher_plan_sha256": file_sha256(launcher_plan_path),
        "assignment_sha256": launcher_plan["assignment_sha256"],
        "launcher_shards": 3,
        "launcher_waves": 4,
        "wave_barrier_policy": "declared_launcher_wave_barrier_v1",
    }
    manifest_path = _resolve(args.eval_manifest)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
        raise ValueError("Evaluation manifest is incomplete or mismatched.")
    if int(manifest.get("run_count", -1)) != 37:
        raise ValueError("Evaluation manifest run count mismatch.")

    expected_ids = {"base"} | {
        matrix_model_id(teacher, rank, seed)
        for teacher in TEACHER_NAMES for rank in RANK_NAMES for seed in TRAINING_SEEDS
    }
    predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metrics: List[Dict[str, Any]] = []
    support: set[str] | None = None
    for run in manifest.get("runs", []):
        current_id = str(run.get("model_id", ""))
        if current_id in predictions:
            raise ValueError(f"Duplicate evaluation model: {current_id}")
        if run.get("eval_status") not in {"complete", "skipped_complete"}:
            raise ValueError(f"Incomplete evaluation model: {current_id}")
        evidence = completed_evaluation_evidence(
            run["prediction_path"], run["summary_path"], expected_n=int(evaluation["limit"]),
            expected_start_index=int(evaluation["start_index"]), expected_split=str(evaluation["dataset_split"]),
        )
        if evidence is None:
            raise ValueError(f"Invalid evaluation artifacts: {current_id}")
        for field in ("prediction_sha256", "summary_sha256"):
            if run.get(field) != evidence[field]:
                raise ValueError(f"Evaluation hash mismatch: {current_id} {field}")
        rows = list(read_jsonl(Path(str(run["prediction_path"]))))
        mapping = {str(row["problem_id"]): dict(row) for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"Duplicate prediction identity: {current_id}")
        if support is None:
            support = set(mapping)
        elif set(mapping) != support:
            raise ValueError(f"Evaluation support mismatch: {current_id}")
        metrics.append(
            {
                "model_id": current_id,
                "generator_name": run.get("generator_name"),
                "budget_name": run.get("budget_name"),
                "seed": run.get("seed"),
                "training_examples": run.get("training_examples"),
                "supervised_tokens": run.get("supervised_tokens"),
                **summarize_predictions(rows),
            }
        )
        predictions[current_id] = mapping
    if set(predictions) != expected_ids:
        raise ValueError("Evaluation model identities mismatch.")
    assert support is not None

    nested = {
        teacher: {
            seed: {
                rank: predictions[matrix_model_id(teacher, rank, seed)]
                for rank in RANK_NAMES
            }
            for seed in TRAINING_SEEDS
        }
        for teacher in TEACHER_NAMES
    }
    aggregate_metrics = _aggregate_metrics(metrics)
    within_contrasts, interaction_rows = analyze_accuracy_contrasts(
        nested,
        within_teacher_pairs=analysis_config["within_teacher_pairs"],
        interaction_rank_pair=analysis_config["interaction_rank_pair"],
        teacher_interaction_pairs=analysis_config["teacher_interaction_pairs"],
        bootstrap_samples=int(analysis_config["bootstrap_samples"]),
        familywise_alpha=float(analysis_config["familywise_alpha"]),
        config_hash=config_hash,
    )

    metrics.sort(key=_metric_key)
    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "ranked_multiteacher_analysis.json"
    metrics_path = output_dir / "run_metrics.csv"
    aggregate_path = output_dir / "aggregate_cell_metrics.csv"
    within_path = output_dir / "within_teacher_rank_contrasts.csv"
    interaction_path = output_dir / "teacher_by_rank_interactions.csv"
    report_path = output_dir / "experiment_report.md"
    figure_prefix = figure_dir / "teacher_capacity_by_rank_accuracy_and_output_length"
    _write_json(
        analysis_path,
        {
            "status": "complete",
            "experiment_name": config["experiment_name"],
            "protocol_variant": config["protocol_variant"],
            "scope": analysis_config["scope"],
            "config_path": str(config_path),
            "config_hash": config_hash,
            "config_file_sha256": file_sha256(config_path),
            "eval_manifest": str(manifest_path),
            "eval_manifest_sha256": file_sha256(manifest_path),
            "validated_adapter_count": len(training_runs),
            "evaluated_model_count": len(metrics),
            "training_seed_count": 3,
            "problem_count": len(support),
            "metrics": metrics,
            "aggregate_metrics": aggregate_metrics,
            "within_teacher_contrasts": within_contrasts,
            "teacher_by_rank_interactions": interaction_rows,
            "operational_training_assignment": operational_assignment,
            "limitations": [
                "Only three training seeds are available.",
                "All conditions reuse the same previously observed locked GSM8K evaluation cohort.",
                "The 1.5B teacher is a self-distillation control, not an external larger teacher.",
                "The evidence scope is GSM8K only.",
            ],
        },
    )
    _write_csv(metrics_path, metrics)
    _write_csv(aggregate_path, aggregate_metrics)
    _write_csv(within_path, within_contrasts)
    _write_csv(interaction_path, interaction_rows)
    _plot(aggregate_metrics, metrics, figure_prefix)
    _write_report(report_path, config, aggregate_metrics, within_contrasts, interaction_rows)
    artifacts = [
        analysis_path, metrics_path, aggregate_path, within_path, interaction_path, report_path,
        figure_prefix.with_suffix(".png"), figure_prefix.with_suffix(".pdf"),
    ]
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {
            "status": "complete",
            "config_hash": config_hash,
            "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
            "bootstrap_library_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/ranked_multiseed_analysis.py"
            ),
            "matrix_analysis_library_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/ranked_multiteacher_analysis.py"
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
        "trained_adapter_count=36\nevaluated_model_count=37\ntraining_seed_count=3\n"
        f"problem_count={len(support)}\n",
        encoding="utf-8",
    )


def _aggregate_metrics(metrics: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for teacher in TEACHER_NAMES:
        for rank in RANK_NAMES:
            rows = [row for row in metrics if row.get("generator_name") == teacher and row.get("budget_name") == rank]
            if len(rows) != 3:
                raise ValueError(f"Expected three seeds for {teacher} {rank}.")
            accuracies = [float(row["accuracy"]) for row in rows]
            lengths = [float(row["mean_output_tokens"]) for row in rows]
            result.append(
                {
                    "generator_name": teacher,
                    "teacher_label": TEACHER_LABELS[teacher],
                    "budget_name": rank,
                    "rank_label": RANK_LABELS[rank],
                    "seed_count": 3,
                    "mean_accuracy": statistics.mean(accuracies),
                    "accuracy_sample_sd": statistics.stdev(accuracies),
                    "mean_output_tokens": statistics.mean(lengths),
                    "output_tokens_sample_sd": statistics.stdev(lengths),
                    "training_examples": rows[0]["training_examples"],
                    "supervised_tokens": rows[0]["supervised_tokens"],
                }
            )
    return result


def _plot(aggregate: List[Mapping[str, Any]], metrics: List[Mapping[str, Any]], output_prefix: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ["#4c78a8", "#f2a541", "#59a14f", "#c44e52"]
    x = list(range(3))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    for teacher, color in zip(TEACHER_NAMES, colors):
        rows = [row for row in aggregate if row["generator_name"] == teacher]
        axes[0].errorbar(
            x, [100 * float(row["mean_accuracy"]) for row in rows],
            yerr=[100 * float(row["accuracy_sample_sd"]) for row in rows], marker="o",
            capsize=3, color=color, label=TEACHER_LABELS[teacher],
        )
        axes[1].errorbar(
            x, [float(row["mean_output_tokens"]) for row in rows],
            yerr=[float(row["output_tokens_sample_sd"]) for row in rows], marker="o",
            capsize=3, color=color, label=TEACHER_LABELS[teacher],
        )
    base = next(row for row in metrics if row["model_id"] == "base")
    axes[0].axhline(100 * float(base["accuracy"]), color="#777777", linestyle=":", label="Base 1.5B")
    axes[1].axhline(float(base["mean_output_tokens"]), color="#777777", linestyle=":", label="Base 1.5B")
    axes[0].set_ylabel("GSM8K accuracy (%)")
    axes[0].set_title("Teacher capacity by supervision rank")
    axes[1].set_ylabel("Mean generated tokens")
    axes[1].set_title("Student response-length transfer")
    for axis in axes:
        axis.set_xticks(x, ["Short", "Medium", "Long"])
        axis.set_xlabel("Within-question teacher-response rank")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)
    figure.suptitle("Fixed 1.5B student, three training seeds, locked GSM8K test[50:1319]")
    figure.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path, config: Mapping[str, Any], aggregate: List[Mapping[str, Any]],
    within: List[Mapping[str, Any]], interactions: List[Mapping[str, Any]],
) -> None:
    lines = [
        "# Teacher-capacity by ranked-length main-matrix report", "",
        "## Protocol", "",
        "- Fixed student: Qwen2.5-1.5B-Instruct with unchanged rank-4 LoRA settings.",
        "- Teachers: Qwen2.5-1.5B, 3B, 7B, and 14B-Instruct; the 1.5B cell is a self-distillation control.",
        "- Matrix: 4 teachers x 3 within-question length ranks x 3 training seeds = 36 adapters.",
        "- Training uses the global problem intersection shared by all four teacher candidate pools and all ranks.",
        "- The 36 runs use a hash-bound node- and wave-balanced launcher plan with a barrier between four node-local waves.",
        "- Evaluation reuses locked GSM8K `test[50:1319]` (`n=1269`); it is not a fresh held-out test.",
        "", "## Cell means across seeds", "",
        "| Teacher | Rank | Accuracy | Seed SD | Mean output tokens | Token seed SD | Training n | Supervision tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['teacher_label']} | {row['rank_label']} | {100*float(row['mean_accuracy']):.2f}% | "
            f"{100*float(row['accuracy_sample_sd']):.2f} pp | {float(row['mean_output_tokens']):.1f} | "
            f"{float(row['output_tokens_sample_sd']):.1f} | {row['training_examples']} | {row['supervised_tokens']} |"
        )
    lines.extend([
        "", "## Registered within-teacher rank effects", "",
        "Positive estimates favor the left-hand rank. Holm adjustment covers all 12 teacher-specific rank contrasts.", "",
        "| Teacher | Contrast | Effect | 95% crossed-bootstrap interval | Holm p |", "|---|---|---:|---:|---:|",
    ])
    for row in within:
        lines.append(
            f"| {TEACHER_LABELS[row['teacher_name']]} | {RANK_LABELS[row['left_rank']]} vs {RANK_LABELS[row['right_rank']]} | "
            f"{100*float(row['estimate']):+.2f} pp | [{100*float(row['ci_low']):+.2f}, {100*float(row['ci_high']):+.2f}] pp | "
            f"{float(row['bootstrap_holm_p_value']):.4g} |"
        )
    lines.extend([
        "", "## Registered teacher-by-rank interactions", "",
        "Each estimate compares the short-minus-long effect between two teachers. Holm adjustment covers all six teacher pairs.", "",
        "| Teacher contrast | Interaction | 95% crossed-bootstrap interval | Holm p |", "|---|---:|---:|---:|",
    ])
    for row in interactions:
        lines.append(
            f"| {TEACHER_LABELS[row['left_teacher']]} vs {TEACHER_LABELS[row['right_teacher']]} | "
            f"{100*float(row['estimate']):+.2f} pp | [{100*float(row['ci_low']):+.2f}, {100*float(row['ci_high']):+.2f}] pp | "
            f"{float(row['bootstrap_holm_p_value']):.4g} |"
        )
    lines.extend([
        "", "## Limitations", "",
        "- Three training seeds provide a first variance estimate but not a precise seed-population distribution.",
        "- The evaluation cohort was previously inspected, so this is a controlled main-matrix comparison rather than a new confirmatory test.",
        "- Equal-example training does not equalize supervision tokens across ranks or teachers.",
        "- Claims remain limited to GSM8K.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    if row["model_id"] == "base":
        return (-1, -1, -1)
    return (
        TEACHER_NAMES.index(str(row["generator_name"])),
        RANK_NAMES.index(str(row["budget_name"])),
        TRAINING_SEEDS.index(int(row["seed"])),
    )


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
