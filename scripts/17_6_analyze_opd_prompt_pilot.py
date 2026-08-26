#!/usr/bin/env python3
"""Analyze common-prompt accuracy, length, and teacher-token cost for the OPD pilot."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256
from length_budget_distill.opd import (
    OPD_ARMS,
    protocol_hash,
    read_json,
    validate_opd_protocol,
    validated_opd_adapter,
)
from length_budget_distill.opd_analysis import (
    completed_opd_evaluation,
    opd_advancement_decision,
    paired_opd_contrast,
)
from length_budget_distill.records import read_jsonl


MODEL_ORDER = ("base",) + OPD_ARMS
MODEL_LABELS = {
    "base": "Raw 1.5B base",
    "standard_prompt": "OPD: standard prompt",
    "bounded_concise_prompt": "OPD: bounded-concise prompt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--primary-eval-manifest", required=True)
    parser.add_argument("--secondary-eval-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    output_dir = _resolve(args.output_dir)
    figure_dir = _resolve(args.figure_dir)
    if output_dir.exists() and args.skip_complete:
        if _completed_analysis(protocol, output_dir):
            logging.info("opd_analysis_already_complete output=%s", output_dir)
            return
        raise ValueError(f"Existing OPD analysis failed validation: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite OPD analysis: {output_dir}")
    figure_prefixes = [
        figure_dir / "opd_accuracy_and_output_length",
        figure_dir / "opd_accuracy_vs_teacher_scored_tokens",
        figure_dir / "opd_training_dynamics",
    ]
    existing_figures = [
        prefix.with_suffix(suffix)
        for prefix in figure_prefixes
        for suffix in (".png", ".pdf")
        if prefix.with_suffix(suffix).exists()
    ]
    if existing_figures:
        raise FileExistsError(f"Refusing to overwrite OPD figures: {existing_figures}")
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=True)

    training = _load_training_evidence(protocol)
    manifests = {
        "primary_evaluation": _resolve(args.primary_eval_manifest),
        "secondary_evaluation": _resolve(args.secondary_eval_manifest),
    }
    predictions: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    metrics: List[Dict[str, Any]] = []
    eval_evidence: Dict[str, Dict[str, Any]] = {}
    for split_name, manifest_path in manifests.items():
        split_predictions, split_metrics, evidence = _load_evaluation(
            protocol,
            split_name,
            manifest_path,
            training,
        )
        predictions[split_name] = split_predictions
        metrics.extend(split_metrics)
        eval_evidence[split_name] = evidence

    contrasts: List[Dict[str, Any]] = []
    evaluation = protocol["evaluation"]
    contrast_pairs = (
        ("bounded_concise_prompt", "standard_prompt", "primary_prompt_arm_comparison"),
        ("standard_prompt", "base", "standard_opd_vs_base"),
        ("bounded_concise_prompt", "base", "concise_opd_vs_base"),
    )
    for split_offset, split_name in enumerate(manifests):
        for pair_offset, (left, right, contrast_name) in enumerate(contrast_pairs):
            result = paired_opd_contrast(
                predictions[split_name][left],
                predictions[split_name][right],
                bootstrap_samples=int(evaluation["bootstrap_samples"]),
                bootstrap_seed=(
                    int(evaluation["bootstrap_seed"]) + 100 * split_offset + pair_offset
                ),
            )
            contrasts.append(
                {
                    "split_name": split_name,
                    "contrast_name": contrast_name,
                    "left_model": left,
                    "right_model": right,
                    **result,
                }
            )
    primary = next(
        row
        for row in contrasts
        if row["split_name"] == "primary_evaluation"
        and row["contrast_name"] == "primary_prompt_arm_comparison"
    )
    decision = opd_advancement_decision(primary, protocol["advancement_gate"])

    metrics.sort(key=lambda row: (row["split_name"], MODEL_ORDER.index(row["model_id"])))
    analysis_path = output_dir / "opd_prompt_pilot_analysis.json"
    metrics_path = output_dir / "run_metrics.csv"
    contrasts_path = output_dir / "paired_contrasts.csv"
    dynamics_path = output_dir / "training_dynamics.csv"
    report_path = output_dir / "experiment_report.md"
    dynamics = _training_dynamics(training)
    analysis = {
        "status": "complete",
        "experiment_name": protocol["experiment_name"],
        "protocol_variant": protocol["protocol_variant"],
        "evidence_level": protocol["evidence_level"],
        "scope": protocol["scope"],
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "evaluation_evidence": eval_evidence,
        "metrics": metrics,
        "contrasts": contrasts,
        "advancement_decision": decision,
        "interpretation_boundary": (
            "The decision compares the two OPD prompt arms under the common standard evaluation "
            "prompt. It is exploratory single-seed evidence, not a formal performance claim."
        ),
        "limitations": [
            "Only training seed 17 is included; training-seed variability is not estimated.",
            "The primary cohort is disjoint GSM8K train[6500:7473], not the locked formal test cohort.",
            "The secondary GSM8K test[50:1319] analysis is adaptive and descriptive.",
            "Teacher-scored tokens measure completion-token scoring, not wall-clock or total prompt-token FLOPs.",
        ],
    }
    _write_json_exclusive(analysis_path, analysis)
    _write_csv(metrics_path, metrics)
    _write_csv(contrasts_path, contrasts)
    _write_csv(dynamics_path, dynamics)
    _plot_accuracy_and_length(metrics, figure_prefixes[0])
    _plot_accuracy_vs_cost(metrics, figure_prefixes[1])
    _plot_training_dynamics(dynamics, figure_prefixes[2])
    _write_report(report_path, protocol, metrics, primary, decision, training)

    artifacts = [
        analysis_path,
        metrics_path,
        contrasts_path,
        dynamics_path,
        report_path,
        *[
            prefix.with_suffix(suffix)
            for prefix in figure_prefixes
            for suffix in (".png", ".pdf")
        ],
    ]
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    artifact_manifest = {
        "status": "complete",
        "protocol_hash": protocol_hash(protocol),
        "analysis_source_sha256": file_sha256(Path(__file__).resolve()),
        "analysis_library_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/opd_analysis.py"
        ),
        "artifacts": [
            {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    _write_json_exclusive(artifact_manifest_path, artifact_manifest)
    _write_json_exclusive(
        output_dir / "ANALYSIS_COMPLETE",
        {
            "status": "complete",
            "evidence_level": protocol["evidence_level"],
            "protocol_hash": protocol_hash(protocol),
            "artifact_manifest_path": str(artifact_manifest_path),
            "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        },
    )
    logging.info("opd_analysis_complete decision=%s output=%s", decision["classification"], output_dir)


def _load_training_evidence(protocol: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    checkpoint_root = _resolve(protocol["outputs"]["checkpoint_root"])
    result_root = _resolve(protocol["outputs"]["result_root"])
    evidence: Dict[str, Dict[str, Any]] = {}
    for arm in OPD_ARMS:
        adapter_dir = checkpoint_root / "pilot" / arm
        marker = validated_opd_adapter(
            protocol,
            arm=arm,
            adapter_dir=adapter_dir,
            stage="pilot",
        )
        if marker is None:
            raise ValueError(f"Invalid OPD adapter: {adapter_dir}")
        metrics_path = adapter_dir / "training_metrics.json"
        training_metrics = read_json(metrics_path)
        arm_manifest_path = result_root / "pilot/training" / arm / "training_manifest.json"
        arm_manifest = read_json(arm_manifest_path)
        checks = {
            "status": "complete",
            "stage": "pilot",
            "arm": arm,
            "protocol_hash": protocol_hash(protocol),
            "adapter_model_sha256": marker["adapter_model_sha256"],
            "adapter_config_sha256": marker["adapter_config_sha256"],
            "train_manifest_sha256": marker["train_manifest_sha256"],
        }
        if any(arm_manifest.get(key) != value for key, value in checks.items()):
            raise ValueError(f"OPD training manifest mismatch: {arm_manifest_path}")
        if int(training_metrics.get("rollouts", -1)) != int(arm_manifest.get("rollouts", -2)):
            raise ValueError(f"OPD training count mismatch: {arm}")
        evidence[arm] = {
            "adapter_dir": str(adapter_dir),
            "adapter_marker": marker,
            "training_metrics_path": str(metrics_path),
            "training_metrics_sha256": file_sha256(metrics_path),
            "training_metrics": training_metrics,
            "arm_manifest_path": str(arm_manifest_path),
            "arm_manifest_sha256": file_sha256(arm_manifest_path),
        }
    return evidence


def _load_evaluation(
    protocol: Mapping[str, Any],
    split_name: str,
    manifest_path: Path,
    training: Mapping[str, Mapping[str, Any]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    manifest = read_json(manifest_path)
    expected_manifest = {
        "status": "complete",
        "stage": "pilot",
        "split_name": split_name,
        "prompt_mode": "common_standard_prompt",
        "protocol_hash": protocol_hash(protocol),
        "run_count": len(MODEL_ORDER),
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError(f"Incomplete or mismatched evaluation manifest: {manifest_path}")
    source_checks = {
        "evaluation_source_sha256": PROJECT_ROOT / "scripts/17_5_eval_opd_model.py",
        "launcher_source_sha256": PROJECT_ROOT / "scripts/17_5_launch_opd_evaluation.py",
        "analysis_library_sha256": PROJECT_ROOT / "src/length_budget_distill/opd_analysis.py",
    }
    for field, path in source_checks.items():
        if manifest.get(field) != file_sha256(path):
            raise ValueError(f"Evaluation source hash mismatch: {field}")

    by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metric_rows: List[Dict[str, Any]] = []
    support: List[str] | None = None
    for run in manifest.get("runs", []):
        model_id = str(run.get("model_id", ""))
        if model_id in by_model or model_id not in MODEL_ORDER:
            raise ValueError(f"Invalid duplicate/unknown evaluation model: {model_id}")
        if run.get("eval_status") not in {"complete", "skipped_complete"}:
            raise ValueError(f"Incomplete OPD evaluation: {model_id}")
        evidence = completed_opd_evaluation(
            protocol,
            split_name=split_name,
            model_id=model_id,
            prediction_path=run["prediction_path"],
            summary_path=run["summary_path"],
        )
        if evidence is None:
            raise ValueError(f"Invalid OPD evaluation artifacts: {model_id}")
        for field in ("prediction_sha256", "summary_sha256", "prediction_count"):
            if run.get(field) != evidence[field]:
                raise ValueError(f"Evaluation evidence mismatch: {model_id} {field}")
        if support is not None and support != evidence["problem_ids"]:
            raise ValueError(f"Evaluation support mismatch: {model_id}")
        support = evidence["problem_ids"] if support is None else support
        rows = list(read_jsonl(Path(run["prediction_path"])))
        by_model[model_id] = {str(row["problem_id"]): dict(row) for row in rows}
        cost = 0 if model_id == "base" else int(training[model_id]["training_metrics"]["sampled_tokens"])
        rollouts = 0 if model_id == "base" else int(training[model_id]["training_metrics"]["rollouts"])
        metric_rows.append(
            {
                "split_name": split_name,
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "teacher_scored_completion_tokens": cost,
                "training_rollouts": rollouts,
                **evidence["metrics"],
            }
        )
    if set(by_model) != set(MODEL_ORDER):
        raise ValueError(f"Evaluation model identities mismatch: {sorted(by_model)}")
    return by_model, metric_rows, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "problem_count": len(support or []),
    }


def _training_dynamics(training: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm in OPD_ARMS:
        cumulative_prompts = 0
        cumulative_rollouts = 0
        cumulative_tokens = 0
        cumulative_steps = 0
        for batch in training[arm]["training_metrics"]["batch_metrics"]:
            cumulative_prompts += int(batch["prompt_count"])
            cumulative_rollouts += int(batch["rollout_count"])
            cumulative_tokens += int(batch["token_count"])
            cumulative_steps += int(batch["optimizer_steps"])
            rows.append(
                {
                    "arm": arm,
                    "batch_index": int(batch["batch_index"]),
                    "cumulative_prompts": cumulative_prompts,
                    "cumulative_rollouts": cumulative_rollouts,
                    "cumulative_teacher_scored_completion_tokens": cumulative_tokens,
                    "cumulative_optimizer_steps": cumulative_steps,
                    "mean_advantage": float(batch["mean_advantage"]),
                    "mean_output_tokens": float(batch["mean_output_tokens"]),
                    "diagnostic_rollout_accuracy": float(batch["diagnostic_accuracy"]),
                    "concise_in_band_rate": float(batch["concise_in_band_rate"]),
                    "mean_loss": float(batch["mean_loss"]),
                    "clip_fraction": float(batch["clip_fraction"]),
                }
            )
    return rows


def _plot_accuracy_and_length(metrics: List[Mapping[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    primary = {row["model_id"]: row for row in metrics if row["split_name"] == "primary_evaluation"}
    labels = [MODEL_LABELS[model_id] for model_id in MODEL_ORDER]
    colors = ["#7A7A7A", "#3B6FB6", "#D07A2D"]
    x_values = list(range(len(MODEL_ORDER)))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    accuracy = [100.0 * float(primary[model_id]["accuracy"]) for model_id in MODEL_ORDER]
    lower = [
        100.0 * (float(primary[model_id]["accuracy"]) - float(primary[model_id]["wilson_ci_low"]))
        for model_id in MODEL_ORDER
    ]
    upper = [
        100.0 * (float(primary[model_id]["wilson_ci_high"]) - float(primary[model_id]["accuracy"]))
        for model_id in MODEL_ORDER
    ]
    axes[0].bar(x_values, accuracy, color=colors, yerr=[lower, upper], capsize=4)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Common standard-prompt accuracy")
    lengths = [float(primary[model_id]["mean_output_tokens"]) for model_id in MODEL_ORDER]
    axes[1].bar(x_values, lengths, color=colors)
    axes[1].set_ylabel("Mean output tokens")
    axes[1].set_title("Common standard-prompt response length")
    for axis in axes:
        axis.set_xticks(x_values, labels, rotation=18, ha="right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Exploratory seed-17 OPD prompt-arm comparison")
    figure.tight_layout()
    _save_figure(figure, prefix)


def _plot_accuracy_vs_cost(metrics: List[Mapping[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    primary = [row for row in metrics if row["split_name"] == "primary_evaluation"]
    colors = {"base": "#7A7A7A", "standard_prompt": "#3B6FB6", "bounded_concise_prompt": "#D07A2D"}
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for row in primary:
        model_id = str(row["model_id"])
        axis.scatter(
            float(row["teacher_scored_completion_tokens"]),
            100.0 * float(row["accuracy"]),
            color=colors[model_id],
            s=75,
            label=MODEL_LABELS[model_id],
            zorder=3,
        )
    axis.set_xlabel("Teacher-scored completion tokens during OPD")
    axis.set_ylabel("Primary accuracy (%)")
    axis.set_title("Final accuracy versus teacher scoring cost")
    axis.grid(alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, prefix)


def _plot_training_dynamics(rows: List[Mapping[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"standard_prompt": "#3B6FB6", "bounded_concise_prompt": "#D07A2D"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for arm in OPD_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        x_values = [float(row["cumulative_teacher_scored_completion_tokens"]) for row in arm_rows]
        axes[0].plot(
            x_values,
            [float(row["mean_advantage"]) for row in arm_rows],
            color=colors[arm],
            label=MODEL_LABELS[arm],
            linewidth=1.4,
        )
        axes[1].plot(
            x_values,
            [float(row["mean_output_tokens"]) for row in arm_rows],
            color=colors[arm],
            label=MODEL_LABELS[arm],
            linewidth=1.4,
        )
    axes[0].set_ylabel("Batch mean log p_teacher - log p_old")
    axes[0].set_title("Dense teacher signal")
    axes[1].set_ylabel("Batch mean output tokens")
    axes[1].set_title("On-policy rollout length")
    for axis in axes:
        axis.set_xlabel("Cumulative teacher-scored completion tokens")
        axis.grid(alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, prefix)


def _save_figure(figure: Any, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def _write_report(
    path: Path,
    protocol: Mapping[str, Any],
    metrics: List[Mapping[str, Any]],
    primary: Mapping[str, Any],
    decision: Mapping[str, Any],
    training: Mapping[str, Mapping[str, Any]],
) -> None:
    primary_metrics = {
        row["model_id"]: row for row in metrics if row["split_name"] == "primary_evaluation"
    }
    lines = [
        "# Pure OPD prompt-arm pilot report",
        "",
        f"Evidence level: `{protocol['evidence_level']}`. Scope: {protocol['scope']}.",
        "",
        "Both trained policies were evaluated with the same standard prompt. The bounded-concise "
        "instruction is therefore tested for behavior internalized during OPD, rather than supplied at evaluation.",
        "",
        "## Primary result",
        "",
        f"Registered decision: `{decision['classification']}` (`{decision['status']}`).",
        "",
        f"Concise minus standard accuracy: {100.0 * float(primary['accuracy_difference']):.2f} pp "
        f"(paired 95% bootstrap CI {100.0 * float(primary['accuracy_difference_ci_low']):.2f} to "
        f"{100.0 * float(primary['accuracy_difference_ci_high']):.2f} pp).",
        "",
        f"Concise/standard mean output-token ratio: {float(primary['mean_output_token_ratio']):.3f}.",
        "",
        "| Model | Accuracy | Mean tokens | Extraction failure | Truncation | Teacher-scored tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_id in MODEL_ORDER:
        row = primary_metrics[model_id]
        lines.append(
            f"| {MODEL_LABELS[model_id]} | {100.0 * float(row['accuracy']):.2f}% | "
            f"{float(row['mean_output_tokens']):.1f} | "
            f"{100.0 * float(row['answer_extraction_failure_rate']):.2f}% | "
            f"{100.0 * float(row['truncation_rate']):.2f}% | "
            f"{int(row['teacher_scored_completion_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "## Objective boundary",
            "",
            "The loss uses only sampled-token teacher and old-student log probabilities with a clipped "
            "importance ratio. Gold correctness, response length, and length-band compliance are diagnostics only.",
            "",
            "## Training evidence",
            "",
        ]
    )
    for arm in OPD_ARMS:
        train_metrics = training[arm]["training_metrics"]
        lines.append(
            f"- `{arm}`: {int(train_metrics['rollouts'])} rollouts, "
            f"{int(train_metrics['sampled_tokens'])} teacher-scored completion tokens, "
            f"{int(train_metrics['optimizer_steps'])} optimizer steps."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a single-seed exploratory pilot and does not estimate training-seed variability. "
            "The primary result uses disjoint GSM8K train data; the test-cohort result is secondary and adaptive. "
            "The teacher-token cost excludes prompt tokens and does not represent total FLOPs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _completed_analysis(protocol: Mapping[str, Any], output_dir: Path) -> bool:
    marker_path = output_dir / "ANALYSIS_COMPLETE"
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    if not marker_path.is_file() or not artifact_manifest_path.is_file():
        return False
    try:
        marker = read_json(marker_path)
        manifest = read_json(artifact_manifest_path)
    except (OSError, ValueError):
        return False
    if (
        marker.get("status") != "complete"
        or marker.get("protocol_hash") != protocol_hash(protocol)
        or marker.get("artifact_manifest_sha256") != file_sha256(artifact_manifest_path)
        or manifest.get("status") != "complete"
        or manifest.get("protocol_hash") != protocol_hash(protocol)
        or manifest.get("analysis_source_sha256") != file_sha256(Path(__file__).resolve())
        or manifest.get("analysis_library_sha256")
        != file_sha256(PROJECT_ROOT / "src/length_budget_distill/opd_analysis.py")
    ):
        return False
    for artifact in manifest.get("artifacts", []):
        path = Path(str(artifact.get("path", "")))
        if (
            not path.is_file()
            or artifact.get("sha256") != file_sha256(path)
            or int(artifact.get("size_bytes", -1)) != path.stat().st_size
        ):
            return False
    return len(manifest.get("artifacts", [])) == 11


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
