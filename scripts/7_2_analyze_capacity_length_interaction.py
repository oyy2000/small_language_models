#!/usr/bin/env python3
"""Analyze generator-capacity by CoT-length effects on formal GSM8K results."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, expected_conditions, file_sha256, trace_condition
from length_budget_distill.factorial_analysis import (
    difference_in_differences_effects,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_problem_effects,
)
from length_budget_distill.records import read_jsonl, trace_from_dict


PredictionKey = Tuple[str, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument("--eval-manifest-glob", required=True)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--selected-traces", required=True)
    parser.add_argument("--selection-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["smoke", "formal"], default="formal")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config.pop("_config_path", None)
    config_hash = canonical_sha256(config)
    expected_n = int(config["evaluation"][f"{args.stage}_limit"])
    generators = {item["name"]: item for item in config["generators"]}
    budget_tokens = {item["name"]: int(item["max_solution_tokens"]) for item in config["length_budgets"]}
    conditions = expected_conditions(config)
    configured_seeds = [int(seed) for seed in config["balancing"]["training_seeds"]]
    active_seeds, protocol_variant, dataset_manifest_evidence = _resolve_active_seeds(
        args.dataset_manifest,
        config_hash=config_hash,
        configured_seeds=configured_seeds,
    )
    selection_audit = _read_json(Path(args.selection_audit))
    if selection_audit.get("status") != "passed" or selection_audit.get("config_hash") != config_hash:
        raise ValueError(f"Selection audit is incomplete or mismatched: {args.selection_audit}")
    teacher_quality_rows = list(selection_audit.get("conditions", []))
    observed_quality_conditions = {
        (str(row.get("generator_name")), str(row.get("budget_name"))) for row in teacher_quality_rows
    }
    if observed_quality_conditions != set(conditions):
        raise ValueError("Selection audit does not contain the exact registered condition matrix.")

    eval_manifests = [Path(path) for path in sorted(glob.glob(args.eval_manifest_glob))]
    if not eval_manifests:
        raise FileNotFoundError(f"No eval manifests matched {args.eval_manifest_glob!r}")
    runs: Dict[str, Dict[str, Any]] = {}
    for path in eval_manifests:
        manifest = _read_json(path)
        if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
            raise ValueError(f"Incomplete or mismatched evaluation manifest: {path}")
        for run in manifest.get("runs", []):
            run_name = str(run["run_name"])
            if run_name in runs:
                raise ValueError(f"Duplicate evaluation run: {run_name}")
            if run.get("eval_status") not in {"complete", "skipped_complete"}:
                raise ValueError(f"Incomplete evaluation run: {run_name}")
            runs[run_name] = run

    prediction_maps: Dict[PredictionKey, Dict[Tuple[str, int], bool]] = {}
    run_rows: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []
    for run_name in sorted(runs):
        run = runs[run_name]
        prediction_path = Path(run["prediction_path"])
        summary_path = Path(run["summary_path"])
        if run.get("prediction_sha256") != file_sha256(prediction_path):
            raise ValueError(f"Prediction hash mismatch for {run_name}")
        if run.get("summary_sha256") != file_sha256(summary_path):
            raise ValueError(f"Evaluation summary hash mismatch for {run_name}")
        predictions = list(read_jsonl(prediction_path))
        if len(predictions) != expected_n:
            raise ValueError(f"Prediction count mismatch for {run_name}: expected={expected_n} actual={len(predictions)}")
        accuracy = sum(bool(row["is_correct"]) for row in predictions) / len(predictions)
        avg_output_tokens = mean(float(row.get("output_token_count", 0)) for row in predictions)
        mode = run.get("mode") or run.get("overrides", {}).get("mode")
        generator_name = run.get("generator_name") or run.get("overrides", {}).get("generator_name")
        budget_name = run.get("budget_name") or run.get("overrides", {}).get("budget_name")
        seed = run.get("seed") if run.get("seed") is not None else run.get("overrides", {}).get("seed")
        run_rows.append(
            {
                "run_name": run_name,
                "mode": mode,
                "generator_name": generator_name,
                "generator_size_b": generators.get(generator_name, {}).get("size_b"),
                "budget_name": budget_name,
                "budget_tokens": budget_tokens.get(budget_name),
                "seed": seed,
                "n": len(predictions),
                "accuracy": accuracy,
                "avg_output_tokens": avg_output_tokens,
                "baseline_name": run.get("baseline_name") or run.get("overrides", {}).get("baseline_name"),
            }
        )
        if mode not in {"equal_example", "equal_token"}:
            continue
        if generator_name is None or budget_name is None or seed is None:
            raise ValueError(f"Factorial metadata missing for {run_name}")
        key: PredictionKey = (str(mode), str(generator_name), str(budget_name), int(seed))
        mapping: Dict[Tuple[str, int], bool] = {}
        for row in predictions:
            identity = (str(row["problem_id"]), int(seed))
            if identity in mapping:
                raise ValueError(f"Duplicate prediction identity in {run_name}: {identity}")
            mapping[identity] = bool(row["is_correct"])
            regression_rows.append(
                {
                    "mode": mode,
                    "generator_name": generator_name,
                    "budget_name": budget_name,
                    "seed": int(seed),
                    "problem_id": row["problem_id"],
                    "is_correct": int(bool(row["is_correct"])),
                }
            )
        prediction_maps[key] = mapping

    expected_factorial_keys = {
        (mode, generator_name, budget_name, seed)
        for mode in ("equal_example", "equal_token")
        for generator_name, budget_name in conditions
        for seed in active_seeds
    }
    missing_keys = sorted(expected_factorial_keys - set(prediction_maps))
    if missing_keys:
        raise ValueError(f"Missing factorial prediction runs; examples={missing_keys[:10]}")

    trace_rows = _trace_summary(Path(args.selected_traces), generators, budget_tokens)
    contrasts = _planned_contrasts(
        prediction_maps,
        generators,
        budget_tokens,
        samples=args.bootstrap_samples,
    )
    adjusted = holm_adjust([float(row["p_value"]) for row in contrasts])
    for row, adjusted_p in zip(contrasts, adjusted):
        row["holm_p_value"] = adjusted_p

    regression = _fit_clustered_logistic_models(regression_rows)
    conclusion = _classify_result(
        contrasts,
        regression,
        stage=args.stage,
        protocol_variant=protocol_variant,
        active_seeds=active_seeds,
        configured_seeds=configured_seeds,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "capacity_length_analysis.json"
    run_metrics_path = output_dir / "run_metrics.csv"
    contrast_path = output_dir / "planned_contrasts.csv"
    report_path = output_dir / "experiment_report.md"
    _write_json(
        analysis_path,
        {
            "status": "complete",
            "stage": args.stage,
            "config_hash": config_hash,
            "scope": "GSM8K only",
            "method": "offline black-box sequence-level response distillation implemented with SFT",
            "run_count": len(run_rows),
            "protocol_variant": protocol_variant,
            "training_seeds": active_seeds,
            "contrasts": contrasts,
            "regression": regression,
            "conclusion": conclusion,
            "trace_summary": trace_rows,
            "teacher_quality": teacher_quality_rows,
            "inputs": {
                "selected_traces_path": args.selected_traces,
                "selected_traces_sha256": file_sha256(args.selected_traces),
                "selection_audit_path": args.selection_audit,
                "selection_audit_sha256": file_sha256(args.selection_audit),
                "dataset_manifest": dataset_manifest_evidence,
                "eval_manifests": [
                    {"path": str(path), "sha256": file_sha256(path)} for path in eval_manifests
                ],
            },
        },
    )
    _write_csv(run_metrics_path, run_rows)
    _write_csv(contrast_path, contrasts)
    _write_figures(output_dir, run_rows, trace_rows, teacher_quality_rows, generators, budget_tokens)
    _write_report(
        report_path,
        run_rows,
        contrasts,
        regression,
        conclusion,
        teacher_quality_rows,
        stage=args.stage,
    )
    artifact_paths = [
        analysis_path,
        run_metrics_path,
        contrast_path,
        report_path,
        output_dir / "capacity_length_interaction.png",
        output_dir / "capacity_length_pareto.png",
        output_dir / "equal_example_vs_equal_token.png",
        output_dir / "teacher_trace_quality.png",
    ]
    artifact_manifest_path = output_dir / "analysis_artifact_manifest.json"
    _write_json(
        artifact_manifest_path,
        {
            "status": "complete",
            "config_hash": config_hash,
            "artifacts": [
                {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in artifact_paths
            ],
        },
    )
    (output_dir / "ANALYSIS_COMPLETE").write_text(
        f"config_hash={config_hash}\nrun_count={len(run_rows)}\n"
        f"classification={conclusion['classification']}\n"
        f"artifact_manifest_sha256={file_sha256(artifact_manifest_path)}\n",
        encoding="utf-8",
    )
    logging.info("analysis_complete output=%s classification=%s", output_dir, conclusion["classification"])


def _planned_contrasts(
    predictions: Mapping[PredictionKey, Mapping[Tuple[str, int], bool]],
    generators: Mapping[str, Mapping[str, Any]],
    budget_tokens: Mapping[str, int],
    samples: int,
) -> List[Dict[str, Any]]:
    by_size = sorted(generators, key=lambda name: float(generators[name]["size_b"]))
    by_budget = sorted(budget_tokens, key=lambda name: budget_tokens[name])
    smallest, largest = by_size[0], by_size[-1]
    shortest, longest = by_budget[0], by_budget[-1]
    seeds = sorted({key[3] for key in predictions})
    contrasts: List[Dict[str, Any]] = []
    for mode in ("equal_example", "equal_token"):
        for generator_name in by_size:
            left = _combine_seed_maps(predictions, mode, generator_name, shortest, seeds)
            right = _combine_seed_maps(predictions, mode, generator_name, longest, seeds)
            result = paired_cluster_bootstrap(
                paired_problem_effects(left, right),
                samples=samples,
                seed=_contrast_seed(mode, generator_name, "length"),
            )
            contrasts.append(
                {
                    "mode": mode,
                    "contrast": "short_minus_long",
                    "generator_name": generator_name,
                    "budget_name": None,
                    **result,
                }
            )
        for budget_name in by_budget:
            left = _combine_seed_maps(predictions, mode, largest, budget_name, seeds)
            right = _combine_seed_maps(predictions, mode, smallest, budget_name, seeds)
            result = paired_cluster_bootstrap(
                paired_problem_effects(left, right),
                samples=samples,
                seed=_contrast_seed(mode, budget_name, "capacity"),
            )
            contrasts.append(
                {
                    "mode": mode,
                    "contrast": "largest_minus_smallest_generator",
                    "generator_name": None,
                    "budget_name": budget_name,
                    **result,
                }
            )
        effects = difference_in_differences_effects(
            _combine_seed_maps(predictions, mode, largest, shortest, seeds),
            _combine_seed_maps(predictions, mode, largest, longest, seeds),
            _combine_seed_maps(predictions, mode, smallest, shortest, seeds),
            _combine_seed_maps(predictions, mode, smallest, longest, seeds),
        )
        contrasts.append(
            {
                "mode": mode,
                "contrast": "difference_in_differences",
                "generator_name": f"{largest}_vs_{smallest}",
                "budget_name": f"{shortest}_vs_{longest}",
                **paired_cluster_bootstrap(
                    effects,
                    samples=samples,
                    seed=_contrast_seed(mode, "interaction", "did"),
                ),
            }
        )
    return contrasts


def _combine_seed_maps(
    predictions: Mapping[PredictionKey, Mapping[Tuple[str, int], bool]],
    mode: str,
    generator_name: str,
    budget_name: str,
    seeds: List[int],
) -> Dict[Tuple[str, int], bool]:
    combined: Dict[Tuple[str, int], bool] = {}
    for seed in seeds:
        combined.update(predictions[(mode, generator_name, budget_name, seed)])
    return combined


def _contrast_seed(*parts: str) -> int:
    return int(canonical_sha256(list(parts))[:8], 16)


def _fit_clustered_logistic_models(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("Install pandas and statsmodels to run the registered factorial analysis.") from exc
    frame = pd.DataFrame(rows)
    output: Dict[str, Any] = {}
    for mode in ("equal_example", "equal_token"):
        subset = frame[frame["mode"] == mode].copy()
        fit = smf.glm(
            "is_correct ~ C(generator_name) * C(budget_name) + C(seed)",
            data=subset,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": subset["problem_id"]})
        interaction_indices = [index for index, name in enumerate(fit.params.index) if ":" in name]
        restriction = [[1.0 if column == index else 0.0 for column in range(len(fit.params))] for index in interaction_indices]
        wald = fit.wald_test(restriction, scalar=True)
        confidence = fit.conf_int()
        output[mode] = {
            "n": int(fit.nobs),
            "interaction_wald_statistic": float(wald.statistic),
            "interaction_p_value": float(wald.pvalue),
            "interaction_df": len(interaction_indices),
            "parameters": {
                name: {
                    "coefficient": float(fit.params[name]),
                    "p_value": float(fit.pvalues[name]),
                    "ci_low": float(confidence.loc[name, 0]),
                    "ci_high": float(confidence.loc[name, 1]),
                }
                for name in fit.params.index
            },
        }
    return output


def _classify_result(
    contrasts: List[Dict[str, Any]],
    regression: Dict[str, Any],
    stage: str,
    protocol_variant: str,
    active_seeds: List[int],
    configured_seeds: List[int],
) -> Dict[str, Any]:
    primary = [row for row in contrasts if row["mode"] == "equal_example"]
    interaction_p = float(regression["equal_example"]["interaction_p_value"])
    capacity_short = next(
        row
        for row in primary
        if row["contrast"] == "largest_minus_smallest_generator" and "128" in str(row["budget_name"])
    )
    length_rows = [row for row in primary if row["contrast"] == "short_minus_long"]
    if capacity_short["holm_p_value"] < 0.05 and capacity_short["estimate"] > 0:
        statistical_pattern = "larger_generator_short_cots_better"
    elif capacity_short["holm_p_value"] < 0.05 and capacity_short["estimate"] < 0:
        statistical_pattern = "smaller_generator_short_cots_better"
    elif interaction_p >= 0.05 and all(row["estimate"] > 0 for row in length_rows) and any(
        row["holm_p_value"] < 0.05 for row in length_rows
    ):
        statistical_pattern = "primarily_length_effect"
    elif interaction_p < 0.05:
        statistical_pattern = "capacity_length_interaction_mixed"
    else:
        statistical_pattern = "inconclusive"
    classification = statistical_pattern if stage == "formal" else "smoke_only_no_scientific_conclusion"
    if stage == "formal" and active_seeds != configured_seeds:
        evidence_level = "revised_formal_single_seed" if len(active_seeds) == 1 else "revised_formal_seed_subset"
    else:
        evidence_level = "registered_formal" if stage == "formal" else "pipeline_smoke_only"
    result = {
        "classification": classification,
        "statistical_pattern": statistical_pattern,
        "evidence_level": evidence_level,
        "protocol_variant": protocol_variant,
        "training_seeds": active_seeds,
        "interaction_p_value": interaction_p,
        "scope": "GSM8K only",
        "robustness_required": "Compare direction and confidence intervals with equal_token results.",
    }
    if active_seeds != configured_seeds:
        result["limitation"] = "Training-seed variability is not estimated by this reduced run."
    return result


def _resolve_active_seeds(
    dataset_manifest_path: str | None,
    *,
    config_hash: str,
    configured_seeds: List[int],
) -> tuple[List[int], str, Dict[str, Any] | None]:
    if dataset_manifest_path is None:
        return configured_seeds, "registered_parent_protocol", None
    path = Path(dataset_manifest_path)
    manifest = _read_json(path)
    if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
        raise ValueError(f"Dataset manifest is incomplete or mismatched: {path}")
    seeds = [int(seed) for seed in manifest.get("training_seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Dataset manifest must contain unique active training seeds.")
    if not set(seeds) <= set(configured_seeds):
        raise ValueError("Dataset manifest contains seeds outside the parent protocol.")
    return (
        seeds,
        str(manifest.get("protocol_variant", "registered_parent_protocol")),
        {"path": str(path), "sha256": file_sha256(path)},
    )


def _trace_summary(
    path: Path,
    generators: Mapping[str, Mapping[str, Any]],
    budget_tokens: Mapping[str, int],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
    for row in read_jsonl(path):
        trace = trace_from_dict(row)
        grouped[trace_condition(trace)].append(trace)
    output = []
    for condition, traces in sorted(grouped.items()):
        output.append(
            {
                "generator_name": condition[0],
                "generator_size_b": generators[condition[0]]["size_b"],
                "budget_name": condition[1],
                "budget_tokens": budget_tokens[condition[1]],
                "n": len(traces),
                "avg_selected_tokens": mean(trace.solution_token_count for trace in traces),
            }
        )
    return output


def _write_figures(
    output_dir: Path,
    run_rows: List[Dict[str, Any]],
    trace_rows: List[Dict[str, Any]],
    teacher_quality_rows: List[Dict[str, Any]],
    generators: Mapping[str, Mapping[str, Any]],
    budget_tokens: Mapping[str, int],
) -> None:
    import matplotlib.pyplot as plt

    factorial_rows = [row for row in run_rows if row["mode"] in {"equal_example", "equal_token"}]
    sizes = sorted(generators, key=lambda name: float(generators[name]["size_b"]))
    budgets = sorted(budget_tokens, key=lambda name: budget_tokens[name])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, mode in zip(axes, ("equal_example", "equal_token")):
        for generator_name in sizes:
            values = []
            for budget_name in budgets:
                bucket = [
                    row["accuracy"]
                    for row in factorial_rows
                    if row["mode"] == mode
                    and row["generator_name"] == generator_name
                    and row["budget_name"] == budget_name
                ]
                values.append(mean(bucket))
            axis.plot(
                [budget_tokens[name] for name in budgets],
                values,
                marker="o",
                label=f"{generators[generator_name]['size_b']}B",
            )
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("CoT budget (tokens)")
        axis.set_ylabel("GSM8K accuracy")
        axis.grid(alpha=0.25)
    axes[1].legend(title="Generator")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_length_interaction.png", dpi=220)
    plt.close(fig)

    primary = [row for row in factorial_rows if row["mode"] == "equal_example"]
    trace_lookup = {(row["generator_name"], row["budget_name"]): row for row in trace_rows}
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    for generator_name in sizes:
        points = []
        for budget_name in budgets:
            accuracy = mean(
                row["accuracy"]
                for row in primary
                if row["generator_name"] == generator_name and row["budget_name"] == budget_name
            )
            points.append((trace_lookup[(generator_name, budget_name)]["avg_selected_tokens"], accuracy, budget_name))
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=f"{generators[generator_name]['size_b']}B")
    axis.set_xlabel("Average selected CoT tokens")
    axis.set_ylabel("GSM8K accuracy")
    axis.set_title("Capacity-length utility frontier")
    axis.grid(alpha=0.25)
    axis.legend(title="Generator")
    fig.tight_layout()
    fig.savefig(output_dir / "capacity_length_pareto.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    for generator_name in sizes:
        for budget_name in budgets:
            x = mean(
                row["accuracy"]
                for row in factorial_rows
                if row["mode"] == "equal_example" and row["generator_name"] == generator_name and row["budget_name"] == budget_name
            )
            y = mean(
                row["accuracy"]
                for row in factorial_rows
                if row["mode"] == "equal_token" and row["generator_name"] == generator_name and row["budget_name"] == budget_name
            )
            axis.scatter(x, y)
            axis.annotate(f"{generators[generator_name]['size_b']}B/{budget_tokens[budget_name]}", (x, y), fontsize=7)
    bounds = [row["accuracy"] for row in factorial_rows]
    lower, upper = min(bounds), max(bounds)
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("Equal-example accuracy")
    axis.set_ylabel("Equal-token accuracy")
    axis.set_title("Training-control robustness")
    fig.tight_layout()
    fig.savefig(output_dir / "equal_example_vs_equal_token.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for generator_name in sizes:
        rows = {
            row["budget_name"]: row
            for row in teacher_quality_rows
            if row["generator_name"] == generator_name
        }
        x_values = [budget_tokens[name] for name in budgets]
        axes[0].plot(
            x_values,
            [float(rows[name]["pass_at_3"]) for name in budgets],
            marker="o",
            label=f"{generators[generator_name]['size_b']}B",
        )
        axes[1].plot(
            x_values,
            [float(rows[name]["correct_candidate_count"]) / float(rows[name]["candidate_count"]) for name in budgets],
            marker="o",
            label=f"{generators[generator_name]['size_b']}B",
        )
    axes[0].set_title("Teacher pass@3")
    axes[0].set_ylabel("Problem success rate")
    axes[1].set_title("Teacher candidate correctness")
    axes[1].set_ylabel("Candidate correctness rate")
    for axis in axes:
        axis.set_xlabel("Requested CoT budget (tokens)")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[1].legend(title="Generator")
    fig.tight_layout()
    fig.savefig(output_dir / "teacher_trace_quality.png", dpi=220)
    plt.close(fig)


def _write_report(
    path: Path,
    run_rows: List[Dict[str, Any]],
    contrasts: List[Dict[str, Any]],
    regression: Dict[str, Any],
    conclusion: Dict[str, Any],
    teacher_quality_rows: List[Dict[str, Any]],
    stage: str,
) -> None:
    if stage == "formal":
        scope_text = "Registered conclusions use the locked GSM8K test[50:1319] cohort."
        interpretation_text = "The classification below is the registered formal result."
    else:
        scope_text = (
            "This smoke run uses GSM8K test[:50], which is a pipeline-validation cohort and is not "
            "authorized for scientific conclusions."
        )
        interpretation_text = (
            "The statistical pattern is retained for debugging, but the registered conclusion remains "
            "`smoke_only_no_scientific_conclusion`."
        )
    lines = [
        "# Capacity-Length Factorial Experiment Report",
        "",
        "## Method",
        "",
        "Teacher responses were generated offline with vLLM and used as completion-only SFT targets. "
        "This is black-box sequence-level response distillation, not logit-level KL distillation.",
        "",
        "## Scope",
        "",
        scope_text,
        "",
        "## Teacher Trace Quality",
        "",
        "Teacher pass@3 and candidate correctness are reported descriptively by generator capacity and requested length. "
        "They are kept separate from downstream student accuracy.",
        "",
        "## Result Classification",
        "",
        f"- Classification: `{conclusion['classification']}`",
        f"- Statistical pattern: `{conclusion['statistical_pattern']}`",
        f"- Evidence level: `{conclusion['evidence_level']}`",
        f"- Equal-example interaction p-value: {conclusion['interaction_p_value']:.6g}",
        f"- Evaluated runs: {len(run_rows)}",
        f"- Interpretation: {interpretation_text}",
        "",
        "## Planned Contrasts",
        "",
    ]
    for row in contrasts:
        lines.append(
            f"- {row['mode']} / {row['contrast']} / {row.get('generator_name') or row.get('budget_name')}: "
            f"effect={row['estimate']:.4f}, 95% CI [{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
            f"Holm p={row['holm_p_value']:.6g}."
        )
    lines.extend(["", "## Teacher Quality Cells", ""])
    for row in sorted(teacher_quality_rows, key=lambda item: (item["generator_name"], item["budget_name"])):
        candidate_accuracy = float(row["correct_candidate_count"]) / float(row["candidate_count"])
        lines.append(
            f"- {row['generator_name']} / {row['budget_name']}: "
            f"pass@3={float(row['pass_at_3']):.4f}, candidate accuracy={candidate_accuracy:.4f}."
        )
    lines.extend(
        [
            "",
            "## Registered Interaction Tests",
            "",
            f"- Equal-example: p={regression['equal_example']['interaction_p_value']:.6g}",
            f"- Equal-token: p={regression['equal_token']['interaction_p_value']:.6g}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
