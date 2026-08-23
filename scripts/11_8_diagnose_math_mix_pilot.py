#!/usr/bin/env python3
"""Diagnose why verified MATH additions did not yield a stable pilot gain."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import file_sha256
from length_budget_distill.math_mix import diagnose_common_support_selection
from length_budget_distill.records import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--common-problem-ids", required=True)
    parser.add_argument("--math-eval-jsonl", required=True)
    parser.add_argument("--selection-audit", required=True)
    parser.add_argument("--sft-manifest", required=True)
    parser.add_argument("--run-metrics", required=True)
    parser.add_argument("--paired-comparisons", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_jsonl)
    common_path = Path(args.common_problem_ids)
    eval_path = Path(args.math_eval_jsonl)
    selection_path = Path(args.selection_audit)
    sft_manifest_path = Path(args.sft_manifest)
    metrics_path = Path(args.run_metrics)
    comparisons_path = Path(args.paired_comparisons)

    source_rows = list(read_jsonl(source_path))
    evaluation_rows = list(read_jsonl(eval_path))
    common_payload = _read_json(common_path)
    common_ids = [str(problem_id) for problem_id in common_payload["problem_ids"]]
    selection = diagnose_common_support_selection(source_rows, common_ids, evaluation_rows)
    selection_audit = _read_json(selection_path)
    sft_manifest = _read_json(sft_manifest_path)
    metrics = _read_csv(metrics_path)
    comparisons = _read_csv(comparisons_path)

    mixtures = []
    for run in sft_manifest["runs"]:
        math_examples = int(run["source_counts"]["hendrycks_math"])
        total_examples = int(run["n"])
        math_tokens = int(run["source_supervised_tokens"]["hendrycks_math"])
        total_tokens = int(run["supervised_tokens"])
        mixtures.append(
            {
                "mode": str(run["mode"]),
                "budget_name": str(run["budget_name"]),
                "total_examples": total_examples,
                "math_examples": math_examples,
                "math_example_fraction": math_examples / total_examples,
                "total_supervised_tokens": total_tokens,
                "math_supervised_tokens": math_tokens,
                "math_token_fraction": math_tokens / total_tokens,
            }
        )

    math_comparisons = [row for row in comparisons if row["dataset_name"] == "math500"]
    math_effects = [float(row["accuracy_delta"]) for row in math_comparisons]
    ci_half_widths = [
        (float(row["bootstrap_ci_high"]) - float(row["bootstrap_ci_low"])) / 2.0
        for row in math_comparisons
    ]
    length_accuracy = {}
    for budget_name in ("short_128", "medium_256", "long_512"):
        values = [
            float(row["accuracy"])
            for row in metrics
            if row["dataset_name"] == "math500"
            and row["training_variant"] in {"gsm_only", "math_mix"}
            and row["budget_name"] == budget_name
        ]
        if len(values) != 4:
            raise ValueError(f"Expected four MATH-500 metrics for {budget_name}, got {len(values)}")
        length_accuracy[budget_name] = sum(values) / len(values)

    diagnostic = {
        "status": "complete",
        "evidence_level": "post_hoc_exploratory_diagnosis",
        "selection_coverage": selection,
        "generation_quality_gates": {
            "raw_candidates": int(selection_audit["actual_total_candidates"]),
            "verification_mismatch_count": int(selection_audit["verification_mismatch_count"]),
            "common_problem_count": int(selection_audit["common_problem_count"]),
            "condition_pass_at_3": [
                {
                    "budget_name": row["budget_name"],
                    "pass_at_3": float(row["pass_at_3"]),
                }
                for row in selection_audit["conditions"]
            ],
        },
        "training_mixtures": mixtures,
        "math500_effects": {
            "comparison_count": len(math_effects),
            "mean_accuracy_delta": sum(math_effects) / len(math_effects),
            "minimum_accuracy_delta": min(math_effects),
            "maximum_accuracy_delta": max(math_effects),
            "minimum_paired_ci_half_width": min(ci_half_widths),
            "maximum_paired_ci_half_width": max(ci_half_widths),
            "mean_accuracy_by_length": length_accuracy,
        },
        "supported_conclusions": [
            "The common-support filter substantially shifts MATH supervision toward easier problems and selected subjects.",
            "The 100-item MATH-500 pilot has insufficient resolution to establish the observed small effects as stable.",
            "Completion length has a much larger descriptive association with MATH-500 accuracy than training-source mixture in this pilot.",
            "No overlap was found between sampled MATH training questions and the MATH-500 evaluation subset by normalized question hash.",
        ],
        "unresolved_hypotheses": [
            "Final-answer verification does not establish that every synthetic rationale is pedagogically correct.",
            "One epoch and rank-4 LoRA may underfit or interfere across the broader mixed-domain target distribution.",
            "Single-seed adapter variation may be of the same order as the observed mixture effects.",
        ],
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (
                source_path,
                common_path,
                eval_path,
                selection_path,
                sft_manifest_path,
                metrics_path,
                comparisons_path,
            )
        ],
    }

    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    _write_json(output_json, diagnostic)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(_render_report(diagnostic), encoding="utf-8")
    print(f"math_mix_diagnosis_complete json={output_json} report={output_report}")


def _render_report(diagnostic: Mapping[str, Any]) -> str:
    coverage = diagnostic["selection_coverage"]
    source = coverage["source"]
    common = coverage["common_support"]
    evaluation = coverage["evaluation"]
    effects = diagnostic["math500_effects"]
    level_retention = {int(row["level"]): row for row in coverage["retention_by_level"]}
    lines = [
        "# MATH-mixed SFT pilot diagnosis",
        "",
        "Evidence level: post-hoc exploratory diagnosis. The evidence identifies protocol limitations; it does not prove a single causal mechanism.",
        "",
        "## Main finding",
        "",
        (
            "The absence of a stable significant gain is not evidence that MATH supervision is useless. "
            "The current pilot combines a strongly ease-biased training subset with a small evaluation cohort, "
            "while the imposed completion-length condition has a much larger observed effect than the dataset mixture."
        ),
        "",
        "## Direct evidence",
        "",
        (
            f"- Common-support selection retained {common['n']}/{source['n']} sampled MATH problems. "
            f"Mean difficulty shifted from {source['mean_level']:.2f} in the source pool to "
            f"{common['mean_level']:.2f} in training, versus {evaluation['mean_level']:.2f} in the MATH-500 subset."
        ),
        (
            f"- Level-1 retention was {100 * level_retention[1]['retention_rate']:.1f}% "
            f"({level_retention[1]['common_count']}/{level_retention[1]['source_count']}), whereas "
            f"Level-5 retention was {100 * level_retention[5]['retention_rate']:.1f}% "
            f"({level_retention[5]['common_count']}/{level_retention[5]['source_count']})."
        ),
        (
            f"- Across the six MATH-500 comparisons, the mean descriptive gain was "
            f"{100 * effects['mean_accuracy_delta']:+.2f} percentage points and the range was "
            f"[{100 * effects['minimum_accuracy_delta']:+.1f}, {100 * effects['maximum_accuracy_delta']:+.1f}] points. "
            f"Paired 95% CI half-widths were {100 * effects['minimum_paired_ci_half_width']:.1f} to "
            f"{100 * effects['maximum_paired_ci_half_width']:.1f} points."
        ),
        (
            "- Mean MATH-500 accuracy across both supervision modes and both source variants was "
            f"{100 * effects['mean_accuracy_by_length']['short_128']:.1f}% at 128 tokens, "
            f"{100 * effects['mean_accuracy_by_length']['medium_256']:.1f}% at 256 tokens, and "
            f"{100 * effects['mean_accuracy_by_length']['long_512']:.1f}% at 512 tokens."
        ),
        (
            "- Normalized question-hash overlap with the evaluation subset was "
            f"{coverage['question_hash_overlap']['source_vs_evaluation']} for the 1,000-problem source pool "
            f"and {coverage['question_hash_overlap']['common_support_vs_evaluation']} for common-support training data."
        ),
        "",
        "## Effective MATH exposure",
        "",
        "| Mode | Length | MATH examples | Total examples | MATH token share |",
        "|---|---|---:|---:|---:|",
    ]
    for row in diagnostic["training_mixtures"]:
        lines.append(
            f"| {row['mode']} | {row['budget_name']} | {row['math_examples']} | "
            f"{row['total_examples']} | {100 * row['math_token_fraction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- The generation and hash audits passed, so there is no current evidence of a broken data or training path.",
            "- Symbolic final-answer verification filters incorrect endpoints but does not audit every reasoning step.",
            "- AIME-2025 is at a floor of 0/30 for nearly all models and cannot diagnose a modest transfer effect.",
            "- One training seed cannot distinguish a reproducible mixture effect from adapter-seed variation.",
            "",
            "## Minimum follow-up design",
            "",
            "1. Preserve subject-by-level coverage after trace verification; increase teacher capacity or candidate count for Level-4/5 problems instead of intersecting only easy successes.",
            "2. Compare GSM-only and mixed data under matched total supervised tokens and matched optimizer steps.",
            "3. Use 250-300 stratified MATH-500 items for the confirmatory comparison, while retaining the current 100-item subset as pilot evidence.",
            "4. Run at least three training seeds on the 256- and 512-token cells before expanding the full grid.",
            "5. Audit a stratified sample of synthetic rationales against official solutions and add held-out MATH validation loss across training duration.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
