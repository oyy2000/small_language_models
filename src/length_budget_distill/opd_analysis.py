"""Analysis helpers for the dual-prompt pure OPD pilot."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Mapping, Sequence

from .factorial import file_sha256
from .factorial_analysis import exact_mcnemar_p_value, paired_cluster_bootstrap, wilson_interval
from .opd import protocol_hash
from .records import read_jsonl
from .verifiers import extract_final_answer, verify_answer


def summarize_opd_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Prediction rows must be non-empty.")
    ids = [str(row["problem_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction rows contain duplicate problem IDs.")
    correct = sum(bool(row["is_correct"]) for row in rows)
    lengths = [int(row["output_token_count"]) for row in rows]
    if any(value < 0 for value in lengths):
        raise ValueError("Prediction output-token counts must be non-negative.")
    extraction_failures = sum(row.get("predicted_answer") is None for row in rows)
    truncations = sum(bool(row.get("hit_max_new_tokens")) for row in rows)
    eos = sum(bool(row.get("eos_emitted")) for row in rows)
    low, high = wilson_interval(correct, len(rows))
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "wilson_ci_low": low,
        "wilson_ci_high": high,
        "mean_output_tokens": mean(lengths),
        "median_output_tokens": median(lengths),
        "minimum_output_tokens": min(lengths),
        "maximum_output_tokens": max(lengths),
        "eos_rate": eos / len(rows),
        "truncation_rate": truncations / len(rows),
        "answer_extraction_failure_rate": extraction_failures / len(rows),
    }


def completed_opd_evaluation(
    protocol: Mapping[str, Any],
    *,
    split_name: str,
    model_id: str,
    prediction_path: str | Path,
    summary_path: str | Path,
) -> Dict[str, Any] | None:
    """Validate one hash-bound, common-prompt OPD evaluation artifact pair."""

    predictions = Path(prediction_path)
    summary_file = Path(summary_path)
    if not predictions.is_file() or not summary_file.is_file():
        return None
    try:
        with summary_file.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        rows = list(read_jsonl(predictions))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if split_name not in protocol.get("splits", {}):
        return None
    split = protocol["splits"][split_name]
    checks = {
        "status": "complete",
        "model_id": model_id,
        "split_name": split_name,
        "dataset_split": split["dataset_split"],
        "start_index": int(split["start_index"]),
        "limit": int(split["limit"]),
        "prompt_mode": "common_standard_prompt",
        "protocol_hash": protocol_hash(protocol),
        "prediction_path": str(predictions),
        "prediction_sha256": file_sha256(predictions),
    }
    if any(summary.get(key) != value for key, value in checks.items()):
        return None
    if len(rows) != int(split["limit"]):
        return None
    problem_ids = [str(row.get("problem_id", "")) for row in rows]
    if not all(problem_ids) or len(problem_ids) != len(set(problem_ids)):
        return None
    for row in rows:
        predicted = extract_final_answer(str(row.get("prediction_text", "")))
        if (
            row.get("model_id") != model_id
            or row.get("split_name") != split_name
            or row.get("prompt_mode") != "common_standard_prompt"
            or row.get("predicted_answer") != predicted
            or bool(row.get("is_correct"))
            != verify_answer(predicted, str(row.get("gold_answer", "")))
        ):
            return None
    metrics = summarize_opd_predictions(rows)
    if summary.get("metrics") != metrics:
        return None
    return {
        "prediction_sha256": file_sha256(predictions),
        "summary_sha256": file_sha256(summary_file),
        "prediction_count": len(rows),
        "problem_ids": problem_ids,
        "metrics": metrics,
    }


def paired_opd_contrast(
    concise: Mapping[str, Mapping[str, Any]],
    standard: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Paired concise-minus-standard accuracy, length, and failure effects."""

    if set(concise) != set(standard):
        raise ValueError("OPD prediction supports are not identical.")
    accuracy_effects: Dict[str, float] = {}
    length_effects: Dict[str, float] = {}
    concise_only = 0
    standard_only = 0
    for problem_id in sorted(concise):
        left = concise[problem_id]
        right = standard[problem_id]
        left_correct = bool(left["is_correct"])
        right_correct = bool(right["is_correct"])
        accuracy_effects[problem_id] = float(left_correct) - float(right_correct)
        length_effects[problem_id] = float(left["output_token_count"]) - float(
            right["output_token_count"]
        )
        concise_only += int(left_correct and not right_correct)
        standard_only += int(right_correct and not left_correct)
    accuracy = paired_cluster_bootstrap(
        accuracy_effects,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    length = paired_cluster_bootstrap(
        length_effects,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    concise_summary = summarize_opd_predictions(list(concise.values()))
    standard_summary = summarize_opd_predictions(list(standard.values()))
    if float(standard_summary["mean_output_tokens"]) <= 0.0:
        raise ValueError("Standard-arm mean output length must be positive for a ratio.")
    return {
        "n": len(concise),
        "accuracy_difference": accuracy["estimate"],
        "accuracy_difference_ci_low": accuracy["ci_low"],
        "accuracy_difference_ci_high": accuracy["ci_high"],
        "accuracy_bootstrap_p_value": accuracy["p_value"],
        "mcnemar_p_value": exact_mcnemar_p_value(concise_only, standard_only),
        "concise_only_correct": concise_only,
        "standard_only_correct": standard_only,
        "mean_output_token_difference": length["estimate"],
        "mean_output_token_difference_ci_low": length["ci_low"],
        "mean_output_token_difference_ci_high": length["ci_high"],
        "mean_output_token_difference_bootstrap_p_value": length["p_value"],
        "mean_output_token_ratio": (
            concise_summary["mean_output_tokens"] / standard_summary["mean_output_tokens"]
        ),
        "extraction_failure_rate_difference": (
            concise_summary["answer_extraction_failure_rate"]
            - standard_summary["answer_extraction_failure_rate"]
        ),
        "truncation_rate_difference": (
            concise_summary["truncation_rate"] - standard_summary["truncation_rate"]
        ),
    }


def opd_advancement_decision(
    contrast: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> Dict[str, Any]:
    """Apply the registered noninferiority-plus-shortening decision rule."""

    accuracy_pass = float(contrast["accuracy_difference_ci_low"]) >= -float(
        gate["accuracy_noninferiority_margin_pp"]
    ) / 100.0
    length_pass = float(contrast["mean_output_token_ratio"]) <= float(
        gate["maximum_mean_output_token_ratio"]
    )
    extraction_pass = float(contrast["extraction_failure_rate_difference"]) <= float(
        gate["maximum_extraction_failure_increase_pp"]
    ) / 100.0
    truncation_pass = float(contrast["truncation_rate_difference"]) <= float(
        gate["maximum_truncation_increase_pp"]
    ) / 100.0
    passed = accuracy_pass and length_pass and extraction_pass and truncation_pass
    if passed:
        classification = "bounded_concise_more_suitable"
    elif not accuracy_pass:
        classification = "standard_more_suitable_accuracy_failure"
    elif not length_pass:
        classification = "inconclusive_no_internalized_length_reduction"
    else:
        classification = "standard_more_suitable_output_failure"
    return {
        "status": "pass" if passed else "fail",
        "classification": classification,
        "accuracy_noninferiority_pass": accuracy_pass,
        "length_reduction_pass": length_pass,
        "extraction_failure_pass": extraction_pass,
        "truncation_pass": truncation_pass,
        "registered_gate": dict(gate),
    }
