#!/usr/bin/env python3
"""Select one shared KD hyperparameter pair using only the registered validation split."""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    file_sha256,
    kd_run_name,
    load_protocol,
    protocol_hash,
    read_json,
    read_jsonl,
    resolve_project_path,
    validated_training_marker,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    return parser.parse_args()


def _metrics(path: Path, budget_tokens: int) -> Dict[str, float]:
    rows = read_jsonl(path)
    lengths = [int(row["output_token_count"]) for row in rows]
    return {
        "n": float(len(rows)),
        "accuracy": sum(bool(row["is_correct"]) for row in rows) / len(rows),
        "mean_output_tokens": statistics.mean(lengths),
        "budget_compliance": sum(length <= budget_tokens for length in lengths) / len(lengths),
    }


def _eval_marker(eval_root: Path, run_name: str) -> Dict[str, Any]:
    path = eval_root / "markers" / f"{run_name}.json"
    marker = read_json(path)
    prediction = Path(marker["prediction_path"])
    if marker.get("status") != "complete" or marker.get("prediction_sha256") != file_sha256(prediction):
        raise ValueError(f"Validation evaluation evidence mismatch: {run_name}")
    return marker


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load_protocol(args.config)
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    validation_root = result_root / "validation"
    eval_root = validation_root / "eval"
    output_path = validation_root / "selection.json"
    marker_path = validation_root / "VALIDATION_COMPLETE"
    if output_path.exists() or marker_path.exists():
        raise FileExistsError(f"Validation selection already exists: {output_path}")

    sft_metrics: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []
    for budget_name, budget in protocol["budgets"].items():
        run_name = f"sft__{budget_name}__seed_17"
        marker = _eval_marker(eval_root, run_name)
        sft_metrics[budget_name] = _metrics(Path(marker["prediction_path"]), int(budget["max_solution_tokens"]))

    candidates: List[Dict[str, Any]] = []
    checkpoint_root = resolve_project_path(protocol["outputs"]["checkpoint_root"])
    for alpha in protocol["kd"]["alpha_grid"]:
        for temperature in protocol["kd"]["temperature_grid"]:
            per_budget = {}
            kd_training_values = []
            for budget_name, budget in protocol["budgets"].items():
                training_run = kd_run_name(budget_name, alpha, temperature)
                training_marker = validated_training_marker(checkpoint_root / "validation" / training_run)
                if training_marker is None:
                    raise ValueError(f"Validation KD adapter is incomplete: {training_run}")
                training_metrics = read_json(checkpoint_root / "validation" / training_run / "training_metrics.json")
                kd_training_values.append(float(training_metrics["mean_kd"]))
                eval_run = f"kd__{training_run}"
                eval_marker = _eval_marker(eval_root, eval_run)
                observed = _metrics(Path(eval_marker["prediction_path"]), int(budget["max_solution_tokens"]))
                baseline = sft_metrics[budget_name]
                observed["accuracy_delta_vs_sft"] = observed["accuracy"] - baseline["accuracy"]
                observed["compliance_delta_vs_sft"] = observed["budget_compliance"] - baseline["budget_compliance"]
                per_budget[budget_name] = observed
                rows.append(
                    {
                        "alpha": alpha,
                        "temperature": temperature,
                        "budget_name": budget_name,
                        **observed,
                        "sft_accuracy": baseline["accuracy"],
                        "sft_budget_compliance": baseline["budget_compliance"],
                    }
                )
            deltas = [values["accuracy_delta_vs_sft"] for values in per_budget.values()]
            feasible = all(values["compliance_delta_vs_sft"] >= 0 for values in per_budget.values())
            candidates.append(
                {
                    "alpha": float(alpha),
                    "temperature": float(temperature),
                    "budget_compliance_feasible": feasible,
                    "minimum_accuracy_delta": min(deltas),
                    "macro_accuracy_delta": statistics.mean(deltas),
                    "mean_training_kl": statistics.mean(kd_training_values),
                    "per_budget": per_budget,
                }
            )
    feasible_candidates = [candidate for candidate in candidates if candidate["budget_compliance_feasible"]]
    pool = feasible_candidates or candidates
    selected = max(
        pool,
        key=lambda candidate: (
            candidate["minimum_accuracy_delta"],
            candidate["macro_accuracy_delta"],
            -candidate["mean_training_kl"],
            -candidate["alpha"],
            -candidate["temperature"],
        ),
    )
    payload = {
        "status": "complete",
        "protocol_hash": protocol_hash(protocol),
        "selection_split": protocol["validation"],
        "selection_rule": [
            "restrict_to_candidates_with_non-decreasing_budget_compliance_in_all_budgets_when_any exist",
            "maximize_minimum_accuracy_delta",
            "maximize_macro_accuracy_delta",
            "minimize_mean_training_kl",
            "prefer_lower_alpha_then_lower_temperature",
        ],
        "budget_compliance_constraint_satisfied": bool(feasible_candidates),
        "source_sha256": {
            "scripts/10_2_select_logit_kd_hparams.py": file_sha256(Path(__file__).resolve())
        },
        "selected_alpha": selected["alpha"],
        "selected_temperature": selected["temperature"],
        "selected_candidate": selected,
        "sft_metrics": sft_metrics,
        "candidates": candidates,
    }
    write_json(output_path, payload)
    csv_path = validation_root / "validation_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        marker_path,
        {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "selection_sha256": file_sha256(output_path),
            "metrics_sha256": file_sha256(csv_path),
        },
    )
    logging.info(
        "validation_selection_complete alpha=%s temperature=%s feasible=%s",
        selected["alpha"],
        selected["temperature"],
        bool(feasible_candidates),
    )


if __name__ == "__main__":
    main()
