#!/usr/bin/env python3
"""Audit the exploratory paired-rewrite pilot and write its completion marker."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256, nonempty_line_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_paired_rewrite_7b_pilot_v1.json")
    parser.add_argument("--output-root", default="results/capacity_length_paired_rewrite_7b_pilot_v1")
    parser.add_argument("--checkpoint-root", default="checkpoints/capacity_length_paired_rewrite_7b_pilot_v1")
    parser.add_argument("--figure-root", default="figures/capacity_length_paired_rewrite_7b_pilot_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    figure_root = Path(args.figure_root)
    errors: List[str] = []
    evidence: Dict[str, Any] = {
        "config_hash": config_hash,
        "config_sha256": file_sha256(_resolve(args.config)),
        "evidence_level": "exploratory_single_seed_pilot",
        "scope": "GSM8K only",
    }

    raw_manifests = [_read_json(Path(path)) for path in sorted(glob.glob(str(output_root / "pilot/raw/manifests/shard_*.json")))]
    expected_candidates = (
        int(config["paired_rewrite"]["expected_records"])
        * len(config["paired_rewrite"]["ratios"])
        * int(config["paired_rewrite"]["num_candidates"])
    )
    observed_candidates = 0
    if not raw_manifests:
        errors.append("no rewrite shard manifests")
    for manifest in raw_manifests:
        raw_path = Path(str(manifest.get("raw_path", "")))
        if manifest.get("status") != "complete" or manifest.get("config_hash") != config_hash:
            errors.append(f"invalid rewrite shard manifest: {raw_path}")
            continue
        if not raw_path.is_file() or manifest.get("raw_sha256") != file_sha256(raw_path):
            errors.append(f"rewrite shard file/hash mismatch: {raw_path}")
        observed_candidates += int(manifest.get("candidate_count", 0))
    if observed_candidates != expected_candidates:
        errors.append(f"rewrite candidate count {observed_candidates} != {expected_candidates}")
    evidence["rewrite_generation"] = {
        "shard_count": len(raw_manifests),
        "candidate_count": observed_candidates,
        "expected_candidate_count": expected_candidates,
    }

    dataset_path = output_root / "pilot/sft_data/dataset_manifest.json"
    if not dataset_path.is_file():
        errors.append("paired SFT dataset manifest missing")
    else:
        dataset = _read_json(dataset_path)
        expected_n = int(config["paired_rewrite"]["expected_records"])
        if dataset.get("status") != "complete" or dataset.get("config_hash") != config_hash:
            errors.append("paired SFT dataset manifest invalid")
        for row in dataset.get("runs", []):
            path = Path(str(row["train_path"]))
            if not path.is_file() or file_sha256(path) != row.get("train_sha256"):
                errors.append(f"SFT dataset hash mismatch: {row.get('condition')}")
            elif nonempty_line_count(path) != expected_n:
                errors.append(f"SFT dataset row count mismatch: {row.get('condition')}")
        evidence["dataset_manifest_sha256"] = file_sha256(dataset_path)

    grid = _audit_training(checkpoint_root / "grid", expected_count=9, errors=errors)
    final = _audit_training(checkpoint_root / "final", expected_count=3, errors=errors)
    evidence["training"] = {"grid": grid, "final": final}

    selection_eval = _audit_eval(
        output_root / "pilot/eval/selection",
        expected_tasks=27,
        errors=errors,
    )
    confirmatory_model_count = len(config["selection"]["conditions"]) + 3
    expected_confirmatory = confirmatory_model_count * (len(config["evaluation"]["budget_sweep"]) + 1)
    confirmatory_eval = _audit_eval(
        output_root / "pilot/eval/confirmatory",
        expected_tasks=expected_confirmatory,
        errors=errors,
    )
    evidence["evaluation"] = {
        "selection": selection_eval,
        "confirmatory": confirmatory_eval,
    }

    selection_path = output_root / "pilot/analysis/selection/recipe_selection.json"
    final_summary_path = output_root / "pilot/analysis/final/paired_rewrite_pilot_summary.json"
    for label, path in (("recipe selection", selection_path), ("final summary", final_summary_path)):
        if not path.is_file() or _read_json(path).get("status") != "complete":
            errors.append(f"{label} is missing or incomplete")
        else:
            evidence[f"{label.replace(' ', '_')}_sha256"] = file_sha256(path)
    expected_figures = [
        figure_root / "selection/paired_training_data_lengths.png",
        figure_root / "selection/paired_training_data_lengths.pdf",
        figure_root / "selection/paired_rewrite_recipe_selection.png",
        figure_root / "selection/paired_rewrite_recipe_selection.pdf",
        figure_root / "selection/paired_rewrite_training_diagnostics.png",
        figure_root / "selection/paired_rewrite_training_diagnostics.pdf",
        figure_root / "final/paired_rewrite_budget_accuracy.png",
        figure_root / "final/paired_rewrite_budget_accuracy.pdf",
        figure_root / "final/paired_rewrite_accuracy_length_frontier.png",
        figure_root / "final/paired_rewrite_accuracy_length_frontier.pdf",
    ]
    for path in expected_figures:
        if not path.is_file():
            errors.append(f"missing figure: {path}")
    evidence["figures"] = [
        {"path": str(path), "sha256": file_sha256(path)} for path in expected_figures if path.is_file()
    ]

    report = {
        "status": "complete" if not errors else "failed",
        "evidence_level": "exploratory_single_seed_pilot",
        "formal_claim_allowed": False,
        "errors": errors,
        "evidence": evidence,
    }
    report_path = output_root / "pilot/audit_report.json"
    _write_json(report_path, report)
    if errors:
        raise SystemExit(f"Paired-rewrite audit failed with {len(errors)} errors; report={report_path}")
    marker = output_root / "pilot/PILOT_COMPLETE"
    marker.write_text(
        f"config_hash={config_hash}\nreport_sha256={file_sha256(report_path)}\n"
        "evidence_level=exploratory_single_seed_pilot\nformal_claim_allowed=false\n",
        encoding="utf-8",
    )
    print(f"paired_rewrite_pilot_complete marker={marker}")


def _audit_training(root: Path, *, expected_count: int, errors: List[str]) -> Dict[str, Any]:
    paths = sorted(root.glob("*/training_manifest.json"))
    if len(paths) != expected_count:
        errors.append(f"training manifest count under {root}: {len(paths)} != {expected_count}")
    snapshot_count = 0
    for path in paths:
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            errors.append(f"incomplete training manifest: {path}")
        for snapshot in manifest.get("snapshots", []):
            snapshot_path = Path(str(snapshot["path"]))
            model_path = snapshot_path / "adapter_model.safetensors"
            config_path = snapshot_path / "adapter_config.json"
            if not model_path.is_file() or file_sha256(model_path) != snapshot.get("adapter_model_sha256"):
                errors.append(f"adapter model hash mismatch: {snapshot_path}")
            if not config_path.is_file() or file_sha256(config_path) != snapshot.get("adapter_config_sha256"):
                errors.append(f"adapter config hash mismatch: {snapshot_path}")
            snapshot_count += 1
    return {"manifest_count": len(paths), "snapshot_count": snapshot_count}


def _audit_eval(root: Path, *, expected_tasks: int, errors: List[str]) -> Dict[str, Any]:
    manifests = sorted(root.glob("eval_manifest_*.json"))
    tasks: List[Mapping[str, Any]] = []
    for path in manifests:
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            errors.append(f"incomplete evaluation manifest: {path}")
        tasks.extend(manifest.get("tasks", []))
    if len(tasks) != expected_tasks:
        errors.append(f"evaluation task count under {root}: {len(tasks)} != {expected_tasks}")
    for task in tasks:
        summary_path = Path(str(task["summary_path"]))
        prediction_path = Path(str(task["prediction_path"]))
        if not summary_path.is_file() or not prediction_path.is_file():
            errors.append(f"missing evaluation artifacts: {task.get('task_id')}")
            continue
        summary = _read_json(summary_path)
        if summary.get("prediction_sha256") != file_sha256(prediction_path):
            errors.append(f"evaluation prediction hash mismatch: {task.get('task_id')}")
    return {"manifest_count": len(manifests), "task_count": len(tasks)}


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
