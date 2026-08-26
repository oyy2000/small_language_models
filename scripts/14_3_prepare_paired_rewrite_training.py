#!/usr/bin/env python3
"""Freeze paired-rewrite SFT run configs for the grid or selected final stage."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/capacity_length_paired_rewrite_7b_pilot_v1.json",
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--stage", choices=["grid", "final"], default="grid")
    parser.add_argument("--selection-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    protocol = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    dataset_manifest_path = _resolve(args.dataset_manifest)
    dataset_manifest = _read_json(dataset_manifest_path)
    if dataset_manifest.get("status") != "complete":
        raise ValueError(f"Dataset manifest is incomplete: {dataset_manifest_path}")
    if dataset_manifest.get("config_hash") != config_hash:
        raise ValueError("Dataset manifest is bound to a different paired-rewrite config")
    by_condition = {str(row["condition"]): row for row in dataset_manifest.get("runs", [])}
    required = {"standard_original", "direct_short", "rewrite_80", "rewrite_65"}
    if set(by_condition) != required:
        raise ValueError(f"Expected datasets {sorted(required)}, got {sorted(by_condition)}")
    for condition, row in by_condition.items():
        path = _resolve(str(row["train_path"]))
        if file_sha256(path) != row.get("train_sha256"):
            raise ValueError(f"Training data hash mismatch for {condition}: {path}")

    if args.stage == "grid":
        entries = _grid_entries(config, by_condition, Path(args.checkpoint_root))
        selection_evidence = None
    else:
        if not args.selection_json:
            raise ValueError("--selection-json is required for --stage final")
        selection_path = _resolve(args.selection_json)
        selection = _read_json(selection_path)
        entries = _final_entries(config, by_condition, Path(args.checkpoint_root), selection)
        selection_evidence = {
            "path": str(selection_path),
            "sha256": file_sha256(selection_path),
        }

    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    manifest_path = output_dir / f"{args.stage}_training_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared manifest: {manifest_path}")
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: List[Dict[str, Any]] = []
    for entry in entries:
        run_config = _run_config(config, entry)
        config_path = config_dir / f"{entry['run_name']}.json"
        _write_json(config_path, run_config)
        manifest_entries.append(
            {
                **entry,
                "config_path": str(config_path),
                "run_config_sha256": file_sha256(config_path),
                "status": "prepared",
            }
        )
    manifest = {
        "status": "prepared",
        "stage": args.stage,
        "evidence_level": "exploratory_single_seed_pilot",
        "config_hash": config_hash,
        "config_path": str(_resolve(args.config)),
        "config_sha256": file_sha256(_resolve(args.config)),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "selection_evidence": selection_evidence,
        "run_count": len(manifest_entries),
        "runs": manifest_entries,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "run_count": len(entries)}, indent=2))


def _grid_entries(
    config: Mapping[str, Any],
    by_condition: Mapping[str, Mapping[str, Any]],
    checkpoint_root: Path,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    training = config["training"]
    for condition in config["selection"]["conditions"]:
        for learning_rate in training["learning_rates"]:
            lr_name = _learning_rate_name(float(learning_rate))
            run_name = f"grid__{condition}__{lr_name}"
            dataset = by_condition[str(condition)]
            entries.append(
                {
                    "run_name": run_name,
                    "stage": "grid",
                    "condition": str(condition),
                    "learning_rate": float(learning_rate),
                    "num_train_epochs": float(training["num_train_epochs"]),
                    "snapshot_epochs": [float(value) for value in training["snapshot_epochs"]],
                    "train_path": str(dataset["train_path"]),
                    "train_sha256": str(dataset["train_sha256"]),
                    "n": int(dataset["n"]),
                    "output_dir": str(checkpoint_root / "grid" / run_name),
                    "resume_adapter_path": None,
                }
            )
    return entries


def _final_entries(
    config: Mapping[str, Any],
    by_condition: Mapping[str, Mapping[str, Any]],
    checkpoint_root: Path,
    selection: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if selection.get("status") != "complete":
        raise ValueError("Recipe-selection artifact is incomplete")
    learning_rate = float(selection["selected_learning_rate"])
    epochs = float(selection["selected_epoch"])
    snapshots = selection.get("selected_snapshots", {})
    if not {"standard_original", "rewrite_65"} <= set(snapshots):
        raise ValueError("Selection must bind standard_original and rewrite_65 snapshots")
    stage2_epochs = 0.5
    stage2_lr = learning_rate / 2.0
    specs = [
        ("direct_short", "direct_short", learning_rate, epochs, None, "base_direct_short"),
        (
            "matched_rewrite_65_extra",
            "rewrite_65",
            stage2_lr,
            stage2_epochs,
            snapshots["rewrite_65"],
            "rewrite_65_then_rewrite_65",
        ),
        (
            "progressive_standard_to_rewrite_65",
            "rewrite_65",
            stage2_lr,
            stage2_epochs,
            snapshots["standard_original"],
            "standard_then_rewrite_65",
        ),
    ]
    entries: List[Dict[str, Any]] = []
    for run_name, condition, lr, run_epochs, resume, schedule in specs:
        dataset = by_condition[condition]
        resume_path = _resolve(str(resume)) if resume else None
        if resume_path is not None:
            _require_snapshot(resume_path)
        entries.append(
            {
                "run_name": run_name,
                "stage": "final",
                "condition": condition,
                "schedule": schedule,
                "learning_rate": lr,
                "num_train_epochs": run_epochs,
                "snapshot_epochs": [run_epochs],
                "train_path": str(dataset["train_path"]),
                "train_sha256": str(dataset["train_sha256"]),
                "n": int(dataset["n"]),
                "output_dir": str(checkpoint_root / "final" / run_name),
                "resume_adapter_path": str(resume_path) if resume_path else None,
            }
        )
    return entries


def _run_config(base: Mapping[str, Any], entry: Mapping[str, Any]) -> Dict[str, Any]:
    training = copy.deepcopy(base["training"])
    training.update(
        {
            "output_dir": str(entry["output_dir"]),
            "learning_rate": float(entry["learning_rate"]),
            "num_train_epochs": float(entry["num_train_epochs"]),
            "seed": int(base["training"]["seed"]),
            "data_seed": int(base["training"]["data_seed"]),
        }
    )
    student = copy.deepcopy(base["student"])
    if entry.get("resume_adapter_path"):
        student["resume_adapter_path"] = str(entry["resume_adapter_path"])
    return {
        "experiment_name": str(entry["run_name"]),
        "data": {
            "train_path": str(entry["train_path"]),
            "eval_path": None,
            "text_format": "prompt_completion",
        },
        "student": student,
        "training": training,
        "paired_rewrite_metadata": dict(entry),
    }


def _require_snapshot(path: Path) -> None:
    required = ["adapter_config.json", "adapter_model.safetensors", "SNAPSHOT_COMPLETE"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete selected snapshot {path}; missing={missing}")


def _learning_rate_name(value: float) -> str:
    return f"lr_{value:.0e}".replace("+", "").replace("-0", "-")


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
