#!/usr/bin/env python3
"""Validate ranked-length generation evidence and prepare three SFT runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    nonempty_line_count,
    read_key_value_marker,
)


LABELS = ("short", "medium", "long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-config",
        default="configs/capacity_length_ranked_sampling_7b_training_seed17_v1.json",
    )
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config_path = _resolve(args.training_config)
    training_config = load_config(str(training_config_path))
    protocol = {key: value for key, value in training_config.items() if key != "_config_path"}
    training_config_hash = canonical_sha256(protocol)
    parent = dict(training_config["parent_generation"])

    generation_config_path = _resolve(str(parent["config_path"]))
    generation_config = load_config(str(generation_config_path))
    generation_protocol = {
        key: value for key, value in generation_config.items() if key != "_config_path"
    }
    generation_config_hash = canonical_sha256(generation_protocol)
    _require_hash(
        "generation canonical config",
        generation_config_hash,
        str(parent["canonical_config_sha256"]),
    )
    _require_hash(
        "generation config file",
        file_sha256(generation_config_path),
        str(parent["config_file_sha256"]),
    )

    source_manifest_path = _resolve(str(parent["dataset_manifest_path"]))
    completion_marker_path = _resolve(str(parent["completion_marker_path"]))
    _require_hash(
        "generation dataset manifest",
        file_sha256(source_manifest_path),
        str(parent["dataset_manifest_sha256"]),
    )
    _require_hash(
        "generation completion marker",
        file_sha256(completion_marker_path),
        str(parent["completion_marker_sha256"]),
    )
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("status") != "complete":
        raise ValueError(f"Generation dataset manifest is incomplete: {source_manifest_path}")
    if source_manifest.get("config_hash") != generation_config_hash:
        raise ValueError("Generation dataset manifest is bound to a different config.")
    marker = read_key_value_marker(completion_marker_path)
    if marker.get("config_hash") != generation_config_hash:
        raise ValueError("Generation completion marker is bound to a different config.")
    if marker.get("dataset_manifest_sha256") != file_sha256(source_manifest_path):
        raise ValueError("Generation completion marker does not bind the dataset manifest.")

    scope = dict(training_config["training_scope"])
    if tuple(scope.get("labels", [])) != LABELS:
        raise ValueError(f"Training labels must equal {LABELS}.")
    seeds = [int(seed) for seed in scope.get("training_seeds", [])]
    if seeds != [17]:
        raise ValueError("Ranked-length training v1 is locked to seed 17.")
    expected_examples = int(scope["expected_examples_per_run"])
    source_by_label = {str(row["label"]): row for row in source_manifest.get("datasets", [])}
    if set(source_by_label) != set(LABELS):
        raise ValueError(f"Expected source datasets {LABELS}, got {sorted(source_by_label)}")

    runs = []
    for label in LABELS:
        source = source_by_label[label]
        train_path = _resolve(str(source["train_path"]))
        if int(source.get("record_count", -1)) != expected_examples:
            raise ValueError(f"Unexpected training examples for {label}: {source.get('record_count')}")
        if nonempty_line_count(train_path) != expected_examples:
            raise ValueError(f"Training JSONL cardinality mismatch for {label}: {train_path}")
        train_sha256 = file_sha256(train_path)
        _require_hash(f"{label} training data", train_sha256, str(source["train_sha256"]))
        runs.append(
            {
                "run_name": f"equal_example__qwen2p5_7b__relative_{label}__seed_17",
                "mode": "equal_example",
                "generator_name": "qwen2p5_7b",
                "budget_name": f"relative_{label}",
                "seed": 17,
                "train_path": str(train_path),
                "train_sha256": train_sha256,
                "n": expected_examples,
                "supervised_tokens": int(source["supervised_tokens"]),
            }
        )

    if len(runs) != int(scope["run_count"]):
        raise ValueError(f"Run count mismatch: expected={scope['run_count']} actual={len(runs)}")
    manifest = {
        "status": "complete",
        "experiment_name": training_config["experiment_name"],
        "protocol_variant": training_config["protocol_variant"],
        "config_path": str(training_config_path),
        "config_hash": training_config_hash,
        "config_file_sha256": file_sha256(training_config_path),
        "generation_config_path": str(generation_config_path),
        "generation_config_hash": generation_config_hash,
        "generation_dataset_manifest": str(source_manifest_path),
        "generation_dataset_manifest_sha256": file_sha256(source_manifest_path),
        "generation_completion_marker": str(completion_marker_path),
        "generation_completion_marker_sha256": file_sha256(completion_marker_path),
        "run_count": len(runs),
        "runs": runs,
    }

    output_path = Path(args.output_manifest)
    if output_path.exists():
        if not args.skip_existing:
            raise FileExistsError(f"Refusing to overwrite prepared training manifest: {output_path}")
        existing = _read_json(output_path)
        if existing != manifest:
            raise ValueError(f"Existing prepared manifest differs from current evidence: {output_path}")
        print(json.dumps({"status": "reused_complete", "manifest": str(output_path)}, indent=2))
        return
    _write_json(output_path, manifest)
    print(json.dumps({"status": "complete", "manifest": str(output_path), "run_count": len(runs)}, indent=2))


def _require_hash(label: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected={expected} actual={actual}")


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
