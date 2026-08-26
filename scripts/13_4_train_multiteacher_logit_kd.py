#!/usr/bin/env python3
"""Train one frozen matched-teacher logit-KD condition on mixed trajectories."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    file_sha256,
    protocol_hash,
    publish_adapter,
    read_json,
    resolve_project_path,
    train_logit_kd_adapter,
    validated_training_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--publish-dir", required=True)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol_path = resolve_project_path(args.protocol)
    frozen = read_json(protocol_path)
    condition_index = {
        str(condition["condition_id"]): condition for condition in frozen["conditions"]
    }
    if args.condition_id not in condition_index:
        raise ValueError(
            f"Unknown condition {args.condition_id!r}; available={sorted(condition_index)}"
        )
    condition = condition_index[args.condition_id]
    publish_dir = resolve_project_path(args.publish_dir)
    marker = validated_training_marker(publish_dir)
    if marker is not None:
        if args.skip_complete:
            if marker.get("frozen_protocol_sha256") != file_sha256(protocol_path):
                raise ValueError(f"Completed KD adapter protocol mismatch: {publish_dir}")
            logging.info("multiteacher_kd_already_complete output=%s", publish_dir)
            return
        raise FileExistsError(f"KD adapter is already complete: {publish_dir}")
    if publish_dir.exists():
        raise FileExistsError(f"Incomplete KD adapter output exists: {publish_dir}")

    view = _condition_protocol_view(frozen, condition)
    if args.runtime_dir:
        runtime_dir = Path(args.runtime_dir)
    else:
        runtime_root = Path(os.environ.get("LBD_RUNTIME_CHECKPOINT_ROOT", "/var/tmp"))
        runtime_dir = runtime_root / frozen["experiment_name"] / args.condition_id
    if runtime_dir.exists():
        raise FileExistsError(f"Runtime output already exists: {runtime_dir}")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_logit_kd_adapter(
        view,
        budget_name=args.condition_id,
        alpha=float(frozen["kd"]["alpha"]),
        temperature=float(frozen["kd"]["temperature"]),
        runtime_output_dir=runtime_dir,
        logger=logging.getLogger(__name__),
    )
    source_files = [
        PROJECT_ROOT / "src/length_budget_distill/logit_kd.py",
        PROJECT_ROOT / "src/length_budget_distill/multiteacher_kd.py",
        Path(__file__).resolve(),
    ]
    evidence = {
        "status": "complete",
        "stage": "pilot",
        "method": "logit_kd",
        "run_name": f"logit_kd__{args.condition_id}__seed_17",
        "condition_id": args.condition_id,
        "generator_name": condition["generator_name"],
        "budget_name": condition["budget_name"],
        "teacher": condition["teacher"],
        "alpha": float(frozen["kd"]["alpha"]),
        "temperature": float(frozen["kd"]["temperature"]),
        "seed": int(frozen["training"]["seed"]),
        "frozen_protocol_path": str(protocol_path),
        "frozen_protocol_sha256": file_sha256(protocol_path),
        "condition_protocol_hash": protocol_hash(view),
        "train_path": metrics["train_path"],
        "train_sha256": metrics["train_sha256"],
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_files
        },
    }
    published = publish_adapter(runtime_dir, publish_dir, evidence=evidence)
    shutil.rmtree(runtime_dir)
    logging.info(
        "multiteacher_kd_complete condition=%s output=%s adapter_sha256=%s",
        args.condition_id,
        publish_dir,
        published["adapter_model_sha256"],
    )


def _condition_protocol_view(
    frozen: Dict[str, Any],
    condition: Dict[str, Any],
) -> Dict[str, Any]:
    condition_id = str(condition["condition_id"])
    budget = {
        "budget_name": str(condition["budget_name"]),
        "max_solution_tokens": int(condition["max_solution_tokens"]),
        "train_path": str(condition["train_path"]),
        "train_sha256": str(condition["train_sha256"]),
        "expected_records": int(condition["expected_records"]),
        "expected_solution_tokens": int(condition["expected_solution_tokens"]),
        "expected_generator_name": str(condition["generator_name"]),
        "expected_dataset_sources": list(condition["expected_dataset_sources"]),
        "expected_source_counts": dict(condition["expected_source_counts"]),
        "expected_source_solution_tokens": dict(
            condition["expected_source_solution_tokens"]
        ),
    }
    return {
        "experiment_name": str(frozen["experiment_name"]),
        "protocol_variant": str(frozen["protocol_variant"]),
        "supervision": {"mode": "equal_token"},
        "parent": copy.deepcopy(frozen["dataset_manifest"]),
        "models": {
            "teacher": copy.deepcopy(condition["teacher"]),
            "student": copy.deepcopy(frozen["models"]["student"]),
            "tokenizer": copy.deepcopy(frozen["models"]["tokenizer"]),
        },
        "budgets": {condition_id: budget},
        "kd": {
            "completion_only": True,
            "loss_direction": str(frozen["kd"]["loss_direction"]),
        },
        "training": copy.deepcopy(frozen["training"]),
        "validation": {},
        "formal": {},
        "outputs": copy.deepcopy(frozen["outputs"]),
    }


if __name__ == "__main__":
    main()
