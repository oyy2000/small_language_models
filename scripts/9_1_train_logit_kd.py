#!/usr/bin/env python3
"""Train one registered 7B-to-1.5B logit-KD adapter."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    file_sha256,
    kd_run_name,
    load_protocol,
    protocol_hash,
    publish_adapter,
    resolve_project_path,
    train_logit_kd_adapter,
    validated_training_marker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--stage", choices=["validation", "formal", "smoke"], required=True)
    parser.add_argument("--budget", choices=["short_128", "medium_256", "long_512"], required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--publish-dir", default=None)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = resolve_project_path(args.config)
    protocol = load_protocol(config_path)
    run_name = kd_run_name(args.budget, args.alpha, args.temperature)
    checkpoint_root = resolve_project_path(protocol["outputs"]["checkpoint_root"])
    if args.publish_dir:
        publish_dir = resolve_project_path(args.publish_dir)
    elif args.stage == "formal":
        publish_dir = checkpoint_root / "formal" / f"{args.budget}__seed_17"
    else:
        publish_dir = checkpoint_root / args.stage / run_name
    marker = validated_training_marker(publish_dir)
    if marker is not None:
        if args.skip_complete:
            logging.info("kd_training_already_complete output=%s", publish_dir)
            return
        raise FileExistsError(f"KD adapter is already complete: {publish_dir}")
    if publish_dir.exists():
        raise FileExistsError(f"Incomplete KD adapter output exists; audit before retrying: {publish_dir}")

    if args.stage == "validation":
        alpha_grid = {float(value) for value in protocol["kd"]["alpha_grid"]}
        temperature_grid = {float(value) for value in protocol["kd"]["temperature_grid"]}
        if args.alpha not in alpha_grid or args.temperature not in temperature_grid:
            raise ValueError("Validation KD parameters are outside the registered grid.")
    elif args.stage == "formal":
        selection_path = resolve_project_path(protocol["outputs"]["result_root"]) / "validation" / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(f"Formal training requires a frozen validation selection: {selection_path}")
        from length_budget_distill.logit_kd import read_json

        selection = read_json(selection_path)
        if selection.get("status") != "complete":
            raise ValueError("Validation selection is incomplete.")
        if float(selection["selected_alpha"]) != args.alpha or float(selection["selected_temperature"]) != args.temperature:
            raise ValueError("Formal KD parameters do not match the frozen validation selection.")

    if args.runtime_dir:
        runtime_dir = Path(args.runtime_dir)
    else:
        runtime_root = Path(os.environ.get("LBD_RUNTIME_CHECKPOINT_ROOT", "/var/tmp"))
        runtime_dir = runtime_root / protocol["experiment_name"] / args.stage / run_name
    if runtime_dir.exists():
        raise FileExistsError(f"Runtime output already exists: {runtime_dir}")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)

    metrics = train_logit_kd_adapter(
        protocol,
        budget_name=args.budget,
        alpha=args.alpha,
        temperature=args.temperature,
        runtime_output_dir=runtime_dir,
        logger=logging.getLogger(__name__),
    )
    source_files = [
        PROJECT_ROOT / "src/length_budget_distill/logit_kd.py",
        Path(__file__).resolve(),
    ]
    evidence = {
        "status": "complete",
        "stage": args.stage,
        "run_name": run_name,
        "budget_name": args.budget,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "seed": int(protocol["training"]["seed"]),
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "train_path": metrics["train_path"],
        "train_sha256": metrics["train_sha256"],
        "context_mode": metrics["context_mode"],
        "teacher_context_field": metrics["teacher_context_field"],
        "student_context_field": metrics["student_context_field"],
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_files},
    }
    published = publish_adapter(runtime_dir, publish_dir, evidence=evidence)
    shutil.rmtree(runtime_dir)
    logging.info(
        "kd_training_complete stage=%s run=%s output=%s adapter_sha256=%s",
        args.stage,
        run_name,
        publish_dir,
        published["adapter_model_sha256"],
    )


if __name__ == "__main__":
    main()
