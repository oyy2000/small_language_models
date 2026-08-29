#!/usr/bin/env python3
"""Train and publish one pure sampled-token OPD prompt arm."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256
from length_budget_distill.opd import (
    GATE_WAIVED_SMOKE_STAGE,
    GATE_WAIVED_TRAINING_STAGE,
    OPD_ARMS,
    protocol_hash,
    publish_opd_adapter,
    read_json,
    remove_runtime_after_publish,
    train_opd_arm,
    validate_gate_waiver,
    validate_opd_protocol,
    validated_temporary_runtime_path,
    validated_opd_adapter,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--arm", choices=OPD_ARMS, required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--gate-waiver-config", default=None)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-prompt-batches", type=int, default=None, help="Smoke-only training cap.")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    reference_manifest = _resolve(args.reference_manifest)
    preflight_dir = _resolve(args.preflight_dir)
    waiver = None
    waiver_evidence = None
    if args.gate_waiver_config:
        waiver_path = _resolve(args.gate_waiver_config)
        waiver = read_json(waiver_path)
        waiver_evidence = validate_gate_waiver(
            protocol,
            waiver,
            waiver_path=waiver_path,
            base_config_path=config_path,
        )
        if reference_manifest.resolve() != Path(
            waiver_evidence["reference_manifest_path"]
        ).resolve():
            raise ValueError("Gate-waived training reference manifest path mismatch.")
        if preflight_dir.resolve() != Path(
            waiver_evidence["failed_preflight_manifest_path"]
        ).resolve().parent:
            raise ValueError("Gate-waived training preflight directory mismatch.")
        checkpoint_root = _resolve(waiver["outputs"]["checkpoint_root"])
        result_root = _resolve(waiver["outputs"]["result_root"])
        stage = (
            GATE_WAIVED_SMOKE_STAGE
            if args.max_prompt_batches is not None
            else GATE_WAIVED_TRAINING_STAGE
        )
        runtime_experiment_name = str(waiver["continuation_name"])
    else:
        _validate_preflight(protocol, preflight_dir)
        checkpoint_root = _resolve(protocol["outputs"]["checkpoint_root"])
        result_root = _resolve(protocol["outputs"]["result_root"])
        stage = "smoke" if args.max_prompt_batches is not None else "pilot"
        runtime_experiment_name = str(protocol["experiment_name"])
    publish_dir = checkpoint_root / stage / args.arm
    existing = validated_opd_adapter(
        protocol,
        arm=args.arm,
        adapter_dir=publish_dir,
        stage=stage,
        required_evidence=waiver_evidence,
    )
    if existing is not None:
        if args.skip_complete:
            logging.info("opd_arm_already_complete arm=%s output=%s", args.arm, publish_dir)
            return
        raise FileExistsError(f"OPD adapter is already complete: {publish_dir}")
    if publish_dir.exists():
        raise FileExistsError(f"Incomplete OPD adapter exists; audit before retry: {publish_dir}")

    rollout_dir = result_root / stage / "training" / args.arm / "rollouts"
    arm_manifest_path = result_root / stage / "training" / args.arm / "training_manifest.json"
    if arm_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite OPD arm manifest: {arm_manifest_path}")
    if args.runtime_dir:
        runtime_dir = Path(args.runtime_dir)
    else:
        runtime_root = Path(os.environ.get("LBD_RUNTIME_CHECKPOINT_ROOT", "/var/tmp"))
        runtime_dir = runtime_root / runtime_experiment_name / stage / args.arm
    runtime_dir = validated_temporary_runtime_path(runtime_dir)
    if rollout_dir.exists() and not args.resume:
        raise FileExistsError(f"OPD rollout directory already exists: {rollout_dir}")

    metrics = train_opd_arm(
        protocol,
        arm=args.arm,
        reference_manifest_path=reference_manifest,
        runtime_dir=runtime_dir,
        rollout_dir=rollout_dir,
        resume=args.resume,
        max_prompt_batches=args.max_prompt_batches,
        logger=logging.getLogger(__name__),
    )
    source_paths = [
        PROJECT_ROOT / "src/length_budget_distill/opd.py",
        Path(__file__).resolve(),
    ]
    rollout_manifest = rollout_dir / "rollout_manifest.json"
    published = publish_opd_adapter(
        protocol,
        arm=args.arm,
        runtime_dir=runtime_dir,
        publish_dir=publish_dir,
        rollout_manifest_path=rollout_manifest,
        reference_manifest_path=reference_manifest,
        source_paths=source_paths,
        stage=stage,
        additional_evidence=waiver_evidence,
    )
    if waiver is not None:
        artifact_evidence_level = (
            "smoke_gate_waived"
            if args.max_prompt_batches is not None
            else str(waiver["evidence_level"])
        )
    else:
        artifact_evidence_level = (
            "smoke" if args.max_prompt_batches is not None else protocol["evidence_level"]
        )
    arm_manifest = {
        "status": "complete",
        "evidence_level": artifact_evidence_level,
        "stage": stage,
        "arm": args.arm,
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "reference_manifest_path": str(reference_manifest),
        "reference_manifest_sha256": file_sha256(reference_manifest),
        "preflight_manifest_path": str(preflight_dir / "preflight_manifest.json"),
        "preflight_manifest_sha256": file_sha256(preflight_dir / "preflight_manifest.json"),
        "adapter_path": str(publish_dir),
        "adapter_model_sha256": published["adapter_model_sha256"],
        "adapter_config_sha256": published["adapter_config_sha256"],
        "train_manifest_sha256": published["train_manifest_sha256"],
        "rollout_manifest_path": str(rollout_manifest),
        "rollout_manifest_sha256": file_sha256(rollout_manifest),
        "prompts": metrics["prompts"],
        "rollouts": metrics["rollouts"],
        "sampled_tokens": metrics["sampled_tokens"],
        "optimizer_steps": metrics["optimizer_steps"],
        "max_prompt_batches": args.max_prompt_batches,
        "gold_labels_used_in_loss": False,
        "length_used_in_loss": False,
    }
    if waiver_evidence is not None:
        arm_manifest["gate_waiver"] = waiver_evidence
    write_json(arm_manifest_path, arm_manifest)
    remove_runtime_after_publish(runtime_dir)
    logging.info(
        "opd_arm_complete arm=%s adapter=%s rollouts=%d sampled_tokens=%d",
        args.arm,
        publish_dir,
        metrics["rollouts"],
        metrics["sampled_tokens"],
    )


def _validate_preflight(protocol: dict, preflight_dir: Path) -> None:
    marker_path = preflight_dir / "PREFLIGHT_COMPLETE"
    manifest_path = preflight_dir / "preflight_manifest.json"
    summary_path = preflight_dir / "preflight_summary.json"
    if not marker_path.is_file() or not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Pure OPD training requires a complete preflight gate.")
    marker = read_json(marker_path)
    summary = read_json(summary_path)
    if marker.get("status") != "passed" or summary.get("status") != "passed":
        raise ValueError("OPD preflight did not pass.")
    if summary.get("finite_teacher_signal") is not True:
        raise ValueError("OPD preflight did not certify finite teacher signals.")
    if marker.get("protocol_hash") != protocol_hash(protocol):
        raise ValueError("OPD preflight protocol hash mismatch.")
    if marker.get("manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("OPD preflight manifest hash mismatch.")
    if marker.get("summary_sha256") != file_sha256(summary_path):
        raise ValueError("OPD preflight summary hash mismatch.")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
