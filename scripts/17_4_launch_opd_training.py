#!/usr/bin/env python3
"""Launch the two independent OPD prompt arms on separate GPUs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256, runtime_metadata
from length_budget_distill.opd import (
    GATE_WAIVED_SMOKE_STAGE,
    GATE_WAIVED_TRAINING_STAGE,
    OPD_ARMS,
    protocol_hash,
    read_json,
    validate_gate_waiver,
    validate_opd_protocol,
    validated_opd_adapter,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-waiver-config", default=None)
    parser.add_argument("--max-prompt-batches", type=int, default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
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
        stage = (
            GATE_WAIVED_SMOKE_STAGE
            if args.max_prompt_batches is not None
            else GATE_WAIVED_TRAINING_STAGE
        )
        result_root = _resolve(waiver["outputs"]["result_root"])
        checkpoint_root = _resolve(waiver["outputs"]["checkpoint_root"])
        expected_output_dir = result_root / stage / "training"
        if _resolve(args.output_dir) != expected_output_dir:
            raise ValueError(
                "Gate-waived launcher output must use its isolated result root: "
                f"expected={expected_output_dir}"
            )
    else:
        if args.max_prompt_batches is not None:
            raise ValueError("The v1 launcher does not expose smoke caps without a gate waiver.")
        stage = "pilot"
        result_root = _resolve(protocol["outputs"]["result_root"])
        checkpoint_root = _resolve(protocol["outputs"]["checkpoint_root"])
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) < len(OPD_ARMS):
        raise ValueError(f"OPD training requires at least {len(OPD_ARMS)} GPUs.")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = (
        "training_launcher_manifest_dry_run.json"
        if args.dry_run
        else "training_launcher_manifest.json"
    )
    manifest_path = output_dir / manifest_name
    if manifest_path.exists() and not args.skip_complete:
        raise FileExistsError(f"Refusing to overwrite training launcher manifest: {manifest_path}")

    tasks: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    launch_tag = f"{int(time.time())}_{os.getpid()}"
    for arm, gpu_id in zip(OPD_ARMS, gpu_ids):
        log_path = output_dir / "logs" / f"{arm}_{launch_tag}.log"
        task = {
            "arm": arm,
            "gpu_id": gpu_id,
            "log_path": str(log_path),
            "status": "prepared",
        }
        command = [
            sys.executable,
            "scripts/17_4_train_opd_policy.py",
            "--config",
            str(config_path),
            "--arm",
            arm,
            "--reference-manifest",
            str(_resolve(args.reference_manifest)),
            "--preflight-dir",
            str(_resolve(args.preflight_dir)),
        ]
        if args.skip_complete:
            command.append("--skip-complete")
        if args.resume:
            command.append("--resume")
        if args.max_prompt_batches is not None:
            command.extend(["--max-prompt-batches", str(args.max_prompt_batches)])
        if args.gate_waiver_config:
            command.extend(["--gate-waiver-config", str(_resolve(args.gate_waiver_config))])
        tasks.append(task)
        commands.append(command)
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "stage": stage,
        "evidence_level": (
            waiver["evidence_level"] if waiver is not None else protocol["evidence_level"]
        ),
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "reference_manifest_path": str(_resolve(args.reference_manifest)),
        "preflight_dir": str(_resolve(args.preflight_dir)),
        "training_source_sha256": file_sha256(
            PROJECT_ROOT / "scripts/17_4_train_opd_policy.py"
        ),
        "launcher_source_sha256": file_sha256(Path(__file__).resolve()),
        "opd_library_sha256": file_sha256(
            PROJECT_ROOT / "src/length_budget_distill/opd.py"
        ),
        "tasks": tasks,
        "max_prompt_batches": args.max_prompt_batches,
        "gate_waiver": waiver_evidence,
        "runtime": runtime_metadata(),
    }
    write_json(manifest_path, manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    for task, command in zip(tasks, commands):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(task["gpu_id"])
        handle = Path(task["log_path"]).open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        task.update({"status": "running", "pid": process.pid})
        running.append((process, task, handle))
        logging.info("launched_opd_arm arm=%s gpu=%s pid=%d", task["arm"], task["gpu_id"], process.pid)
    write_json(manifest_path, manifest)

    failures = []
    while running:
        still_running = []
        for process, task, handle in running:
            returncode = process.poll()
            if returncode is None:
                still_running.append((process, task, handle))
                continue
            handle.close()
            task["returncode"] = returncode
            evidence = (
                _completed_arm(
                    protocol,
                    str(task["arm"]),
                    checkpoint_root=checkpoint_root,
                    result_root=result_root,
                    stage=stage,
                    required_evidence=waiver_evidence,
                )
                if returncode == 0
                else None
            )
            task["status"] = "complete" if evidence is not None else "failed"
            if evidence is None:
                failures.append(dict(task))
            else:
                task.update(evidence)
            logging.info("finished_opd_arm arm=%s status=%s", task["arm"], task["status"])
            write_json(manifest_path, manifest)
        running = still_running
        if running:
            time.sleep(5)
    manifest["status"] = "failed" if failures else "complete"
    write_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"OPD training arm failures={len(failures)}; manifest={manifest_path}")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _completed_arm(
    protocol: Dict[str, Any],
    arm: str,
    *,
    checkpoint_root: Path,
    result_root: Path,
    stage: str,
    required_evidence: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    adapter_dir = checkpoint_root / stage / arm
    marker = validated_opd_adapter(
        protocol,
        arm=arm,
        adapter_dir=adapter_dir,
        stage=stage,
        required_evidence=required_evidence,
    )
    arm_manifest_path = result_root / stage / "training" / arm / "training_manifest.json"
    if marker is None or not arm_manifest_path.is_file():
        return None
    arm_manifest = read_json(arm_manifest_path)
    checks = {
        "status": "complete",
        "stage": stage,
        "arm": arm,
        "protocol_hash": protocol_hash(protocol),
        "adapter_model_sha256": marker["adapter_model_sha256"],
        "adapter_config_sha256": marker["adapter_config_sha256"],
        "train_manifest_sha256": marker["train_manifest_sha256"],
    }
    if any(arm_manifest.get(key) != value for key, value in checks.items()):
        return None
    return {
        "adapter_path": str(adapter_dir),
        "adapter_train_manifest_sha256": marker["train_manifest_sha256"],
        "arm_manifest_path": str(arm_manifest_path),
        "arm_manifest_sha256": file_sha256(arm_manifest_path),
        "rollouts": int(arm_manifest["rollouts"]),
        "sampled_tokens": int(arm_manifest["sampled_tokens"]),
    }


if __name__ == "__main__":
    main()
