#!/usr/bin/env python3
"""Prepare and launch fixed-config capacity-length student SFT runs."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    canonical_sha256,
    declared_launcher_wave_groups,
    file_sha256,
    runtime_metadata,
    select_launcher_shard_runs,
    validated_adapter_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_factorial_v1.json")
    parser.add_argument(
        "--training-config",
        default="configs/capacity_length_factorial_sft_v1.json",
        help="Stage-specific SFT overrides bound to the parent protocol hash.",
    )
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint-root", default="checkpoints/capacity_length_factorial_v1")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated local GPU IDs.")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--launcher-shards", type=int, default=1)
    parser.add_argument("--launcher-shard-index", type=int, default=0)
    parser.add_argument(
        "--modes",
        default="equal_example,equal_token,calibration",
        help="Comma-separated data modes to run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.launcher_shards <= 0 or not 0 <= args.launcher_shard_index < args.launcher_shards:
        raise ValueError("Invalid launcher shard topology.")
    config = load_config(args.config)
    config.pop("_config_path", None)
    config_hash = canonical_sha256(config)
    training_overlay = _read_json(Path(args.training_config))
    if training_overlay.get("parent_config_sha256") != config_hash:
        raise ValueError(
            "Training overlay parent hash mismatch: "
            f"overlay={training_overlay.get('parent_config_sha256')} config={config_hash}"
        )
    training_overrides = dict(training_overlay.get("training_overrides", {}))
    if not training_overrides:
        raise ValueError("Training overlay must define non-empty training_overrides.")
    training_overlay_sha256 = file_sha256(args.training_config)
    dataset_manifest = _read_json(Path(args.dataset_manifest))
    if dataset_manifest.get("status") != "complete":
        raise ValueError(f"Dataset manifest is not complete: {args.dataset_manifest}")
    if dataset_manifest.get("config_hash") != config_hash:
        raise ValueError(
            f"Dataset/config hash mismatch: dataset={dataset_manifest.get('config_hash')} config={config_hash}"
        )
    requested_modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    valid_modes = {"equal_example", "equal_token", "calibration"}
    if not requested_modes or not requested_modes <= valid_modes:
        raise ValueError(f"--modes must be a non-empty subset of {sorted(valid_modes)}")

    all_runs = [run for run in dataset_manifest.get("runs", []) if run.get("mode") in requested_modes]
    indexed_runs = select_launcher_shard_runs(
        all_runs,
        launcher_shards=args.launcher_shards,
        launcher_shard_index=args.launcher_shard_index,
    )
    if not indexed_runs:
        raise ValueError("No training runs were assigned to this launcher shard.")
    declared_waves = declared_launcher_wave_groups(indexed_runs)
    wave_barrier_policy = (
        "declared_launcher_wave_barrier_v1" if declared_waves else "legacy_continuous_queue"
    )

    work_dir = Path(args.work_dir)
    config_dir = work_dir / "configs"
    log_dir = work_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    training_source_sha256 = file_sha256(PROJECT_ROOT / "src/length_budget_distill/training.py")
    launcher_source_sha256 = file_sha256(Path(__file__).resolve())
    for run in indexed_runs:
        run_name = str(run["run_name"])
        train_path = Path(str(run["train_path"]))
        resolved_train_path = train_path if train_path.is_absolute() else PROJECT_ROOT / train_path
        if not resolved_train_path.is_file():
            raise FileNotFoundError(f"Missing SFT dataset for {run_name}: {resolved_train_path}")
        actual_train_sha256 = file_sha256(resolved_train_path)
        if run.get("train_sha256") != actual_train_sha256:
            raise ValueError(
                f"SFT dataset hash mismatch for {run_name}: "
                f"manifest={run.get('train_sha256')} actual={actual_train_sha256}"
            )
        run_config = _training_config(config, run, args.checkpoint_root, training_overrides)
        config_path = config_dir / f"{run_name}.json"
        _write_json(config_path, run_config)
        run_config_sha256 = file_sha256(config_path)
        output_dir = Path(run_config["training"]["output_dir"])
        marker_path = output_dir / "TRAIN_COMPLETE"
        entries.append(
            {
                **run,
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "log_path": str(log_dir / f"{run_name}.log"),
                "marker_path": str(marker_path),
                "train_sha256": actual_train_sha256,
                "run_config_sha256": run_config_sha256,
                "training_source_sha256": training_source_sha256,
                "launcher_source_sha256": launcher_source_sha256,
                "status": "prepared",
                "overrides": {
                    "mode": run.get("mode"),
                    "generator_name": run.get("generator_name"),
                    "budget_name": run.get("budget_name"),
                    "baseline_name": run.get("baseline_name"),
                    "seed": run.get("seed"),
                    "n": run.get("n"),
                    "supervised_tokens": run.get("supervised_tokens"),
                },
            }
        )

    manifest_path = work_dir / (
        f"training_manifest_shard_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    )
    manifest = {
        "status": "prepared" if args.dry_run else "running",
        "config_hash": config_hash,
        "dataset_manifest": args.dataset_manifest,
        "training_overlay": args.training_config,
        "training_overlay_sha256": training_overlay_sha256,
        "launcher_shard_index": args.launcher_shard_index,
        "launcher_shards": args.launcher_shards,
        "wave_barrier_policy": wave_barrier_policy,
        "launcher_wave_count": len(declared_waves),
        "run_count": len(entries),
        "runtime": runtime_metadata(),
        "runs": entries,
    }
    _write_json(manifest_path, manifest)
    commands = [
        [sys.executable, "scripts/2_1_train_student_sft.py", "--config", entry["config_path"]]
        for entry in entries
    ]
    logging.info("prepared_runs=%d manifest=%s", len(entries), manifest_path)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive.")
    failures = _run_commands(
        entries,
        commands,
        gpu_ids,
        max_parallel,
        args.skip_complete,
        wave_barrier=bool(declared_waves),
    )
    manifest["status"] = "failed" if failures else "complete"
    manifest["runs"] = entries
    _write_json(manifest_path, manifest)
    if failures:
        for entry in failures:
            logging.error("failed run=%s log=%s", entry["run_name"], entry["log_path"])
        raise SystemExit(f"Training failures={len(failures)}; manifest={manifest_path}")
    logging.info("training_shard_complete runs=%d manifest=%s", len(entries), manifest_path)


def _training_config(
    base: Dict[str, Any],
    run: Dict[str, Any],
    checkpoint_root: str,
    training_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    run_config = {
        "experiment_name": run["run_name"],
        "data": {
            "train_path": run["train_path"],
            "eval_path": None,
            "text_format": "prompt_completion",
        },
        "student": copy.deepcopy(base["student"]),
        "training": copy.deepcopy(base["training"]),
        "factorial_metadata": copy.deepcopy(run),
    }
    seed = int(run["seed"])
    run_config["training"].update(copy.deepcopy(training_overrides))
    run_config["training"]["seed"] = seed
    run_config["training"]["data_seed"] = seed
    run_config["training"]["output_dir"] = str(Path(checkpoint_root) / str(run["run_name"]))
    return run_config


def _run_commands(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
    skip_complete: bool,
    *,
    wave_barrier: bool = False,
) -> List[Dict[str, Any]]:
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)
    while pending or running:
        active_wave = None
        if wave_barrier:
            if running:
                running_waves = {int(entry["launcher_wave_index"]) for _, entry, _ in running}
                if len(running_waves) != 1:
                    raise ValueError(f"Multiple launcher waves are running concurrently: {running_waves}")
                active_wave = next(iter(running_waves))
            elif pending:
                active_wave = int(pending[0][0]["launcher_wave_index"])
        while pending and len(running) < max_parallel and (not gpu_ids or available_gpus):
            if wave_barrier and int(pending[0][0]["launcher_wave_index"]) != active_wave:
                break
            entry, command = pending.pop(0)
            output_dir = Path(entry["output_dir"])
            evidence = validated_adapter_evidence(output_dir)
            if evidence is not None:
                if skip_complete:
                    if evidence.get("train_sha256") != entry["train_sha256"]:
                        raise ValueError(f"Completed adapter train hash mismatch: {entry['run_name']}")
                    if evidence.get("run_config_sha256") != entry["run_config_sha256"]:
                        raise ValueError(f"Completed adapter config hash mismatch: {entry['run_name']}")
                    if evidence.get("training_source_sha256") != entry["training_source_sha256"]:
                        raise ValueError(f"Completed adapter training source mismatch: {entry['run_name']}")
                    if evidence.get("launcher_source_sha256") != entry["launcher_source_sha256"]:
                        raise ValueError(f"Completed adapter launcher source mismatch: {entry['run_name']}")
                    entry.update(evidence)
                    entry["status"] = "skipped_complete"
                    logging.info("skip_complete run=%s", entry["run_name"])
                    continue
                raise FileExistsError(f"Complete training output already exists: {output_dir}")
            if output_dir.exists():
                if any(output_dir.iterdir()):
                    raise FileExistsError(
                        f"Incomplete non-empty output directory exists for {entry['run_name']}: "
                        f"{output_dir}; audit it before rerun."
                    )
                entry["reused_empty_output_dir"] = True
            else:
                _mkdir_with_retry(output_dir)
            runtime_output_dir = _runtime_checkpoint_root() / entry["run_name"]
            if runtime_output_dir.exists() and any(runtime_output_dir.iterdir()):
                raise FileExistsError(
                    f"Node-local runtime output is unexpectedly non-empty: {runtime_output_dir}"
                )
            _mkdir_with_retry(runtime_output_dir)
            entry["runtime_output_dir"] = str(runtime_output_dir)
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            env = os.environ.copy()
            env["LBD_RUNTIME_OUTPUT_DIR"] = str(runtime_output_dir)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                entry["gpu_id"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            entry["status"] = "running"
            logging.info("launch run=%s gpu=%s", entry["run_name"], gpu_id or "default")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((process, entry, log_handle))

        still_running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
        for process, entry, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            gpu_id = entry.get("gpu_id")
            if gpu_id is not None:
                available_gpus.append(str(gpu_id))
            output_dir = Path(entry["output_dir"])
            runtime_output_dir = Path(entry["runtime_output_dir"])
            if return_code == 0 and _adapter_files_exist(runtime_output_dir):
                _publish_adapter_with_retry(runtime_output_dir, output_dir)
                adapter_config_sha256 = file_sha256(output_dir / "adapter_config.json")
                adapter_model_sha256 = file_sha256(output_dir / "adapter_model.safetensors")
                marker = output_dir / "TRAIN_COMPLETE"
                _write_text_with_retry(
                    marker,
                    f"run_name={entry['run_name']}\nseed={entry['seed']}\n"
                    f"train_sha256={entry['train_sha256']}\n"
                    f"run_config_sha256={entry['run_config_sha256']}\n"
                    f"training_source_sha256={entry['training_source_sha256']}\n"
                    f"launcher_source_sha256={entry['launcher_source_sha256']}\n"
                    f"adapter_config_sha256={adapter_config_sha256}\n"
                    f"adapter_model_sha256={adapter_model_sha256}\n",
                )
                entry["adapter_config_sha256"] = adapter_config_sha256
                entry["adapter_model_sha256"] = adapter_model_sha256
                entry["status"] = "complete"
            else:
                entry["status"] = "failed"
                entry["returncode"] = return_code
                failures.append(entry)
            logging.info("finished run=%s status=%s", entry["run_name"], entry["status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _adapter_files_exist(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()


def _runtime_checkpoint_root() -> Path:
    configured = os.environ.get("LBD_RUNTIME_CHECKPOINT_ROOT")
    if not configured:
        raise ValueError(
            "LBD_RUNTIME_CHECKPOINT_ROOT is required for factorial training so intermediate "
            "Trainer checkpoints never target the shared filesystem."
        )
    return Path(configured).resolve()


def _publish_adapter_with_retry(
    source_dir: Path,
    destination_dir: Path,
    attempts: int = 10,
    wait_seconds: float = 5.0,
) -> None:
    """Publish only the two PEFT files required for evaluation, with hash checks."""

    _mkdir_with_retry(destination_dir, attempts=attempts, wait_seconds=wait_seconds)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        source = source_dir / filename
        expected_sha256 = file_sha256(source)
        destination = destination_dir / filename
        partial = destination_dir / f".{filename}.partial-{os.getpid()}"
        for attempt in range(1, attempts + 1):
            try:
                shutil.copyfile(source, partial)
                if file_sha256(partial) != expected_sha256:
                    raise OSError(f"Post-copy hash mismatch for {partial}")
                os.replace(partial, destination)
                if file_sha256(destination) != expected_sha256:
                    raise OSError(f"Published hash mismatch for {destination}")
                break
            except OSError:
                if attempt == attempts:
                    raise
                logging.warning(
                    "adapter publish attempt %d/%d failed file=%s; retrying",
                    attempt,
                    attempts,
                    filename,
                )
                time.sleep(wait_seconds)


def _parse_gpu_ids(raw: str | None) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def _mkdir_with_retry(path: Path, attempts: int = 10, wait_seconds: float = 5.0) -> None:
    """Create a BeeGFS-backed directory while tolerating transient EIO failures."""

    for attempt in range(1, attempts + 1):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except OSError:
            if attempt == attempts:
                raise
            logging.warning(
                "directory creation attempt %d/%d failed path=%s; retrying",
                attempt,
                attempts,
                path,
            )
            time.sleep(wait_seconds)


def _write_text_with_retry(
    path: Path,
    text: str,
    attempts: int = 10,
    wait_seconds: float = 5.0,
) -> None:
    """Write a small completion marker with bounded retries for transient EIO."""

    _mkdir_with_retry(path.parent, attempts=attempts, wait_seconds=wait_seconds)
    for attempt in range(1, attempts + 1):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError:
            if attempt == attempts:
                raise
            logging.warning(
                "marker write attempt %d/%d failed path=%s; retrying",
                attempt,
                attempts,
                path,
            )
            time.sleep(wait_seconds)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
