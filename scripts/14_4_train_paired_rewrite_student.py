#!/usr/bin/env python3
"""Train one paired-rewrite SFT run and publish hash-bound epoch snapshots."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256, runtime_metadata
from length_budget_distill.training import run_trl_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    config = load_config(str(config_path))
    metadata = dict(config["paired_rewrite_metadata"])
    run_name = str(metadata["run_name"])
    publication_dir = _resolve(str(config["training"]["output_dir"]))
    run_marker = publication_dir / "TRAINING_RUN_COMPLETE"
    if run_marker.is_file() and args.skip_complete:
        _validate_completed_run(publication_dir, metadata)
        logging.info("skip_complete_run=%s", run_name)
        return
    if publication_dir.exists() and any(publication_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite training output: {publication_dir}")
    runtime_override = os.environ.get("LBD_RUNTIME_OUTPUT_DIR")
    if not runtime_override:
        raise ValueError("LBD_RUNTIME_OUTPUT_DIR must point to a node-local empty directory")
    runtime_dir = Path(runtime_override).resolve()
    if runtime_dir.exists() and any(runtime_dir.iterdir()):
        raise FileExistsError(f"Node-local training directory is non-empty: {runtime_dir}")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    configured_train_path = _resolve(str(config["data"]["train_path"]))
    if file_sha256(configured_train_path) != metadata.get("train_sha256"):
        raise ValueError(f"Training data hash mismatch for {run_name}")
    started = time.monotonic()
    trainer = run_trl_sft(config)
    log_history = list(getattr(trainer.state, "log_history", []))
    history_path = runtime_dir / "training_log_history.json"
    _write_json(history_path, {"run_name": run_name, "log_history": log_history})

    checkpoints = _checkpoint_epochs(runtime_dir)
    final_epoch = float(getattr(trainer.state, "epoch", metadata["num_train_epochs"]))
    checkpoints.append((final_epoch, runtime_dir))
    published_snapshots: List[Dict[str, Any]] = []
    for target_epoch in metadata["snapshot_epochs"]:
        actual_epoch, source_dir = _nearest_snapshot(checkpoints, float(target_epoch))
        destination = publication_dir / "snapshots" / _epoch_name(float(target_epoch))
        snapshot = _publish_snapshot(
            source_dir,
            destination,
            run_name=run_name,
            target_epoch=float(target_epoch),
            actual_epoch=actual_epoch,
            train_sha256=str(metadata["train_sha256"]),
            run_config_sha256=file_sha256(config_path),
            history_path=history_path,
        )
        published_snapshots.append(snapshot)

    manifest = {
        "status": "complete",
        "stage": metadata["stage"],
        "evidence_level": "exploratory_single_seed_pilot",
        "run_name": run_name,
        "condition": metadata["condition"],
        "schedule": metadata.get("schedule", "base_single_stage"),
        "learning_rate": metadata["learning_rate"],
        "num_train_epochs": metadata["num_train_epochs"],
        "train_path": str(configured_train_path),
        "train_sha256": metadata["train_sha256"],
        "run_config_path": str(config_path),
        "run_config_sha256": file_sha256(config_path),
        "training_source_sha256": file_sha256(PROJECT_ROOT / "src/length_budget_distill/training.py"),
        "runner_source_sha256": file_sha256(Path(__file__).resolve()),
        "resume_adapter_path": metadata.get("resume_adapter_path"),
        "training_log_history": log_history,
        "snapshots": published_snapshots,
        "runtime": runtime_metadata(),
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest_path = publication_dir / "training_manifest.json"
    _write_json(manifest_path, manifest)
    run_marker.write_text(
        f"run_name={run_name}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        f"snapshot_count={len(published_snapshots)}\ntrain_sha256={metadata['train_sha256']}\n",
        encoding="utf-8",
    )
    logging.info("paired_training_complete run=%s snapshots=%d", run_name, len(published_snapshots))


def _checkpoint_epochs(runtime_dir: Path) -> List[Tuple[float, Path]]:
    checkpoints: List[Tuple[float, Path]] = []
    for path in sorted(runtime_dir.glob("checkpoint-*")):
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = _read_json(state_path)
        epoch = state.get("epoch")
        if epoch is not None and _adapter_files_exist(path):
            checkpoints.append((float(epoch), path))
    return checkpoints


def _nearest_snapshot(candidates: Sequence[Tuple[float, Path]], target: float) -> Tuple[float, Path]:
    eligible = [(epoch, path) for epoch, path in candidates if _adapter_files_exist(path)]
    if not eligible:
        raise FileNotFoundError("Trainer produced no complete adapter snapshot")
    epoch, path = min(eligible, key=lambda item: (abs(item[0] - target), item[0]))
    if abs(epoch - target) > 0.06:
        raise ValueError(f"No checkpoint close to epoch {target}; nearest={epoch} path={path}")
    return epoch, path


def _publish_snapshot(
    source_dir: Path,
    destination: Path,
    *,
    run_name: str,
    target_epoch: float,
    actual_epoch: float,
    train_sha256: str,
    run_config_sha256: str,
    history_path: Path,
) -> Dict[str, Any]:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite snapshot: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for filename in required:
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing trainer snapshot file: {source}")
        _copy_with_verification(source, destination / filename)
    state_source = source_dir / "trainer_state.json"
    if state_source.is_file():
        _copy_with_verification(state_source, destination / "trainer_state.json")
    _copy_with_verification(history_path, destination / "training_log_history.json")
    hashes = {filename: file_sha256(destination / filename) for filename in required}
    marker_path = destination / "SNAPSHOT_COMPLETE"
    marker_path.write_text(
        f"run_name={run_name}\ntarget_epoch={target_epoch}\nactual_epoch={actual_epoch}\n"
        f"train_sha256={train_sha256}\nrun_config_sha256={run_config_sha256}\n"
        f"adapter_config_sha256={hashes['adapter_config.json']}\n"
        f"adapter_model_sha256={hashes['adapter_model.safetensors']}\n",
        encoding="utf-8",
    )
    return {
        "target_epoch": target_epoch,
        "actual_epoch": actual_epoch,
        "path": str(destination),
        "marker_sha256": file_sha256(marker_path),
        "adapter_config_sha256": hashes["adapter_config.json"],
        "adapter_model_sha256": hashes["adapter_model.safetensors"],
    }


def _copy_with_verification(source: Path, destination: Path) -> None:
    expected = file_sha256(source)
    for attempt in range(3):
        shutil.copy2(source, destination)
        if file_sha256(destination) == expected:
            return
        if attempt < 2:
            time.sleep(2**attempt)
    raise IOError(f"Failed hash verification while copying {source} to {destination}")


def _validate_completed_run(path: Path, metadata: Mapping[str, Any]) -> None:
    manifest = _read_json(path / "training_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("run_name") != metadata["run_name"]:
        raise ValueError(f"Existing completion marker is invalid: {path}")
    if manifest.get("train_sha256") != metadata["train_sha256"]:
        raise ValueError(f"Existing completion is bound to different training data: {path}")
    for snapshot in manifest.get("snapshots", []):
        snapshot_path = Path(snapshot["path"])
        if file_sha256(snapshot_path / "adapter_config.json") != snapshot["adapter_config_sha256"]:
            raise ValueError(f"Snapshot adapter config hash mismatch: {snapshot_path}")
        if file_sha256(snapshot_path / "adapter_model.safetensors") != snapshot["adapter_model_sha256"]:
            raise ValueError(f"Snapshot adapter model hash mismatch: {snapshot_path}")


def _adapter_files_exist(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()


def _epoch_name(epoch: float) -> str:
    return f"epoch_{epoch:g}".replace(".", "p")


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
