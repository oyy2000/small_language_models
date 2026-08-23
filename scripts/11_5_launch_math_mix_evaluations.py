#!/usr/bin/env python3
"""Launch the 13-model mixed-domain pilot evaluation across available GPUs."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256, runtime_metadata, validated_adapter_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_math_mix_pilot_v1.json")
    parser.add_argument("--eval-suite-manifest", required=True)
    parser.add_argument("--reference-training-manifest-glob", required=True)
    parser.add_argument("--pilot-training-manifest-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--launcher-shards", type=int, default=1)
    parser.add_argument("--launcher-shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.launcher_shards <= 0 or not 0 <= args.launcher_shard_index < args.launcher_shards:
        raise ValueError("Invalid launcher shard topology")
    config = load_config(args.config)
    model_name = str(config["student"]["model_name"])
    suite = _read_json(Path(args.eval_suite_manifest))
    if suite.get("status") != "complete":
        raise ValueError("Evaluation suite manifest is incomplete")

    reference_runs = _load_runs(args.reference_training_manifest_glob)
    pilot_runs = _load_runs(args.pilot_training_manifest_glob)
    models = _model_registry(model_name, reference_runs, pilot_runs)
    if len(models) != 13:
        raise RuntimeError(f"Expected base + 6 reference + 6 mixed models, got {len(models)}")
    assigned = [
        model
        for index, model in enumerate(models)
        if index % args.launcher_shards == args.launcher_shard_index
    ]
    if not assigned:
        raise ValueError("No models assigned to this launcher shard")

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    launcher_manifest_path = output_dir / (
        f"eval_launcher_manifest_{args.launcher_shard_index:02d}_of_{args.launcher_shards:02d}.json"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    if launcher_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite launcher manifest: {launcher_manifest_path}")

    entries = []
    commands = []
    for model in assigned:
        entry = {
            **model,
            "log_path": str(log_dir / f"{model['model_id']}.log"),
            "status": "prepared",
        }
        entries.append(entry)
        command = [
            sys.executable,
            "scripts/11_4_eval_math_mix_model.py",
            "--model-id",
            str(model["model_id"]),
            "--model-name",
            model_name,
            "--model-metadata-json",
            json.dumps(model["metadata"], ensure_ascii=False, separators=(",", ":")),
            "--eval-suite-manifest",
            args.eval_suite_manifest,
            "--output-dir",
            args.output_dir,
            "--torch-dtype",
            "bfloat16",
        ]
        if model.get("adapter_path"):
            command.extend(["--adapter-path", str(model["adapter_path"])])
        if args.skip_complete:
            command.append("--skip-complete")
        commands.append(command)

    launcher_manifest = {
        "status": "prepared" if args.dry_run else "running",
        "eval_suite_manifest": args.eval_suite_manifest,
        "eval_suite_manifest_sha256": file_sha256(args.eval_suite_manifest),
        "launcher_shards": args.launcher_shards,
        "launcher_shard_index": args.launcher_shard_index,
        "model_count": len(entries),
        "models": entries,
        "runtime": runtime_metadata(),
    }
    _write_json(launcher_manifest_path, launcher_manifest)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    max_parallel = args.max_parallel or (len(gpu_ids) if gpu_ids else 1)
    failures = _run_commands(entries, commands, gpu_ids, max_parallel)
    launcher_manifest["status"] = "failed" if failures else "complete"
    launcher_manifest["models"] = entries
    _write_json(launcher_manifest_path, launcher_manifest)
    if failures:
        raise SystemExit(f"Pilot evaluation failures={len(failures)} manifest={launcher_manifest_path}")
    logging.info("evaluation_launcher_complete models=%d manifest=%s", len(entries), launcher_manifest_path)


def _load_runs(pattern: str) -> List[Mapping[str, Any]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No training manifests matched {pattern!r}")
    runs = []
    for path in paths:
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            raise ValueError(f"Training manifest is incomplete: {path}")
        runs.extend(manifest.get("runs", []))
    return runs


def _model_registry(
    model_name: str,
    reference_runs: Iterable[Mapping[str, Any]],
    pilot_runs: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    models = [
        {
            "model_id": "base_qwen2p5_1p5b_instruct",
            "adapter_path": None,
            "metadata": {
                "training_variant": "base",
                "mode": "base",
                "budget_name": None,
                "seed": None,
            },
        }
    ]
    for variant, runs in (("gsm_only", reference_runs), ("math_mix", pilot_runs)):
        selected = []
        for run in runs:
            if run.get("generator_name") != "qwen2p5_7b":
                continue
            if run.get("mode") not in {"equal_example", "equal_token"} or int(run.get("seed", -1)) != 17:
                continue
            if variant == "math_mix" and run.get("training_variant") != "math_mix":
                continue
            adapter_path = str(run["output_dir"])
            if validated_adapter_evidence(adapter_path) is None:
                raise FileNotFoundError(f"Invalid completed adapter: {adapter_path}")
            selected.append(
                {
                    "model_id": str(run["run_name"]),
                    "adapter_path": adapter_path,
                    "metadata": {
                        "training_variant": variant,
                        "mode": str(run["mode"]),
                        "generator_name": "qwen2p5_7b",
                        "budget_name": str(run["budget_name"]),
                        "seed": 17,
                    },
                }
            )
        if len(selected) != 6:
            raise ValueError(f"Expected six {variant} 7B seed-17 adapters, got {len(selected)}")
        selected.sort(
            key=lambda item: (
                item["metadata"]["mode"],
                _budget_order(str(item["metadata"]["budget_name"])),
            )
        )
        models.extend(selected)
    if len({model["model_id"] for model in models}) != len(models):
        raise ValueError("Duplicate model IDs in pilot evaluation registry")
    return models


def _budget_order(name: str) -> int:
    return {"short_128": 0, "medium_256": 1, "long_512": 2}[name]


def _run_commands(
    entries: List[Dict[str, Any]],
    commands: List[List[str]],
    gpu_ids: List[str],
    max_parallel: int,
) -> List[Dict[str, Any]]:
    if max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    pending = list(zip(entries, commands))
    running: List[Tuple[subprocess.Popen[Any], Dict[str, Any], Any]] = []
    failures: List[Dict[str, Any]] = []
    available_gpus = list(gpu_ids)
    while pending or running:
        while pending and len(running) < max_parallel and (not gpu_ids or available_gpus):
            entry, command = pending.pop(0)
            gpu_id = available_gpus.pop(0) if gpu_ids else None
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                entry["gpu_id"] = gpu_id
            log_handle = Path(entry["log_path"]).open("w", encoding="utf-8")
            entry["status"] = "running"
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append((process, entry, log_handle))
            logging.info("launch model=%s gpu=%s", entry["model_id"], gpu_id or "default")
        still_running = []
        for process, entry, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, entry, log_handle))
                continue
            log_handle.close()
            if entry.get("gpu_id") is not None:
                available_gpus.append(str(entry["gpu_id"]))
            if return_code == 0:
                entry["status"] = "complete"
            else:
                entry["status"] = "failed"
                entry["returncode"] = return_code
                failures.append(entry)
            logging.info("finished model=%s status=%s", entry["model_id"], entry["status"])
        running = still_running
        if running:
            time.sleep(5)
    return failures


def _parse_gpu_ids(raw: str | None) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


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
