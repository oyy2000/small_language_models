#!/usr/bin/env python3
"""Audit and seal the six seed-42/73 ranked-length adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import (
    canonical_sha256,
    file_sha256,
    nonempty_line_count,
    validated_adapter_evidence,
)
from length_budget_distill.ranked_multiseed import (
    expected_run_names,
    manifest_field_equal,
    validate_training_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-config",
        default="configs/capacity_length_ranked_sampling_7b_training_seed42_73_v1.json",
    )
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--launch-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    report_path = output_dir / "training_audit.json"
    marker_path = output_dir / "TRAINING_COMPLETE"
    if report_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite training audit evidence: {output_dir}")

    errors: List[str] = []
    training_config_path = _resolve(args.training_config)
    prepared_path = _resolve(args.prepared_manifest)
    launch_path = _resolve(args.launch_manifest)
    training_config = load_config(str(training_config_path))
    scope = validate_training_scope(training_config)
    protocol = {key: value for key, value in training_config.items() if key != "_config_path"}
    config_hash = canonical_sha256(protocol)
    config_file_hash = file_sha256(training_config_path)
    prepared = _read_json(prepared_path)
    launch = _read_json(launch_path)

    _expect(prepared.get("status") == "complete", "Prepared manifest is incomplete.", errors)
    _expect(prepared.get("config_hash") == config_hash, "Prepared config hash mismatch.", errors)
    _expect(
        prepared.get("config_file_sha256") == config_file_hash,
        "Prepared config file hash mismatch.",
        errors,
    )
    _expect(launch.get("status") == "complete", "Training launch manifest is incomplete.", errors)
    _expect(launch.get("config_hash") == config_hash, "Launch config hash mismatch.", errors)
    _expect(
        _resolve(str(launch.get("dataset_manifest", ""))) == prepared_path,
        "Launch manifest references a different prepared manifest.",
        errors,
    )

    generator_name = "qwen2p5_7b"
    expected_names = expected_run_names(generator_name, scope["training_seeds"])
    expected_run_count = int(scope["run_count"])
    prepared_runs = _unique_runs(prepared.get("runs", []), "prepared", errors)
    launch_runs = _unique_runs(launch.get("runs", []), "launch", errors)
    _expect(set(prepared_runs) == expected_names, "Prepared run identities mismatch.", errors)
    _expect(set(launch_runs) == expected_names, "Launch run identities mismatch.", errors)
    _expect(
        int(prepared.get("run_count", -1)) == expected_run_count == len(prepared_runs),
        "Prepared run count mismatch.",
        errors,
    )
    _expect(
        int(launch.get("run_count", -1)) == expected_run_count == len(launch_runs),
        "Launch run count mismatch.",
        errors,
    )

    audited_runs: List[Dict[str, Any]] = []
    output_dirs: set[str] = set()
    for run_name in sorted(expected_names):
        prepared_run = prepared_runs.get(run_name)
        launch_run = launch_runs.get(run_name)
        if prepared_run is None or launch_run is None:
            continue
        _expect(
            launch_run.get("status") in {"complete", "skipped_complete"},
            f"Training run is not complete: {run_name}",
            errors,
        )
        for key in (
            "mode",
            "generator_name",
            "budget_name",
            "seed",
            "train_path",
            "train_sha256",
            "n",
            "supervised_tokens",
        ):
            _expect(
                manifest_field_equal(key, launch_run.get(key), prepared_run.get(key)),
                f"Prepared/launch field mismatch for {run_name}: {key}",
                errors,
            )

        train_path = _resolve(str(prepared_run.get("train_path", "")))
        expected_train_hash = str(prepared_run.get("train_sha256", ""))
        if not train_path.is_file():
            errors.append(f"Training dataset is missing for {run_name}: {train_path}")
        else:
            _expect(
                file_sha256(train_path) == expected_train_hash,
                f"Training dataset hash mismatch for {run_name}.",
                errors,
            )
            _expect(
                nonempty_line_count(train_path) == int(prepared_run.get("n", -1)),
                f"Training dataset row-count mismatch for {run_name}.",
                errors,
            )

        run_config_path = _resolve(str(launch_run.get("config_path", "")))
        if not run_config_path.is_file():
            errors.append(f"Run config is missing for {run_name}: {run_config_path}")
        else:
            _expect(
                file_sha256(run_config_path) == launch_run.get("run_config_sha256"),
                f"Run config hash mismatch for {run_name}.",
                errors,
            )

        output_path = _resolve(str(launch_run.get("output_dir", "")))
        normalized_output = str(output_path)
        _expect(
            normalized_output not in output_dirs,
            f"Duplicate adapter output directory: {normalized_output}",
            errors,
        )
        output_dirs.add(normalized_output)
        evidence = validated_adapter_evidence(output_path)
        if evidence is None:
            errors.append(f"Adapter evidence is incomplete for {run_name}: {output_path}")
            continue
        expected_evidence = {
            "run_name": run_name,
            "seed": str(prepared_run["seed"]),
            "train_sha256": expected_train_hash,
            "run_config_sha256": str(launch_run.get("run_config_sha256", "")),
            "training_source_sha256": str(launch_run.get("training_source_sha256", "")),
            "launcher_source_sha256": str(launch_run.get("launcher_source_sha256", "")),
        }
        for key, expected_value in expected_evidence.items():
            _expect(
                str(evidence.get(key, "")) == expected_value,
                f"Adapter marker mismatch for {run_name}: {key}",
                errors,
            )
        audited_runs.append(
            {
                "run_name": run_name,
                "generator_name": prepared_run["generator_name"],
                "budget_name": prepared_run["budget_name"],
                "seed": int(prepared_run["seed"]),
                "n": int(prepared_run["n"]),
                "supervised_tokens": int(prepared_run["supervised_tokens"]),
                "train_path": str(train_path),
                "train_sha256": expected_train_hash,
                "run_config_path": str(run_config_path),
                "run_config_sha256": str(launch_run["run_config_sha256"]),
                "output_dir": str(output_path),
                "training_source_sha256": str(evidence["training_source_sha256"]),
                "launcher_source_sha256": str(evidence["launcher_source_sha256"]),
                "adapter_config_sha256": str(evidence["adapter_config_sha256"]),
                "adapter_model_sha256": str(evidence["adapter_model_sha256"]),
            }
        )

    _expect(
        len(audited_runs) == expected_run_count,
        f"Validated adapter count mismatch: expected={expected_run_count} actual={len(audited_runs)}",
        errors,
    )
    report = {
        "status": "passed" if not errors else "failed",
        "evidence_level": "comparative_multiseed_extension_training",
        "experiment_name": training_config["experiment_name"],
        "config_path": str(training_config_path),
        "config_hash": config_hash,
        "config_file_sha256": config_file_hash,
        "prepared_manifest": str(prepared_path),
        "prepared_manifest_sha256": file_sha256(prepared_path),
        "launch_manifest": str(launch_path),
        "launch_manifest_sha256": file_sha256(launch_path),
        "expected_run_count": expected_run_count,
        "validated_run_count": len(audited_runs),
        "seeds": scope["training_seeds"],
        "runs": audited_runs,
        "errors": errors,
    }
    if errors:
        raise SystemExit("Ranked multi-seed training audit failed: " + " | ".join(errors))
    _write_json(report_path, report)
    _write_text(
        marker_path,
        "status=passed\n"
        "evidence_level=comparative_multiseed_extension_training\n"
        f"config_hash={config_hash}\n"
        f"prepared_manifest_sha256={file_sha256(prepared_path)}\n"
        f"launch_manifest_sha256={file_sha256(launch_path)}\n"
        f"training_audit_sha256={file_sha256(report_path)}\n"
        f"run_count={len(audited_runs)}\n"
        "seeds=42,73\n",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "training_audit": str(report_path),
                "completion_marker": str(marker_path),
                "run_count": len(audited_runs),
            },
            indent=2,
        )
    )


def _unique_runs(raw_runs: Any, label: str, errors: List[str]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_runs, list):
        errors.append(f"{label.capitalize()} runs must be a list.")
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            errors.append(f"{label.capitalize()} run entry is not an object.")
            continue
        run_name = str(raw_run.get("run_name", ""))
        if not run_name:
            errors.append(f"{label.capitalize()} run is missing run_name.")
        elif run_name in result:
            errors.append(f"Duplicate {label} run_name: {run_name}")
        else:
            result[run_name] = raw_run
    return result


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_text(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


if __name__ == "__main__":
    main()
