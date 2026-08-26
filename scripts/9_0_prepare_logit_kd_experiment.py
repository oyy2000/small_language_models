#!/usr/bin/env python3
"""Validate and register immutable parent evidence for the logit-KD experiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import validated_adapter_evidence
from length_budget_distill.logit_kd import (
    baseline_sft_run_name,
    file_sha256,
    load_and_validate_tokenizers,
    load_protocol,
    protocol_hash,
    read_json,
    resolve_project_path,
    runtime_metadata,
    tokenize_kd_record,
    validate_budget_dataset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def _ensure_beegfs_result_link(protocol: Dict[str, Any]) -> Path:
    relative = Path(protocol["outputs"]["result_root"])
    if relative.is_absolute() or relative.parts[:1] != ("results",):
        raise ValueError("outputs.result_root must be a relative path directly under results/.")
    local_path = PROJECT_ROOT / relative
    beegfs_base = Path(os.environ.get("LBD_BEEGFS_PROJECT_ROOT", "/mnt/beegfs/youyang7/projects/small_language_model"))
    if not beegfs_base.is_dir():
        raise FileNotFoundError(f"BeeGFS project root is unavailable: {beegfs_base}")
    target = beegfs_base / relative
    target.mkdir(parents=True, exist_ok=True)
    if local_path.is_symlink():
        if local_path.resolve() != target.resolve():
            raise ValueError(f"Result symlink points to the wrong target: {local_path}")
    elif local_path.exists():
        raise FileExistsError(f"Result path exists but is not the registered BeeGFS symlink: {local_path}")
    else:
        local_path.symlink_to(target)
    return local_path


def _key_value_marker(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"Invalid parent completion marker line: {line!r}")
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _parent_evaluations(parent_root: Path) -> Dict[str, Dict[str, Any]]:
    by_run: Dict[str, Dict[str, Any]] = {}
    for path in sorted((parent_root / "eval").glob("eval_manifest_formal_shard_*_of_*.json")):
        manifest = read_json(path)
        if manifest.get("status") != "complete":
            raise ValueError(f"Parent evaluation manifest is incomplete: {path}")
        for run in manifest.get("runs", []):
            run_name = str(run.get("run_name"))
            if run_name in by_run:
                raise ValueError(f"Duplicate parent evaluation run: {run_name}")
            prediction = resolve_project_path(str(run.get("prediction_path")))
            summary = resolve_project_path(str(run.get("summary_path")))
            if not prediction.is_file() or run.get("prediction_sha256") != file_sha256(prediction):
                raise ValueError(f"Parent prediction evidence mismatch: {run_name}")
            if not summary.is_file() or run.get("summary_sha256") != file_sha256(summary):
                raise ValueError(f"Parent summary evidence mismatch: {run_name}")
            by_run[run_name] = {
                "prediction_path": str(prediction),
                "prediction_sha256": file_sha256(prediction),
                "summary_path": str(summary),
                "summary_sha256": file_sha256(summary),
            }
    return by_run


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = resolve_project_path(args.config)
    protocol = load_protocol(config_path)
    result_root = _ensure_beegfs_result_link(protocol)
    prepared_marker_path = result_root / "PREPARED"
    evidence_path = result_root / "preflight" / "parent_evidence.json"
    if prepared_marker_path.is_file() and evidence_path.is_file():
        marker = read_json(prepared_marker_path)
        if marker.get("parent_evidence_sha256") == file_sha256(evidence_path):
            if args.skip_complete:
                logging.info("preflight_already_complete evidence=%s", evidence_path)
                return
            raise FileExistsError(f"Preflight is already complete: {prepared_marker_path}")
        raise ValueError(f"Existing preflight marker is invalid: {prepared_marker_path}")

    parent = protocol["parent"]
    parent_config = resolve_project_path(parent["config_path"])
    run_config = resolve_project_path(parent["run_config_path"])
    if file_sha256(parent_config) != parent["config_file_sha256"]:
        raise ValueError("Parent protocol file hash mismatch.")
    if file_sha256(run_config) != parent["run_config_sha256"]:
        raise ValueError("Parent run-config file hash mismatch.")
    parent_root = resolve_project_path(parent["result_root"])
    parent_marker = resolve_project_path(parent["completion_marker"])
    if not parent_marker.is_file():
        raise FileNotFoundError(f"Parent formal completion marker is missing: {parent_marker}")
    marker_values = _key_value_marker(parent_marker)
    if marker_values.get("config_hash") != parent["config_canonical_sha256"]:
        raise ValueError("Parent completion marker config hash mismatch.")
    completion_audit = parent_root / "completion_audit.json"
    if not completion_audit.is_file() or marker_values.get("completion_audit_sha256") != file_sha256(completion_audit):
        raise ValueError("Parent completion-audit evidence mismatch.")
    audit = read_json(completion_audit)
    if audit.get("status") != "passed":
        raise ValueError("Parent completion audit did not pass.")

    evaluation_evidence = _parent_evaluations(parent_root)
    expected_runs = {"base_qwen2p5_1p5b_instruct"}
    budget_evidence: Dict[str, Any] = {}
    budget_rows: Dict[str, Any] = {}
    for budget_name, budget in protocol["budgets"].items():
        data_path, rows = validate_budget_dataset(protocol, budget_name)
        budget_rows[budget_name] = rows
        adapter_path = resolve_project_path(budget["baseline_adapter"])
        adapter = validated_adapter_evidence(adapter_path)
        if adapter is None:
            raise ValueError(f"Parent SFT adapter is incomplete: {adapter_path}")
        run_name = baseline_sft_run_name(protocol, budget_name)
        expected_runs.add(run_name)
        if run_name not in evaluation_evidence:
            raise ValueError(f"Parent SFT evaluation is missing: {run_name}")
        if Path(evaluation_evidence[run_name]["prediction_path"]).resolve() != resolve_project_path(
            budget["baseline_prediction"]
        ).resolve():
            raise ValueError(f"Configured parent prediction path mismatch: {budget_name}")
        budget_evidence[budget_name] = {
            "train_path": str(data_path),
            "train_sha256": file_sha256(data_path),
            "records": len(rows),
            "baseline_adapter": str(adapter_path),
            "baseline_adapter_evidence": adapter,
            "baseline_evaluation": evaluation_evidence[run_name],
        }
    missing_runs = expected_runs - set(evaluation_evidence)
    if missing_runs:
        raise ValueError(f"Missing registered parent evaluation runs: {sorted(missing_runs)}")

    tokenizer_evidence = None
    context_alignment: Dict[str, Any] = {}
    if not args.skip_tokenizer_check:
        student_tokenizer, teacher_tokenizer, valid_vocab_size, tokenizer_evidence = load_and_validate_tokenizers(
            protocol
        )
        if valid_vocab_size != int(protocol["models"]["tokenizer"]["expected_length"]):
            raise ValueError("Registered valid vocabulary size mismatch.")
        max_length = int(protocol["training"]["max_length"])
        for budget_name, rows in budget_rows.items():
            encoded = [
                tokenize_kd_record(protocol, student_tokenizer, row, max_length)
                for row in rows
            ]
            context_alignment[budget_name] = {
                "status": "passed",
                "records": len(encoded),
                "target_mismatch_count": 0,
                "context_mode": encoded[0]["context_mode"],
                "student_context_field": encoded[0]["student_context_field"],
                "teacher_context_field": encoded[0]["teacher_context_field"],
                "max_teacher_sequence_tokens": max(
                    len(item["teacher_input_ids"]) for item in encoded
                ),
                "max_student_sequence_tokens": max(
                    len(item["student_input_ids"]) for item in encoded
                ),
                "prompt_token_delta_min": min(
                    item["teacher_prompt_token_count"] - item["student_prompt_token_count"]
                    for item in encoded
                ),
                "prompt_token_delta_max": max(
                    item["teacher_prompt_token_count"] - item["student_prompt_token_count"]
                    for item in encoded
                ),
            }
        del student_tokenizer, teacher_tokenizer

    comparison_evidence = None
    comparison = protocol.get("comparison")
    if comparison:
        comparison_marker_path = resolve_project_path(
            comparison["same_prompt_completion_marker"]
        )
        analysis_path = resolve_project_path(comparison["same_prompt_analysis"])
        if not comparison_marker_path.is_file() or not analysis_path.is_file():
            raise FileNotFoundError("Registered same-prompt comparison evidence is missing.")
        comparison_marker = read_json(comparison_marker_path)
        comparison_audit = Path(str(comparison_marker.get("completion_audit_path")))
        if (
            comparison_marker.get("status") != "complete"
            or not comparison_audit.is_file()
            or comparison_marker.get("completion_audit_sha256") != file_sha256(comparison_audit)
        ):
            raise ValueError("Same-prompt comparison completion evidence is invalid.")
        if read_json(comparison_audit).get("status") != "passed":
            raise ValueError("Same-prompt comparison completion audit did not pass.")
        comparison_analysis = read_json(analysis_path)
        if comparison_analysis.get("status") != "complete":
            raise ValueError("Same-prompt comparison analysis is incomplete.")
        comparison_evidence = {
            "completion_marker": str(comparison_marker_path),
            "completion_marker_sha256": file_sha256(comparison_marker_path),
            "completion_audit": str(comparison_audit),
            "completion_audit_sha256": file_sha256(comparison_audit),
            "analysis": str(analysis_path),
            "analysis_sha256": file_sha256(analysis_path),
        }

    evidence = {
        "status": "passed",
        "experiment_name": protocol["experiment_name"],
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "parent_config_path": str(parent_config),
        "parent_config_sha256": file_sha256(parent_config),
        "parent_run_config_path": str(run_config),
        "parent_run_config_sha256": file_sha256(run_config),
        "parent_completion_marker": str(parent_marker),
        "parent_completion_marker_sha256": file_sha256(parent_marker),
        "parent_completion_audit": str(completion_audit),
        "parent_completion_audit_sha256": file_sha256(completion_audit),
        "budgets": budget_evidence,
        "base_evaluation": evaluation_evidence["base_qwen2p5_1p5b_instruct"],
        "tokenizer": tokenizer_evidence,
        "context_alignment": context_alignment,
        "comparison": comparison_evidence,
        "runtime": runtime_metadata(),
    }
    write_json(evidence_path, evidence)
    write_json(
        prepared_marker_path,
        {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "parent_evidence_sha256": file_sha256(evidence_path),
        },
    )
    logging.info("logit_kd_preflight_complete evidence=%s", evidence_path)


if __name__ == "__main__":
    main()
