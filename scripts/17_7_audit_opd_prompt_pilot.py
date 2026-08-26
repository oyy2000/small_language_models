#!/usr/bin/env python3
"""Independently audit the complete exploratory pure-OPD pilot evidence chain."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import file_sha256
from length_budget_distill.opd import (
    OPD_ARMS,
    preflight_summary,
    protocol_hash,
    read_gzip_jsonl,
    read_json,
    validate_opd_protocol,
    validate_reference_manifest,
    validated_opd_adapter,
    write_json,
)
from length_budget_distill.opd_analysis import (
    completed_opd_evaluation,
    opd_advancement_decision,
    paired_opd_contrast,
)
from length_budget_distill.records import read_jsonl


MODEL_IDS = ("base",) + OPD_ARMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--preflight-dir", default=None)
    parser.add_argument("--primary-eval-manifest", default=None)
    parser.add_argument("--secondary-eval-manifest", default=None)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    result_root = _resolve(protocol["outputs"]["result_root"])
    checkpoint_root = _resolve(protocol["outputs"]["checkpoint_root"])
    defaults = {
        "reference_dir": result_root / "pilot/references",
        "preflight_dir": result_root / "pilot/preflight",
        "primary_eval_manifest": result_root
        / "pilot/evaluation/primary/evaluation_launcher_manifest.json",
        "secondary_eval_manifest": result_root
        / "pilot/evaluation/secondary/evaluation_launcher_manifest.json",
        "analysis_dir": result_root / "pilot/analysis",
        "figure_dir": _resolve(protocol["outputs"]["figure_root"]) / "pilot",
        "output_dir": result_root / "pilot/audit",
    }
    reference_dir = _optional_resolve(args.reference_dir, defaults["reference_dir"])
    preflight_dir = _optional_resolve(args.preflight_dir, defaults["preflight_dir"])
    eval_manifests = {
        "primary_evaluation": _optional_resolve(
            args.primary_eval_manifest, defaults["primary_eval_manifest"]
        ),
        "secondary_evaluation": _optional_resolve(
            args.secondary_eval_manifest, defaults["secondary_eval_manifest"]
        ),
    }
    analysis_dir = _optional_resolve(args.analysis_dir, defaults["analysis_dir"])
    figure_dir = _optional_resolve(args.figure_dir, defaults["figure_dir"])
    output_dir = _optional_resolve(args.output_dir, defaults["output_dir"])
    completion_marker = result_root / "pilot/PILOT_COMPLETE"
    if output_dir.exists() and completion_marker.is_file() and args.skip_complete:
        marker = read_json(completion_marker)
        audit_path = Path(str(marker.get("completion_audit_path", "")))
        if (
            marker.get("status") == "passed"
            and marker.get("protocol_hash") == protocol_hash(protocol)
            and audit_path.is_file()
            and marker.get("completion_audit_sha256") == file_sha256(audit_path)
            and read_json(audit_path).get("audit_source_sha256")
            == file_sha256(Path(__file__).resolve())
        ):
            print(f"OPD exploratory pilot audit already complete: {completion_marker}")
            return
        raise ValueError("Existing OPD pilot completion evidence failed validation.")
    if output_dir.exists() or completion_marker.exists():
        raise FileExistsError("Refusing to overwrite OPD completion-audit evidence.")
    output_dir.mkdir(parents=True, exist_ok=False)

    errors: List[str] = []
    storage_evidence = {
        "stable_result_root": str(result_root),
        "result_root_is_symlink": result_root.is_symlink(),
        "resolved_result_root": str(result_root.resolve()),
        "resolved_checkpoint_root": str(checkpoint_root.resolve()),
    }
    _expect(result_root.is_symlink(), "OPD result root is not an experiment-scoped symlink.", errors)
    _expect(_on_beegfs(result_root.resolve()), "OPD result root does not resolve under BeeGFS.", errors)
    _expect(_on_beegfs(checkpoint_root.resolve()), "OPD checkpoint root does not resolve under BeeGFS.", errors)
    reference_evidence, reference_rows = _audit_references(
        protocol, reference_dir, errors
    )
    preflight_evidence = _audit_preflight(protocol, preflight_dir, errors)
    training_evidence = _audit_training(
        protocol,
        checkpoint_root,
        result_root,
        reference_rows,
        errors,
    )
    evaluation_evidence, predictions = _audit_evaluations(
        protocol,
        eval_manifests,
        errors,
    )
    analysis_evidence = _audit_analysis(
        protocol,
        analysis_dir,
        figure_dir,
        predictions,
        errors,
    )

    report = {
        "status": "passed" if not errors else "failed",
        "stage": "pilot",
        "evidence_level": protocol["evidence_level"],
        "scope": protocol["scope"],
        "experiment_name": protocol["experiment_name"],
        "protocol_variant": protocol["protocol_variant"],
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "audit_source_sha256": file_sha256(Path(__file__).resolve()),
        "method": "pure_sampled_token_opd",
        "objective": protocol["objective"]["name"],
        "counts": {
            "training_references": len(reference_rows),
            "preflight_rollouts": preflight_evidence.get("rollout_count"),
            "trained_adapters": len(training_evidence.get("arms", {})),
            "evaluated_models_per_split": len(MODEL_IDS),
            "evaluation_splits": len(evaluation_evidence),
        },
        "evidence": {
            "storage": storage_evidence,
            "references": reference_evidence,
            "preflight": preflight_evidence,
            "training": training_evidence,
            "evaluation": evaluation_evidence,
            "analysis": analysis_evidence,
        },
        "loss_boundary": {
            "gold_labels_used_in_loss": False,
            "length_used_in_loss": False,
            "scalar_reward_used": False,
            "value_head_used": False,
        },
        "errors": errors,
    }
    audit_path = output_dir / (
        "completion_audit.json" if not errors else "completion_audit_failed.json"
    )
    write_json(audit_path, report)
    if errors:
        raise SystemExit("OPD pilot completion audit failed: " + " | ".join(errors))
    write_json(
        completion_marker,
        {
            "status": "passed",
            "stage": "exploratory_single_seed_pilot",
            "scope": "GSM8K_only",
            "protocol_hash": protocol_hash(protocol),
            "completion_audit_path": str(audit_path),
            "completion_audit_sha256": file_sha256(audit_path),
            "analysis_artifact_manifest_sha256": analysis_evidence[
                "artifact_manifest_sha256"
            ],
        },
    )
    print(f"OPD exploratory pilot audit passed: {completion_marker}")


def _audit_references(
    protocol: Mapping[str, Any],
    reference_dir: Path,
    errors: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = reference_dir / "reference_manifest.json"
    marker_path = reference_dir / "REFERENCES_COMPLETE"
    try:
        manifest, rows = validate_reference_manifest(protocol, manifest_path)
        marker = read_json(marker_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"Reference validation failed: {exc}")
        return {}, []
    expected_marker = {
        "status": "complete",
        "protocol_hash": protocol_hash(protocol),
        "reference_manifest_sha256": file_sha256(manifest_path),
        "reference_sha256": manifest["reference_sha256"],
        "record_count": int(protocol["splits"]["training"]["limit"]),
    }
    _expect(
        all(marker.get(key) == value for key, value in expected_marker.items()),
        "Reference completion marker mismatch.",
        errors,
    )
    _expect(
        manifest.get("source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/17_2_merge_opd_reference_lengths.py"),
        "Reference merge source hash mismatch.",
        errors,
    )
    for shard in manifest.get("shards", []):
        try:
            shard_manifest = read_json(shard["manifest_path"])
            _expect(
                shard.get("manifest_sha256") == file_sha256(shard["manifest_path"]),
                f"Reference shard manifest hash mismatch: {shard.get('manifest_path')}",
                errors,
            )
            _expect(
                shard_manifest.get("source_sha256")
                == file_sha256(PROJECT_ROOT / "scripts/17_1_generate_opd_reference_lengths.py"),
                "Reference generation source hash mismatch.",
                errors,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Reference shard audit failed: {exc}")
    _expect(
        len(manifest.get("shards", []))
        == int(protocol["reference_generation"]["num_shards"]),
        "Reference shard count mismatch.",
        errors,
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "marker_path": str(marker_path),
        "marker_sha256": file_sha256(marker_path),
        "record_count": len(rows),
    }, rows


def _audit_preflight(
    protocol: Mapping[str, Any],
    preflight_dir: Path,
    errors: List[str],
) -> Dict[str, Any]:
    paths = {
        "marker": preflight_dir / "PREFLIGHT_COMPLETE",
        "manifest": preflight_dir / "preflight_manifest.json",
        "summary": preflight_dir / "preflight_summary.json",
        "rollouts": preflight_dir / "preflight_rollouts.jsonl.gz",
    }
    try:
        marker = read_json(paths["marker"])
        manifest = read_json(paths["manifest"])
        summary = read_json(paths["summary"])
        rows = list(read_gzip_jsonl(paths["rollouts"]))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"Preflight validation failed: {exc}")
        return {}
    expected_n = (
        int(protocol["preflight"]["prompt_count"])
        * len(OPD_ARMS)
        * int(protocol["training"]["rollouts_per_prompt"])
    )
    checks = {
        "status": "passed",
        "protocol_hash": protocol_hash(protocol),
        "manifest_sha256": file_sha256(paths["manifest"]),
        "summary_sha256": file_sha256(paths["summary"]),
        "rollout_sha256": file_sha256(paths["rollouts"]),
    }
    _expect(
        all(marker.get(key) == value for key, value in checks.items()),
        "Preflight marker mismatch.",
        errors,
    )
    _expect(manifest.get("status") == "passed", "Preflight manifest did not pass.", errors)
    _expect(manifest.get("stage") == "preflight_merge", "Preflight merge stage mismatch.", errors)
    _expect(manifest.get("protocol_hash") == protocol_hash(protocol), "Preflight protocol mismatch.", errors)
    _expect(manifest.get("rollout_sha256") == file_sha256(paths["rollouts"]), "Preflight rollout hash mismatch.", errors)
    _expect(manifest.get("summary_sha256") == file_sha256(paths["summary"]), "Preflight summary hash mismatch.", errors)
    _expect(
        manifest.get("generation_source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/17_3_preflight_opd_signal.py"),
        "Preflight generation source hash mismatch.",
        errors,
    )
    _expect(
        manifest.get("merge_source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/17_3_merge_opd_preflight.py"),
        "Preflight merge source hash mismatch.",
        errors,
    )
    _expect(
        len(manifest.get("shards", [])) == int(protocol["preflight"]["num_shards"]),
        "Preflight shard count mismatch.",
        errors,
    )
    for shard in manifest.get("shards", []):
        try:
            shard_manifest_path = Path(str(shard["manifest_path"]))
            shard_rollout_path = Path(str(shard["rollout_path"]))
            shard_manifest = read_json(shard_manifest_path)
            _expect(
                shard.get("manifest_sha256") == file_sha256(shard_manifest_path),
                f"Preflight shard manifest hash mismatch: {shard_manifest_path}",
                errors,
            )
            _expect(
                shard.get("rollout_sha256") == file_sha256(shard_rollout_path),
                f"Preflight shard rollout hash mismatch: {shard_rollout_path}",
                errors,
            )
            _expect(
                shard_manifest.get("source_sha256")
                == file_sha256(PROJECT_ROOT / "scripts/17_3_preflight_opd_signal.py"),
                f"Preflight shard source mismatch: {shard_manifest_path}",
                errors,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Preflight shard audit failed: {exc}")
    _expect(len(rows) == expected_n == int(manifest.get("rollout_count", -1)), "Preflight rollout count mismatch.", errors)
    identities = [(row.get("problem_id"), row.get("arm"), row.get("candidate_index")) for row in rows]
    _expect(len(identities) == len(set(identities)), "Preflight rollout identities are not unique.", errors)
    arm_counts = Counter(str(row.get("arm")) for row in rows)
    expected_per_arm = int(protocol["preflight"]["prompt_count"]) * int(
        protocol["training"]["rollouts_per_prompt"]
    )
    _expect(
        arm_counts == Counter({arm: expected_per_arm for arm in OPD_ARMS}),
        "Preflight arm counts mismatch.",
        errors,
    )
    teacher_prompts: Dict[str, str] = {}
    topk_count = 0
    for row in rows:
        _audit_rollout_row(row, errors, context="preflight")
        problem_id = str(row.get("problem_id"))
        teacher_prompt = str(row.get("teacher_prompt", ""))
        if problem_id in teacher_prompts:
            _expect(
                teacher_prompts[problem_id] == teacher_prompt,
                f"Preflight teacher prompt is not common across arms: {problem_id}",
                errors,
            )
        else:
            teacher_prompts[problem_id] = teacher_prompt
        if row.get("arm") == "standard_prompt":
            _expect(
                row.get("student_prompt") == row.get("teacher_prompt"),
                f"Preflight standard-arm prompt mismatch: {problem_id}",
                errors,
            )
        elif row.get("arm") == "bounded_concise_prompt":
            _expect(
                row.get("student_prompt") != row.get("teacher_prompt"),
                f"Preflight concise-arm prompt was not distinct: {problem_id}",
                errors,
            )
        topk_count += int("topk_diagnostic" in row)
    _expect(
        topk_count == len(OPD_ARMS) * int(protocol["diagnostics"]["top_k_rollouts"]),
        "Preflight top-k diagnostic count mismatch.",
        errors,
    )
    try:
        recomputed = preflight_summary(rows)
        for key, value in recomputed.items():
            _expect(summary.get(key) == value, f"Preflight summary mismatch: {key}", errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"Preflight summary recomputation failed: {exc}")
    return {
        "marker_path": str(paths["marker"]),
        "marker_sha256": file_sha256(paths["marker"]),
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": file_sha256(paths["manifest"]),
        "summary_path": str(paths["summary"]),
        "summary_sha256": file_sha256(paths["summary"]),
        "rollout_count": len(rows),
    }


def _audit_training(
    protocol: Mapping[str, Any],
    checkpoint_root: Path,
    result_root: Path,
    references: List[Mapping[str, Any]],
    errors: List[str],
) -> Dict[str, Any]:
    launcher_manifest_path = result_root / "pilot/training/training_launcher_manifest.json"
    try:
        launcher = read_json(launcher_manifest_path)
        launcher_checks = {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "training_source_sha256": file_sha256(
                PROJECT_ROOT / "scripts/17_4_train_opd_policy.py"
            ),
            "launcher_source_sha256": file_sha256(
                PROJECT_ROOT / "scripts/17_4_launch_opd_training.py"
            ),
            "opd_library_sha256": file_sha256(
                PROJECT_ROOT / "src/length_budget_distill/opd.py"
            ),
        }
        _expect(
            all(launcher.get(key) == value for key, value in launcher_checks.items()),
            "Training launcher manifest mismatch.",
            errors,
        )
        _expect(
            {str(row.get("arm")) for row in launcher.get("tasks", [])} == set(OPD_ARMS)
            and all(row.get("status") == "complete" for row in launcher.get("tasks", [])),
            "Training launcher task evidence mismatch.",
            errors,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"Training launcher audit failed: {exc}")
    evidence: Dict[str, Any] = {
        "launcher_manifest_path": str(launcher_manifest_path),
        "launcher_manifest_sha256": (
            file_sha256(launcher_manifest_path) if launcher_manifest_path.is_file() else None
        ),
        "arms": {},
    }
    expected_problem_ids = {str(row["problem_id"]) for row in references}
    expected_rollouts = len(references) * int(protocol["training"]["rollouts_per_prompt"])
    expected_batches = math.ceil(
        len(references) / int(protocol["training"]["prompts_per_batch"])
    )
    reference_by_id = {str(row["problem_id"]): row for row in references}
    for arm in OPD_ARMS:
        adapter_dir = checkpoint_root / "pilot" / arm
        marker = validated_opd_adapter(protocol, arm=arm, adapter_dir=adapter_dir, stage="pilot")
        if marker is None:
            errors.append(f"Invalid OPD adapter: {adapter_dir}")
            continue
        try:
            rollout_manifest_path = Path(marker["rollout_manifest_path"])
            rollout_manifest = read_json(rollout_manifest_path)
            arm_manifest_path = result_root / "pilot/training" / arm / "training_manifest.json"
            arm_manifest = read_json(arm_manifest_path)
            training_metrics_path = adapter_dir / "training_metrics.json"
            training_metrics = read_json(training_metrics_path)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"Training evidence failed to load for {arm}: {exc}")
            continue
        _expect(rollout_manifest.get("status") == "complete", f"Rollout manifest incomplete: {arm}", errors)
        _expect(rollout_manifest.get("protocol_hash") == protocol_hash(protocol), f"Rollout protocol mismatch: {arm}", errors)
        _expect(int(rollout_manifest.get("record_count", -1)) == expected_rollouts, f"Rollout count mismatch: {arm}", errors)
        _expect(len(rollout_manifest.get("shards", [])) == expected_batches, f"Rollout shard count mismatch: {arm}", errors)
        counts: Counter[str] = Counter()
        identities = set()
        record_count = 0
        sampled_tokens = 0
        topk_count = 0
        for shard in rollout_manifest.get("shards", []):
            shard_path = Path(str(shard.get("path", "")))
            if not shard_path.is_file():
                errors.append(f"Missing rollout shard: {shard_path}")
                continue
            _expect(shard.get("sha256") == file_sha256(shard_path), f"Rollout shard hash mismatch: {shard_path}", errors)
            try:
                shard_rows = list(read_gzip_jsonl(shard_path))
            except (OSError, ValueError) as exc:
                errors.append(f"Rollout shard failed to load: {shard_path}: {exc}")
                continue
            _expect(int(shard.get("records", -1)) == len(shard_rows), f"Rollout shard record mismatch: {shard_path}", errors)
            record_count += len(shard_rows)
            for row in shard_rows:
                _audit_rollout_row(row, errors, context=f"training:{arm}")
                _expect(row.get("arm") == arm, f"Rollout arm mismatch: {shard_path}", errors)
                _expect(row.get("protocol_hash") == protocol_hash(protocol), f"Rollout protocol mismatch: {shard_path}", errors)
                problem_id = str(row.get("problem_id"))
                identity = (problem_id, int(row.get("candidate_index", -1)))
                _expect(identity not in identities, f"Duplicate rollout identity: {arm} {identity}", errors)
                identities.add(identity)
                counts[problem_id] += 1
                sampled_tokens += len(row.get("completion_token_ids", []))
                topk_count += int("topk_diagnostic" in row)
                reference = reference_by_id.get(problem_id)
                if reference is not None:
                    _expect(
                        row.get("teacher_prompt") == reference.get("standard_prompt"),
                        f"Training teacher prompt mismatch: {arm} {problem_id}",
                        errors,
                    )
                    expected_student_prompt = reference[
                        "standard_prompt" if arm == "standard_prompt" else "concise_prompt"
                    ]
                    _expect(
                        row.get("student_prompt") == expected_student_prompt,
                        f"Training student prompt mismatch: {arm} {problem_id}",
                        errors,
                    )
        _expect(record_count == expected_rollouts, f"Audited rollout count mismatch: {arm}", errors)
        _expect(set(counts) == expected_problem_ids, f"Rollout problem support mismatch: {arm}", errors)
        _expect(set(counts.values()) == {int(protocol["training"]["rollouts_per_prompt"])}, f"Per-problem rollout multiplicity mismatch: {arm}", errors)
        _expect(topk_count == int(protocol["diagnostics"]["top_k_rollouts"]), f"Training top-k diagnostic count mismatch: {arm}", errors)
        _expect(int(training_metrics.get("sampled_tokens", -1)) == sampled_tokens, f"Sampled-token count mismatch: {arm}", errors)
        _expect(int(training_metrics.get("rollouts", -1)) == record_count, f"Training-metrics rollout count mismatch: {arm}", errors)
        _expect(len(training_metrics.get("batch_metrics", [])) == expected_batches, f"Training-metrics batch count mismatch: {arm}", errors)
        _expect(
            all(int(batch.get("disabled_dropout_modules", 0)) > 0 for batch in training_metrics.get("batch_metrics", [])),
            f"Deterministic update-time dropout evidence is missing: {arm}",
            errors,
        )
        expected_optimizer_steps = sum(
            math.ceil(
                int(batch.get("rollout_count", 0))
                / int(protocol["training"]["mini_batch_rollouts"])
            )
            for batch in training_metrics.get("batch_metrics", [])
        )
        _expect(int(training_metrics.get("optimizer_steps", -1)) == expected_optimizer_steps, f"Optimizer-step count mismatch: {arm}", errors)
        registered_optimizer_steps = sum(
            math.ceil(
                min(
                    int(protocol["training"]["prompts_per_batch"]),
                    len(references) - batch_index * int(protocol["training"]["prompts_per_batch"]),
                )
                * int(protocol["training"]["rollouts_per_prompt"])
                / int(protocol["training"]["mini_batch_rollouts"])
            )
            for batch_index in range(expected_batches)
        )
        _expect(
            expected_optimizer_steps == registered_optimizer_steps,
            f"Registered optimizer-step exposure mismatch: {arm}",
            errors,
        )
        _expect(training_metrics.get("gold_labels_used_in_loss") is False, f"Gold-label loss flag mismatch: {arm}", errors)
        _expect(training_metrics.get("length_used_in_loss") is False, f"Length loss flag mismatch: {arm}", errors)
        marker_checks = {
            "adapter_model_sha256": marker["adapter_model_sha256"],
            "adapter_config_sha256": marker["adapter_config_sha256"],
            "train_manifest_sha256": marker["train_manifest_sha256"],
            "rollout_manifest_sha256": file_sha256(rollout_manifest_path),
            "protocol_hash": protocol_hash(protocol),
            "stage": "pilot",
        }
        _expect(all(arm_manifest.get(key) == value for key, value in marker_checks.items()), f"Training manifest mismatch: {arm}", errors)
        for source_path, source_hash in marker.get("source_sha256", {}).items():
            _expect(Path(source_path).is_file() and file_sha256(source_path) == source_hash, f"Training source hash mismatch: {source_path}", errors)
        evidence["arms"][arm] = {
            "adapter_path": str(adapter_dir),
            "train_marker_sha256": file_sha256(adapter_dir / "TRAIN_COMPLETE"),
            "arm_manifest_path": str(arm_manifest_path),
            "arm_manifest_sha256": file_sha256(arm_manifest_path),
            "rollout_manifest_path": str(rollout_manifest_path),
            "rollout_manifest_sha256": file_sha256(rollout_manifest_path),
            "rollouts": record_count,
            "sampled_tokens": sampled_tokens,
        }
    return evidence


def _audit_rollout_row(row: Mapping[str, Any], errors: List[str], *, context: str) -> None:
    completion = row.get("completion_token_ids", [])
    old = row.get("old_student_logprobs", [])
    teacher = row.get("teacher_logprobs", [])
    advantages = row.get("advantages", [])
    _expect(bool(completion) and len(completion) == len(old) == len(teacher) == len(advantages), f"Token/log-prob alignment mismatch: {context}", errors)
    _expect(
        all(0 <= int(token_id) < 151665 for token_id in completion),
        f"Completion token is outside the registered valid vocabulary: {context}",
        errors,
    )
    _expect(
        int(row.get("valid_vocab_size", -1)) == 151665,
        f"Valid-vocabulary evidence mismatch: {context}",
        errors,
    )
    _expect(row.get("teacher_context_mode") == "common_standard_prompt", f"Teacher context mismatch: {context}", errors)
    _expect(bool(str(row.get("teacher_prompt", "")).strip()), f"Teacher prompt is empty: {context}", errors)
    _expect(row.get("scalar_reward_used") is False, f"Scalar reward flag mismatch: {context}", errors)
    _expect(row.get("value_head_used") is False, f"Value-head flag mismatch: {context}", errors)
    _expect(row.get("correctness_is_diagnostic_only") is True, f"Correctness diagnostic flag mismatch: {context}", errors)
    _expect(row.get("length_is_diagnostic_only") is True, f"Length diagnostic flag mismatch: {context}", errors)
    try:
        finite = all(math.isfinite(float(value)) for values in (old, teacher, advantages) for value in values)
        aligned = all(abs(float(adv) - (float(tch) - float(stu))) <= 3e-6 for adv, tch, stu in zip(advantages, teacher, old))
        _expect(finite, f"Non-finite sampled-token signal: {context}", errors)
        _expect(aligned, f"Sampled-token advantage mismatch: {context}", errors)
    except (TypeError, ValueError):
        errors.append(f"Malformed sampled-token signal: {context}")
    diagnostic = row.get("topk_diagnostic")
    if diagnostic is not None:
        k = int(diagnostic.get("k", -1))
        _expect(k > 0, f"Top-k diagnostic width mismatch: {context}", errors)
        _expect(0.0 <= float(diagnostic.get("overlap_ratio", -1.0)) <= 1.0, f"Top-k overlap mismatch: {context}", errors)
        for field in ("student_ids", "student_logprobs", "teacher_ids", "teacher_logprobs"):
            values = diagnostic.get(field, [])
            _expect(
                len(values) == len(completion)
                and all(len(token_values) == k for token_values in values),
                f"Top-k diagnostic tensor shape mismatch: {context} {field}",
                errors,
            )


def _audit_evaluations(
    protocol: Mapping[str, Any],
    manifests: Mapping[str, Path],
    errors: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]]:
    evidence: Dict[str, Any] = {}
    predictions: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for split_name, manifest_path in manifests.items():
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            errors.append(f"Evaluation manifest failed: {split_name}: {exc}")
            continue
        expected = {
            "status": "complete",
            "stage": "pilot",
            "split_name": split_name,
            "prompt_mode": "common_standard_prompt",
            "protocol_hash": protocol_hash(protocol),
            "run_count": len(MODEL_IDS),
        }
        _expect(all(manifest.get(key) == value for key, value in expected.items()), f"Evaluation manifest mismatch: {split_name}", errors)
        source_checks = {
            "evaluation_source_sha256": PROJECT_ROOT / "scripts/17_5_eval_opd_model.py",
            "launcher_source_sha256": PROJECT_ROOT / "scripts/17_5_launch_opd_evaluation.py",
            "analysis_library_sha256": PROJECT_ROOT / "src/length_budget_distill/opd_analysis.py",
        }
        for field, path in source_checks.items():
            _expect(manifest.get(field) == file_sha256(path), f"Evaluation source mismatch: {split_name} {field}", errors)
        support: List[str] | None = None
        by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for run in manifest.get("runs", []):
            model_id = str(run.get("model_id", ""))
            result = completed_opd_evaluation(
                protocol,
                split_name=split_name,
                model_id=model_id,
                prediction_path=run.get("prediction_path", ""),
                summary_path=run.get("summary_path", ""),
            )
            if result is None:
                errors.append(f"Invalid evaluation artifacts: {split_name} {model_id}")
                continue
            for field in ("prediction_sha256", "summary_sha256", "prediction_count"):
                _expect(run.get(field) == result[field], f"Evaluation hash/count mismatch: {split_name} {model_id} {field}", errors)
            _expect(run.get("eval_status") in {"complete", "skipped_complete"}, f"Evaluation run incomplete: {split_name} {model_id}", errors)
            if support is not None:
                _expect(support == result["problem_ids"], f"Evaluation support mismatch: {split_name} {model_id}", errors)
            support = result["problem_ids"] if support is None else support
            rows = list(read_jsonl(Path(run["prediction_path"])))
            by_model[model_id] = {str(row["problem_id"]): dict(row) for row in rows}
        _expect(set(by_model) == set(MODEL_IDS), f"Evaluation model identities mismatch: {split_name}", errors)
        predictions[split_name] = by_model
        evidence[split_name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "models": sorted(by_model),
            "problem_count": len(support or []),
        }
    return evidence, predictions


def _audit_analysis(
    protocol: Mapping[str, Any],
    analysis_dir: Path,
    figure_dir: Path,
    predictions: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    errors: List[str],
) -> Dict[str, Any]:
    analysis_path = analysis_dir / "opd_prompt_pilot_analysis.json"
    artifact_manifest_path = analysis_dir / "analysis_artifact_manifest.json"
    marker_path = analysis_dir / "ANALYSIS_COMPLETE"
    try:
        analysis = read_json(analysis_path)
        artifact_manifest = read_json(artifact_manifest_path)
        marker = read_json(marker_path)
    except (OSError, ValueError) as exc:
        errors.append(f"Analysis validation failed: {exc}")
        return {}
    _expect(analysis.get("status") == "complete", "Analysis status mismatch.", errors)
    _expect(analysis.get("protocol_hash") == protocol_hash(protocol), "Analysis protocol mismatch.", errors)
    _expect(artifact_manifest.get("status") == "complete", "Analysis artifact manifest incomplete.", errors)
    _expect(artifact_manifest.get("protocol_hash") == protocol_hash(protocol), "Analysis artifact protocol mismatch.", errors)
    _expect(artifact_manifest.get("analysis_source_sha256") == file_sha256(PROJECT_ROOT / "scripts/17_6_analyze_opd_prompt_pilot.py"), "Analysis source hash mismatch.", errors)
    _expect(artifact_manifest.get("analysis_library_sha256") == file_sha256(PROJECT_ROOT / "src/length_budget_distill/opd_analysis.py"), "Analysis library hash mismatch.", errors)
    _expect(marker.get("status") == "complete", "Analysis marker status mismatch.", errors)
    _expect(marker.get("protocol_hash") == protocol_hash(protocol), "Analysis marker protocol mismatch.", errors)
    _expect(marker.get("artifact_manifest_sha256") == file_sha256(artifact_manifest_path), "Analysis marker manifest hash mismatch.", errors)
    artifact_count = 0
    for artifact in artifact_manifest.get("artifacts", []):
        path = Path(str(artifact.get("path", "")))
        if not path.is_file():
            errors.append(f"Missing analysis artifact: {path}")
            continue
        _expect(artifact.get("sha256") == file_sha256(path), f"Analysis artifact hash mismatch: {path}", errors)
        _expect(int(artifact.get("size_bytes", -1)) == path.stat().st_size, f"Analysis artifact size mismatch: {path}", errors)
        artifact_count += 1
    _expect(artifact_count == 11, f"Analysis artifact count mismatch: {artifact_count}", errors)
    for stem in (
        "opd_accuracy_and_output_length",
        "opd_accuracy_vs_teacher_scored_tokens",
        "opd_training_dynamics",
    ):
        for suffix in (".png", ".pdf"):
            _expect((figure_dir / f"{stem}{suffix}").is_file(), f"Missing analysis figure: {stem}{suffix}", errors)
    primary_predictions = predictions.get("primary_evaluation", {})
    if set(primary_predictions) == set(MODEL_IDS):
        recomputed = paired_opd_contrast(
            primary_predictions["bounded_concise_prompt"],
            primary_predictions["standard_prompt"],
            bootstrap_samples=int(protocol["evaluation"]["bootstrap_samples"]),
            bootstrap_seed=int(protocol["evaluation"]["bootstrap_seed"]),
        )
        stored = next(
            (
                row
                for row in analysis.get("contrasts", [])
                if row.get("split_name") == "primary_evaluation"
                and row.get("contrast_name") == "primary_prompt_arm_comparison"
            ),
            None,
        )
        if stored is None:
            errors.append("Primary OPD contrast is missing from analysis.")
        else:
            for key, value in recomputed.items():
                _expect(stored.get(key) == value, f"Primary contrast mismatch: {key}", errors)
            decision = opd_advancement_decision(recomputed, protocol["advancement_gate"])
            _expect(analysis.get("advancement_decision") == decision, "Advancement decision mismatch.", errors)
    return {
        "analysis_path": str(analysis_path),
        "analysis_sha256": file_sha256(analysis_path),
        "artifact_manifest_path": str(artifact_manifest_path),
        "artifact_manifest_sha256": file_sha256(artifact_manifest_path),
        "artifact_count": artifact_count,
    }


def _expect(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _optional_resolve(path_value: str | None, default: Path) -> Path:
    return _resolve(path_value) if path_value is not None else default


def _on_beegfs(path: Path) -> bool:
    beegfs_root = Path("/mnt/beegfs")
    return path == beegfs_root or beegfs_root in path.parents


if __name__ == "__main__":
    main()
