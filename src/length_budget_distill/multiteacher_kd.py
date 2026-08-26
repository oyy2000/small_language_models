"""Data construction helpers for multi-benchmark, multi-teacher KD pilots."""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .factorial import (
    canonical_sha256,
    deterministic_equal_token_subset,
    expected_conditions,
    file_sha256,
    read_key_value_marker,
    validated_adapter_evidence,
)
from .logit_kd import validated_training_marker
from .math_mix import stable_mixed_sft_order, tag_sft_record
from .records import read_jsonl, trace_from_dict, write_jsonl
from .sft_format import trace_to_sft_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
Condition = Tuple[str, str]


def build_multiteacher_equal_token_data(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Build one GSM8K+MATH equal-token dataset per teacher-length condition."""

    config_payload = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(config_payload)
    conditions = expected_conditions(config)
    generator_names = [str(item["name"]) for item in config["generators"]]
    budget_names = [str(item["name"]) for item in config["length_budgets"]]
    expected = {(generator, budget) for generator in generator_names for budget in budget_names}
    if set(conditions) != expected or len(conditions) != 12:
        raise ValueError(f"Expected a four-teacher by three-length matrix, got {conditions}")
    seeds = [int(seed) for seed in config["balancing"]["training_seeds"]]
    if seeds != [17]:
        raise ValueError(f"The pilot is restricted to training seed 17, got {seeds}")
    seed = seeds[0]

    resolved_output = Path(output_dir)
    marker_path = resolved_output / "DATASETS_COMPLETE"
    manifest_path = resolved_output / "dataset_manifest.json"
    if marker_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing data artifacts in {resolved_output}")

    math_by_condition: Dict[Condition, List[Any]] = {}
    math_inputs: Dict[str, Any] = {}
    common_counts: Dict[str, int] = {}
    selected_root = resolved_output.parent / "selected"
    minimum_common = int(config["balancing"]["pilot_min_common_problems"])
    for generator_name in generator_names:
        paths = _math_selection_paths(config, selected_root, generator_name)
        audit = _read_json(paths["audit"])
        if audit.get("status") != "passed":
            raise ValueError(f"MATH selection audit is not passed for {generator_name}: {paths['audit']}")
        if generator_name != "qwen2p5_7b" and audit.get("config_hash") != config_hash:
            raise ValueError(f"MATH selection audit config mismatch for {generator_name}")
        observed_conditions = {
            (str(row["generator_name"]), str(row["budget_name"]))
            for row in audit.get("conditions", [])
        }
        expected_generator_conditions = {(generator_name, budget) for budget in budget_names}
        if observed_conditions != expected_generator_conditions:
            raise ValueError(
                f"MATH selection condition mismatch for {generator_name}: {sorted(observed_conditions)}"
            )
        if file_sha256(paths["selected"]) != audit.get("selected_sha256"):
            raise ValueError(f"Selected-trace hash mismatch for {generator_name}")
        if file_sha256(paths["common"]) != audit.get("common_problem_ids_sha256"):
            raise ValueError(f"Common-support hash mismatch for {generator_name}")
        common_payload = _read_json(paths["common"])
        common_ids = [str(value) for value in common_payload["problem_ids"]]
        common_count = int(audit.get("common_problem_count", -1))
        if (
            common_count < minimum_common
            or len(common_ids) != common_count
            or len(common_ids) != len(set(common_ids))
        ):
            raise ValueError(
                f"MATH common-support gate failed for {generator_name}: "
                f"actual={common_count} required={minimum_common}"
            )
        common_set = set(common_ids)
        grouped = {(generator_name, budget): [] for budget in budget_names}
        for row in read_jsonl(paths["selected"]):
            trace = trace_from_dict(row)
            condition = (str(trace.generator_name), str(trace.budget_name))
            if condition in grouped and trace.problem_id in common_set:
                grouped[condition].append(trace)
        for condition, traces in grouped.items():
            traces.sort(key=lambda item: item.problem_id)
            if len(traces) != common_count or {item.problem_id for item in traces} != common_set:
                raise ValueError(f"Incomplete within-teacher MATH common matrix: {condition}")
            math_by_condition[condition] = traces
        common_counts[generator_name] = common_count
        math_inputs[generator_name] = {
            "selected_traces": _file_evidence(paths["selected"]),
            "selection_audit": _file_evidence(paths["audit"]),
            "common_problem_ids": _file_evidence(paths["common"]),
            "common_problem_count": common_count,
            "source_experiment": (
                "capacity_length_math_mix_pilot_v1"
                if generator_name == "qwen2p5_7b"
                else str(config["experiment_name"])
            ),
        }

    math_condition_totals = {
        condition: sum(int(trace.solution_token_count) for trace in traces)
        for condition, traces in math_by_condition.items()
    }
    math_target = min(math_condition_totals.values())
    max_gap = int(config["balancing"].get("max_equal_token_gap", 512))

    gsm_manifest_path = _resolve(str(config["source_artifacts"]["gsm8k_dataset_manifest"]))
    gsm_manifest = _read_json(gsm_manifest_path)
    if (
        gsm_manifest.get("status") != "complete"
        or gsm_manifest.get("training_seeds") != [17]
        or int(gsm_manifest.get("common_problem_count", -1)) != 881
    ):
        raise ValueError("Expected the complete 881-problem seed-17 GSM8K dataset manifest")
    gsm_runs = _index_equal_token_gsm_runs(gsm_manifest.get("runs", []), expected)
    registered_gsm_target = int(
        config["balancing"]["gsm8k_registered_solution_token_target"]
    )

    generators = {str(item["name"]): dict(item) for item in config["generators"]}
    run_entries: List[Dict[str, Any]] = []
    for generator_name, budget_name in conditions:
        condition = (generator_name, budget_name)
        subset_seed = int(
            canonical_sha256(
                [config_hash, "global_math_equal_token", generator_name, budget_name, seed]
            )[:8],
            16,
        )
        math_subset, math_tokens = deterministic_equal_token_subset(
            math_by_condition[condition],
            target_tokens=math_target,
            seed=subset_seed,
        )
        math_gap = math_target - math_tokens
        if not 0 <= math_gap <= max_gap:
            raise ValueError(
                f"MATH equal-token gap exceeds tolerance for {condition}: "
                f"target={math_target} actual={math_tokens} gap={math_gap}"
            )
        math_records = [
            tag_sft_record(
                trace_to_sft_record(trace), source="hendrycks_math", id_prefix="math::"
            )
            for trace in math_subset
        ]

        gsm_run = gsm_runs[condition]
        gsm_path = _resolve(str(gsm_run["train_path"]))
        if file_sha256(gsm_path) != gsm_run.get("train_sha256"):
            raise ValueError(f"GSM8K training-data hash mismatch: {gsm_path}")
        gsm_tokens = int(gsm_run["supervised_tokens"])
        if abs(gsm_tokens - registered_gsm_target) > max_gap:
            raise ValueError(
                f"GSM8K token total violates registered target for {condition}: {gsm_tokens}"
            )
        gsm_rows = list(read_jsonl(gsm_path))
        if len(gsm_rows) != int(gsm_run["n"]):
            raise ValueError(f"GSM8K row-count mismatch: {gsm_path}")
        _validate_source_rows(gsm_rows, generator_name, budget_name, source="gsm8k")
        gsm_records = [
            tag_sft_record(row, source="gsm8k", id_prefix="gsm8k::") for row in gsm_rows
        ]
        combined = stable_mixed_sft_order(
            [*gsm_records, *math_records],
            config_hash=config_hash,
            mode="equal_token",
            generator_name=generator_name,
            budget_name=budget_name,
            seed=seed,
        )
        output_path = (
            resolved_output
            / "equal_token"
            / f"{generator_name}__{budget_name}__seed_{seed}.jsonl"
        )
        write_jsonl(output_path, combined)
        total_tokens = gsm_tokens + math_tokens
        generator = generators[generator_name]
        run_entries.append(
            {
                "run_name": (
                    f"multibench__equal_token__{generator_name}__{budget_name}__seed_{seed}"
                ),
                "training_variant": "gsm8k_math_multibench",
                "mode": "equal_token",
                "generator_name": generator_name,
                "generator_model_name": str(generator["model_name"]),
                "generator_revision": str(generator["revision"]),
                "generator_size_b": float(generator["size_b"]),
                "self_distillation_control": generator_name == "qwen2p5_1p5b",
                "budget_name": budget_name,
                "seed": seed,
                "train_path": str(output_path),
                "train_sha256": file_sha256(output_path),
                "n": len(combined),
                "supervised_tokens": total_tokens,
                "source_counts": {
                    "gsm8k": len(gsm_records),
                    "hendrycks_math": len(math_records),
                },
                "source_supervised_tokens": {
                    "gsm8k": gsm_tokens,
                    "hendrycks_math": math_tokens,
                },
                "math_equal_token_target": math_target,
                "math_equal_token_gap": math_gap,
                "math_common_problem_count": common_counts[generator_name],
            }
        )

    if len(run_entries) != len(conditions):
        raise RuntimeError(
            f"Dataset manifest size mismatch: expected={len(conditions)} actual={len(run_entries)}"
        )
    total_tokens = [int(run["supervised_tokens"]) for run in run_entries]
    allowed_total_gap = max_gap + max(
        abs(int(run["source_supervised_tokens"]["gsm8k"]) - registered_gsm_target)
        for run in run_entries
    )
    if max(total_tokens) - min(total_tokens) > allowed_total_gap:
        raise ValueError(
            "Cross-condition mixed supervision totals exceed the registered whole-record gap: "
            f"min={min(total_tokens)} max={max(total_tokens)} allowed={allowed_total_gap}"
        )

    manifest = {
        "status": "complete",
        "stage": "pilot",
        "evidence_level": str(config["evidence_level"]),
        "protocol_variant": str(config["protocol_variant"]),
        "config_hash": config_hash,
        "training_seeds": [seed],
        "support_scope": str(config["balancing"]["support_scope"]),
        "expected_run_count": len(conditions),
        "math_global_equal_token_target": math_target,
        "mixed_supervision_token_range": {
            "minimum": min(total_tokens),
            "maximum": max(total_tokens),
            "gap": max(total_tokens) - min(total_tokens),
            "allowed_gap": allowed_total_gap,
        },
        "math_condition_token_totals_before_balancing": [
            {
                "generator_name": condition[0],
                "budget_name": condition[1],
                "supervised_tokens": math_condition_totals[condition],
            }
            for condition in conditions
        ],
        "inputs": {
            "gsm8k_dataset_manifest": _file_evidence(gsm_manifest_path),
            "math_selection_by_generator": math_inputs,
        },
        "runs": run_entries,
    }
    _write_json(manifest_path, manifest)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"config_hash={config_hash}\nrun_count={len(run_entries)}\n"
        f"math_global_equal_token_target={math_target}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\n",
        encoding="utf-8",
    )
    return manifest


def freeze_multiteacher_kd_protocol(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    dataset_manifest_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Bind completed mixed datasets to inherited 7B KD hyperparameters and model hashes."""

    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = PROJECT_ROOT / resolved_config_path
    config_payload = {key: value for key, value in config.items() if key != "_config_path"}
    config_hash = canonical_sha256(config_payload)
    dataset_path = Path(dataset_manifest_path)
    dataset_manifest = _read_json(dataset_path)
    if (
        dataset_manifest.get("status") != "complete"
        or dataset_manifest.get("config_hash") != config_hash
        or int(dataset_manifest.get("expected_run_count", -1)) != 12
    ):
        raise ValueError("The multi-teacher mixed dataset manifest is incomplete or unbound")

    artifacts = config["source_artifacts"]
    inherited_config_path = _resolve(str(artifacts["inherited_kd_config"]))
    inherited_protocol = _read_json(inherited_config_path)
    inherited_selection_path = _resolve(str(artifacts["inherited_kd_selection"]))
    inherited_selection = _read_json(inherited_selection_path)
    inherited_marker_path = _resolve(str(artifacts["inherited_kd_completion_marker"]))
    inherited_marker = _read_json(inherited_marker_path)
    inherited_protocol_hash = canonical_sha256(inherited_protocol)
    if inherited_selection.get("status") != "complete":
        raise ValueError("Inherited equal-token KD hyperparameter selection is incomplete")
    if inherited_selection.get("protocol_hash") != inherited_protocol_hash:
        raise ValueError("Inherited KD selection is bound to another protocol")
    if (
        inherited_marker.get("status") != "complete"
        or inherited_marker.get("protocol_hash") != inherited_protocol_hash
    ):
        raise ValueError("Inherited equal-token KD completion marker is missing or inconsistent")
    inherited_audit_path = Path(str(inherited_marker["completion_audit_path"]))
    if file_sha256(inherited_audit_path) != inherited_marker.get("completion_audit_sha256"):
        raise ValueError("Inherited equal-token KD completion-audit hash mismatch")
    inherited_audit = _read_json(inherited_audit_path)
    if inherited_audit.get("status") != "passed":
        raise ValueError("Inherited equal-token KD completion audit did not pass")

    if dict(config["training"]) != dict(inherited_protocol["training"]):
        raise ValueError("Pilot training settings drifted from the inherited equal-token KD protocol")
    inherited_student = inherited_protocol["models"]["student"]
    configured_student = config["student"]
    for key in ("model_name", "revision", "torch_dtype", "use_lora", "lora"):
        if configured_student.get(key) != inherited_student.get(key):
            raise ValueError(f"Pilot student setting drifted from inherited KD: {key}")

    run_index = {
        (str(run["generator_name"]), str(run["budget_name"])): run
        for run in dataset_manifest["runs"]
    }
    expected_conditions = {
        (str(generator["name"]), str(budget["name"]))
        for generator in config["generators"]
        for budget in config["length_budgets"]
    }
    if set(run_index) != expected_conditions:
        raise ValueError("Mixed dataset manifest does not cover the registered teacher-length matrix")
    generators = {str(item["name"]): dict(item) for item in config["generators"]}
    budget_limits = {
        str(item["name"]): int(item["max_solution_tokens"])
        for item in config["length_budgets"]
    }
    conditions = []
    for generator_name, budget_name in sorted(
        expected_conditions,
        key=lambda value: (
            [str(item["name"]) for item in config["generators"]].index(value[0]),
            [str(item["name"]) for item in config["length_budgets"]].index(value[1]),
        ),
    ):
        run = run_index[(generator_name, budget_name)]
        generator = generators[generator_name]
        conditions.append(
            {
                "condition_id": f"{generator_name}__{budget_name}",
                "generator_name": generator_name,
                "teacher": {
                    "model_name": str(generator["model_name"]),
                    "revision": str(generator["revision"]),
                    "torch_dtype": str(generator["dtype"]),
                },
                "teacher_size_b": float(generator["size_b"]),
                "self_distillation_control": generator_name == "qwen2p5_1p5b",
                "budget_name": budget_name,
                "max_solution_tokens": budget_limits[budget_name],
                "train_path": str(run["train_path"]),
                "train_sha256": str(run["train_sha256"]),
                "expected_records": int(run["n"]),
                "expected_solution_tokens": int(run["supervised_tokens"]),
                "expected_dataset_sources": ["gsm8k", "hendrycks_math"],
                "expected_source_counts": dict(run["source_counts"]),
                "expected_source_solution_tokens": dict(run["source_supervised_tokens"]),
            }
        )

    eval_manifest_path = _resolve(str(artifacts["evaluation_suite_manifest"]))
    eval_manifest = _read_json(eval_manifest_path)
    if eval_manifest.get("status") != "complete" or len(eval_manifest.get("datasets", [])) != 3:
        raise ValueError("The inherited multi-benchmark evaluation suite is incomplete")
    for dataset in eval_manifest["datasets"]:
        dataset_file = Path(str(dataset["path"]))
        if file_sha256(dataset_file) != dataset["sha256"]:
            raise ValueError(f"Evaluation dataset hash mismatch: {dataset_file}")

    frozen = {
        "experiment_name": str(config["experiment_name"]),
        "protocol_variant": str(config["protocol_variant"]),
        "evidence_level": str(config["evidence_level"]),
        "config": {
            "path": str(resolved_config_path),
            "file_sha256": file_sha256(resolved_config_path),
            "canonical_sha256": config_hash,
        },
        "dataset_manifest": _file_evidence(dataset_path),
        "inherited_kd": {
            "config": _file_evidence(inherited_config_path),
            "protocol_hash": inherited_protocol_hash,
            "selection": _file_evidence(inherited_selection_path),
            "completion_marker": _file_evidence(inherited_marker_path),
            "completion_audit": _file_evidence(inherited_audit_path),
            "selected_alpha": float(inherited_selection["selected_alpha"]),
            "selected_temperature": float(inherited_selection["selected_temperature"]),
            "selection_rule": inherited_selection["selection_rule"],
        },
        "models": {
            "student": inherited_student,
            "tokenizer": inherited_protocol["models"]["tokenizer"],
        },
        "conditions": conditions,
        "kd": {
            "loss_direction": str(config["kd"]["loss_direction"]),
            "completion_only": True,
            "alpha": float(inherited_selection["selected_alpha"]),
            "temperature": float(inherited_selection["selected_temperature"]),
            "hard_target_sft_baseline": True,
        },
        "training": dict(config["training"]),
        "evaluation_suite_manifest": _file_evidence(eval_manifest_path),
        "outputs": dict(config["outputs"]),
    }
    resolved_output_path = Path(output_path)
    marker_path = resolved_output_path.with_name("KD_PROTOCOL_FROZEN")
    if resolved_output_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen KD protocol: {resolved_output_path}")
    _write_json(resolved_output_path, frozen)
    marker_path.write_text(
        f"protocol_sha256={file_sha256(resolved_output_path)}\n"
        f"config_hash={config_hash}\nconditions={len(conditions)}\n",
        encoding="utf-8",
    )
    return frozen


def build_multiteacher_model_registry(
    frozen: Mapping[str, Any],
    *,
    sft_manifest_glob: str,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Register the base, matched SFT, and matched-teacher KD adapters for evaluation."""

    manifest_paths = [Path(value) for value in sorted(glob.glob(sft_manifest_glob))]
    if not manifest_paths:
        raise FileNotFoundError(f"No SFT training manifests matched {sft_manifest_glob!r}")
    sft_runs: Dict[Condition, Mapping[str, Any]] = {}
    sft_inputs = []
    for path in manifest_paths:
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            raise ValueError(f"SFT training manifest is incomplete: {path}")
        sft_inputs.append(_file_evidence(path))
        for run in manifest.get("runs", []):
            if run.get("mode") != "equal_token" or int(run.get("seed", -1)) != 17:
                continue
            condition = (str(run["generator_name"]), str(run["budget_name"]))
            if condition in sft_runs:
                raise ValueError(f"Duplicate SFT condition across manifests: {condition}")
            sft_runs[condition] = run

    registered_conditions = {
        (str(item["generator_name"]), str(item["budget_name"])): item
        for item in frozen["conditions"]
    }
    if set(sft_runs) != set(registered_conditions):
        raise ValueError("SFT manifests do not cover the frozen teacher-length matrix")

    student_name = str(frozen["models"]["student"]["model_name"])
    checkpoint_root = _resolve(str(frozen["outputs"]["checkpoint_root"]))
    models: List[Dict[str, Any]] = [
        {
            "model_id": "base_qwen2p5_1p5b_instruct",
            "adapter_path": None,
            "metadata": {
                "training_variant": "base",
                "method": "base",
                "generator_name": None,
                "budget_name": None,
                "seed": None,
            },
        }
    ]
    condition_order = [
        (str(item["generator_name"]), str(item["budget_name"]))
        for item in frozen["conditions"]
    ]
    for method in ("sft", "logit_kd"):
        for condition in condition_order:
            frozen_condition = registered_conditions[condition]
            generator_name, budget_name = condition
            if method == "sft":
                run = sft_runs[condition]
                adapter_path = Path(str(run["output_dir"]))
                evidence = validated_adapter_evidence(adapter_path)
            else:
                adapter_path = (
                    checkpoint_root
                    / "logit_kd"
                    / f"{frozen_condition['condition_id']}__seed_17"
                )
                evidence = validated_training_marker(adapter_path)
            if evidence is None:
                raise FileNotFoundError(f"Invalid completed {method} adapter: {adapter_path}")
            models.append(
                {
                    "model_id": f"{method}__{generator_name}__{budget_name}__seed_17",
                    "adapter_path": str(adapter_path),
                    "metadata": {
                        "training_variant": "gsm8k_math_multibench",
                        "method": method,
                        "mode": "equal_token",
                        "generator_name": generator_name,
                        "generator_size_b": frozen_condition["teacher_size_b"],
                        "self_distillation_control": frozen_condition[
                            "self_distillation_control"
                        ],
                        "budget_name": budget_name,
                        "seed": 17,
                    },
                }
            )
    if len(models) != 25 or len({item["model_id"] for item in models}) != 25:
        raise RuntimeError(f"Expected 25 unique evaluation models, got {len(models)}")
    registry = {
        "status": "complete",
        "experiment_name": str(frozen["experiment_name"]),
        "model_name": student_name,
        "model_count": len(models),
        "sft_training_manifests": sft_inputs,
        "models": models,
    }
    resolved_output = Path(output_path)
    if resolved_output.exists():
        raise FileExistsError(f"Refusing to overwrite model registry: {resolved_output}")
    _write_json(resolved_output, registry)
    return registry


def audit_multiteacher_kd_completion(
    *,
    config_path: str | Path,
    output_root: str | Path,
    checkpoint_root: str | Path,
    figure_root: str | Path,
    output_json: str | Path,
) -> Dict[str, Any]:
    """Independently audit the complete exploratory artifact contract."""

    errors: List[str] = []
    resolved_config = _resolve(str(config_path))
    config = _read_json(resolved_config)
    config_hash = canonical_sha256(config)
    result_root = _resolve(str(output_root)) / "pilot"
    checkpoints = _resolve(str(checkpoint_root))
    figures = _resolve(str(figure_root))

    data_manifest_path = result_root / "sft_data" / "dataset_manifest.json"
    data_manifest = _safe_json(data_manifest_path, errors, "dataset manifest")
    if data_manifest:
        if data_manifest.get("status") != "complete" or data_manifest.get("config_hash") != config_hash:
            errors.append("dataset manifest status/config hash mismatch")
        runs = data_manifest.get("runs", [])
        if len(runs) != 12:
            errors.append(f"expected 12 mixed datasets, found {len(runs)}")
        identities = set()
        for run in runs:
            identity = (str(run.get("generator_name")), str(run.get("budget_name")))
            identities.add(identity)
            train_path = _resolve(str(run.get("train_path", "")))
            if not train_path.is_file() or file_sha256(train_path) != run.get("train_sha256"):
                errors.append(f"mixed dataset file/hash mismatch: {identity}")
        if len(identities) != 12:
            errors.append("mixed dataset identities are duplicate or incomplete")

    frozen_path = result_root / "frozen_kd_protocol.json"
    frozen = _safe_json(frozen_path, errors, "frozen KD protocol")
    frozen_marker_path = result_root / "KD_PROTOCOL_FROZEN"
    if frozen:
        if len(frozen.get("conditions", [])) != 12:
            errors.append("frozen KD protocol does not contain 12 conditions")
        if frozen.get("config", {}).get("canonical_sha256") != config_hash:
            errors.append("frozen KD protocol config hash mismatch")
    if frozen_marker_path.is_file() and frozen_path.is_file():
        marker = read_key_value_marker(frozen_marker_path)
        if marker.get("protocol_sha256") != file_sha256(frozen_path):
            errors.append("frozen KD protocol marker hash mismatch")
    else:
        errors.append("missing frozen KD protocol completion marker")

    sft_manifest_paths = [
        Path(value)
        for value in sorted(
            glob.glob(str(result_root / "training" / "training_manifest_shard_*_of_*.json"))
        )
    ]
    sft_conditions = set()
    for path in sft_manifest_paths:
        manifest = _safe_json(path, errors, f"SFT manifest {path.name}")
        if not manifest:
            continue
        if manifest.get("status") != "complete":
            errors.append(f"incomplete SFT training manifest: {path}")
        for run in manifest.get("runs", []):
            condition = (str(run.get("generator_name")), str(run.get("budget_name")))
            sft_conditions.add(condition)
            if validated_adapter_evidence(str(run.get("output_dir", ""))) is None:
                errors.append(f"invalid SFT adapter: {condition}")
    if len(sft_manifest_paths) != 2 or len(sft_conditions) != 12:
        errors.append(
            f"SFT coverage mismatch: manifests={len(sft_manifest_paths)} conditions={len(sft_conditions)}"
        )

    kd_conditions = set()
    if frozen:
        for condition in frozen.get("conditions", []):
            condition_id = str(condition["condition_id"])
            adapter_path = checkpoints / "logit_kd" / f"{condition_id}__seed_17"
            marker = validated_training_marker(adapter_path)
            if marker is None:
                errors.append(f"invalid logit-KD adapter: {condition_id}")
                continue
            if marker.get("frozen_protocol_sha256") != file_sha256(frozen_path):
                errors.append(f"logit-KD protocol hash mismatch: {condition_id}")
            kd_conditions.add(condition_id)
    if len(kd_conditions) != 12:
        errors.append(f"logit-KD coverage mismatch: conditions={len(kd_conditions)}")

    registry = _safe_json(result_root / "model_registry.json", errors, "model registry")
    if registry and (registry.get("status") != "complete" or len(registry.get("models", [])) != 25):
        errors.append("model registry status/cardinality mismatch")

    eval_manifest_paths = [
        Path(value)
        for value in sorted(glob.glob(str(result_root / "eval" / "model_manifests" / "*.json")))
    ]
    eval_artifact_count = 0
    for path in eval_manifest_paths:
        manifest = _safe_json(path, errors, f"evaluation manifest {path.name}")
        if not manifest or manifest.get("status") != "complete":
            errors.append(f"incomplete evaluation manifest: {path}")
            continue
        artifacts = manifest.get("artifacts", [])
        if len(artifacts) != 3:
            errors.append(f"evaluation dataset count mismatch: {path}")
        for artifact in artifacts:
            prediction_path = Path(str(artifact.get("prediction_path", "")))
            summary_path = Path(str(artifact.get("summary_path", "")))
            if (
                not prediction_path.is_file()
                or file_sha256(prediction_path) != artifact.get("prediction_sha256")
                or not summary_path.is_file()
                or file_sha256(summary_path) != artifact.get("summary_sha256")
            ):
                errors.append(f"evaluation artifact hash mismatch: {path.name}")
            eval_artifact_count += 1
    if len(eval_manifest_paths) != 25 or eval_artifact_count != 75:
        errors.append(
            f"evaluation coverage mismatch: models={len(eval_manifest_paths)} artifacts={eval_artifact_count}"
        )

    analysis_manifest = _safe_json(
        result_root / "analysis" / "analysis_artifact_manifest.json",
        errors,
        "analysis manifest",
    )
    analysis_artifact_count = 0
    if analysis_manifest:
        if analysis_manifest.get("status") != "complete":
            errors.append("analysis manifest is incomplete")
        for artifact in analysis_manifest.get("artifacts", []):
            path = Path(str(artifact.get("path", "")))
            if not path.is_file() or file_sha256(path) != artifact.get("sha256"):
                errors.append(f"analysis artifact hash mismatch: {path}")
            analysis_artifact_count += 1
    if analysis_artifact_count != 8:
        errors.append(f"expected 8 analysis artifacts, found {analysis_artifact_count}")
    for filename in (
        "multibench_accuracy_by_teacher_length.png",
        "multibench_accuracy_by_teacher_length.pdf",
        "kd_minus_sft_accuracy_heatmap.png",
        "kd_minus_sft_accuracy_heatmap.pdf",
    ):
        if not (figures / filename).is_file():
            errors.append(f"missing publication figure: {figures / filename}")

    audit = {
        "status": "passed" if not errors else "failed",
        "evidence_level": "exploratory_single_seed_pilot",
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "config_canonical_sha256": config_hash,
        "dataset_count": len(data_manifest.get("runs", [])) if data_manifest else 0,
        "sft_adapter_count": len(sft_conditions),
        "logit_kd_adapter_count": len(kd_conditions),
        "evaluation_model_count": len(eval_manifest_paths),
        "evaluation_artifact_count": eval_artifact_count,
        "analysis_artifact_count": analysis_artifact_count,
        "errors": errors,
    }
    audit_path = Path(output_json)
    _write_json(audit_path, audit)
    if errors:
        return audit
    completion_marker = _resolve(str(output_root)) / "PILOT_COMPLETE"
    _write_json(
        completion_marker,
        {
            "status": "complete",
            "evidence_level": "exploratory_single_seed_pilot",
            "completion_audit_path": str(audit_path),
            "completion_audit_sha256": file_sha256(audit_path),
            "sft_adapters": 12,
            "logit_kd_adapters": 12,
            "evaluation_models": 25,
            "evaluation_artifacts": 75,
        },
    )
    return audit


def _math_selection_paths(
    config: Mapping[str, Any],
    selected_root: Path,
    generator_name: str,
) -> Dict[str, Path]:
    if generator_name == "qwen2p5_7b":
        artifacts = config["source_artifacts"]
        return {
            "selected": _resolve(str(artifacts["qwen2p5_7b_math_selected_traces"])),
            "audit": _resolve(str(artifacts["qwen2p5_7b_math_selection_audit"])),
            "common": _resolve(str(artifacts["qwen2p5_7b_math_common_problem_ids"])),
        }
    root = selected_root / generator_name
    return {
        "selected": root / "selected_traces.jsonl",
        "audit": root / "selection_audit.json",
        "common": root / "common_problem_ids.json",
    }


def _index_equal_token_gsm_runs(
    runs: Iterable[Mapping[str, Any]],
    expected_conditions: Sequence[Condition] | set[Condition],
) -> Dict[Condition, Mapping[str, Any]]:
    indexed: Dict[Condition, Mapping[str, Any]] = {}
    for run in runs:
        if run.get("mode") != "equal_token" or int(run.get("seed", -1)) != 17:
            continue
        condition = (str(run["generator_name"]), str(run["budget_name"]))
        if condition in indexed:
            raise ValueError(f"Duplicate GSM8K equal-token run: {condition}")
        indexed[condition] = run
    if set(indexed) != set(expected_conditions):
        raise ValueError(
            "GSM8K equal-token condition mismatch: "
            f"missing={sorted(set(expected_conditions) - set(indexed))} "
            f"unexpected={sorted(set(indexed) - set(expected_conditions))}"
        )
    return indexed


def _validate_source_rows(
    rows: Iterable[Mapping[str, Any]],
    generator_name: str,
    budget_name: str,
    *,
    source: str,
) -> None:
    seen = set()
    for row in rows:
        record_id = str(row.get("id", ""))
        metadata = row.get("metadata", {})
        if not record_id or record_id in seen:
            raise ValueError(f"Missing or duplicate {source} record ID: {record_id!r}")
        seen.add(record_id)
        if (
            metadata.get("generator_name") != generator_name
            or metadata.get("budget_name") != budget_name
            or not bool(metadata.get("is_correct"))
            or not bool(metadata.get("budget_compliant"))
        ):
            raise ValueError(f"Invalid {source} source metadata: {record_id}")


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_evidence(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _safe_json(path: Path, errors: List[str], label: str) -> Dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label}: {path}: {exc}")
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
