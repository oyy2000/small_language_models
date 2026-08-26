"""Reusable helpers for the generator-capacity by CoT-length experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .records import TraceRecord


Condition = Tuple[str, str]


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading large artifacts into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_line_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_key_value_marker(path: str | Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            if not separator or not key:
                raise ValueError(f"Malformed completion marker line in {path}: {raw_line!r}")
            values[key] = value
    return values


def validated_adapter_evidence(path: str | Path) -> Dict[str, Any] | None:
    """Validate either an SFT key-value marker or a logit-KD JSON marker."""

    root = Path(path)
    marker_path = root / "TRAIN_COMPLETE"
    adapter_config = root / "adapter_config.json"
    adapter_model = root / "adapter_model.safetensors"
    if not marker_path.is_file() or not adapter_config.is_file() or not adapter_model.is_file():
        return None
    try:
        marker = read_key_value_marker(marker_path)
    except ValueError:
        # Logit-KD adapters use a hash-bound JSON marker and train manifest.
        # Import lazily to keep the generic factorial helpers lightweight.
        from .logit_kd import validated_training_marker

        try:
            return validated_training_marker(root)
        except (OSError, ValueError):
            return None
    required_marker_fields = {
        "run_name",
        "seed",
        "train_sha256",
        "run_config_sha256",
        "training_source_sha256",
        "launcher_source_sha256",
        "adapter_config_sha256",
        "adapter_model_sha256",
    }
    if not required_marker_fields <= set(marker):
        return None
    actual = {
        "adapter_config_sha256": file_sha256(adapter_config),
        "adapter_model_sha256": file_sha256(adapter_model),
    }
    if any(marker.get(key) != value for key, value in actual.items()):
        return None
    return {**marker, **actual}


def runtime_metadata(packages: Sequence[str] | None = None) -> Dict[str, Any]:
    package_names = list(
        packages
        or ("python", "torch", "transformers", "datasets", "vllm", "trl", "peft", "numpy", "statsmodels")
    )
    versions: Dict[str, str | None] = {}
    for package_name in package_names:
        if package_name == "python":
            versions[package_name] = platform.python_version()
            continue
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    gpu_inventory: List[Dict[str, str]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", maxsplit=3)]
            if len(fields) == 4:
                gpu_inventory.append(
                    {
                        "index": fields[0],
                        "name": fields[1],
                        "memory_total_mib": fields[2],
                        "driver_version": fields[3],
                    }
                )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": versions,
        "gpus": gpu_inventory,
    }


def stable_generation_seed(
    base_seed: int,
    generator_name: str,
    budget_name: str,
    problem_id: str,
) -> int:
    identity = f"{base_seed}|{generator_name}|{budget_name}|{problem_id}".encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generator_by_name(config: Mapping[str, Any], generator_name: str) -> Dict[str, Any]:
    matches = [item for item in config.get("generators", []) if item.get("name") == generator_name]
    if len(matches) != 1:
        names = [item.get("name") for item in config.get("generators", [])]
        raise ValueError(f"Expected one generator named {generator_name!r}; available={names}")
    return dict(matches[0])


def expected_conditions(config: Mapping[str, Any]) -> List[Condition]:
    return [
        (str(generator["name"]), str(budget["name"]))
        for generator in config.get("generators", [])
        for budget in config.get("length_budgets", [])
    ]


def trace_condition(trace: TraceRecord) -> Condition:
    generator_name = trace.generator_name or str(trace.metadata.get("generator_name") or trace.teacher_model)
    return generator_name, trace.budget_name


def select_shortest_correct(traces: Iterable[TraceRecord]) -> List[TraceRecord]:
    """Select one shortest correct, budget-compliant candidate per condition/problem."""

    grouped: Dict[Tuple[str, str, str], List[TraceRecord]] = defaultdict(list)
    for trace in traces:
        generator_name, budget_name = trace_condition(trace)
        grouped[(generator_name, budget_name, trace.problem_id)].append(trace)

    selected: List[TraceRecord] = []
    for key in sorted(grouped):
        eligible = [
            trace
            for trace in grouped[key]
            if trace.is_correct
            and (
                trace.budget_compliant
                if trace.budget_compliant is not None
                else trace.solution_token_count <= trace.max_solution_tokens
            )
        ]
        if not eligible:
            continue
        winner = min(eligible, key=lambda item: (item.solution_token_count, item.candidate_index, item.trace_id))
        selected.append(replace(winner, selected_for_sft=True))
    return selected


def common_problem_ids(
    selected: Iterable[TraceRecord],
    conditions: Sequence[Condition],
) -> List[str]:
    by_condition: Dict[Condition, set[str]] = {condition: set() for condition in conditions}
    for trace in selected:
        condition = trace_condition(trace)
        if condition in by_condition:
            by_condition[condition].add(trace.problem_id)
    if not by_condition:
        return []
    return sorted(set.intersection(*by_condition.values()))


def selected_by_condition(
    selected: Iterable[TraceRecord],
    problem_ids: Iterable[str],
) -> Dict[Condition, List[TraceRecord]]:
    allowed = set(problem_ids)
    grouped: Dict[Condition, List[TraceRecord]] = defaultdict(list)
    seen: set[Tuple[Condition, str]] = set()
    for trace in selected:
        if trace.problem_id not in allowed:
            continue
        condition = trace_condition(trace)
        identity = (condition, trace.problem_id)
        if identity in seen:
            raise ValueError(f"Duplicate selected trace for condition/problem: {identity}")
        seen.add(identity)
        grouped[condition].append(trace)
    for condition in grouped:
        grouped[condition].sort(key=lambda item: item.problem_id)
    return dict(grouped)


def deterministic_equal_token_subset(
    traces: Sequence[TraceRecord],
    target_tokens: int,
    seed: int,
) -> Tuple[List[TraceRecord], int]:
    """Choose whole traces without exceeding a token target."""

    shuffled = list(traces)
    random.Random(seed).shuffle(shuffled)
    chosen: List[TraceRecord] = []
    total = 0
    for trace in shuffled:
        token_count = int(trace.solution_token_count)
        if total + token_count <= target_tokens:
            chosen.append(trace)
            total += token_count
    chosen.sort(key=lambda item: item.problem_id)
    return chosen, total


def candidate_audit_rows(
    traces: Iterable[TraceRecord],
    conditions: Sequence[Condition],
) -> List[Dict[str, Any]]:
    grouped: Dict[Condition, List[TraceRecord]] = defaultdict(list)
    for trace in traces:
        grouped[trace_condition(trace)].append(trace)
    rows = []
    for condition in conditions:
        bucket = grouped.get(condition, [])
        problems = {trace.problem_id for trace in bucket}
        correct = [trace for trace in bucket if trace.is_correct]
        compliant = [
            trace
            for trace in bucket
            if trace.budget_compliant
            if trace.budget_compliant is not None
        ]
        eligible_problems = {
            trace.problem_id
            for trace in bucket
            if trace.is_correct
            and (
                trace.budget_compliant
                if trace.budget_compliant is not None
                else trace.solution_token_count <= trace.max_solution_tokens
            )
        }
        rows.append(
            {
                "generator_name": condition[0],
                "budget_name": condition[1],
                "candidate_count": len(bucket),
                "problem_count": len(problems),
                "correct_candidate_count": len(correct),
                "compliant_candidate_count": len(compliant),
                "pass_at_3_problem_count": len(eligible_problems),
                "pass_at_3": len(eligible_problems) / len(problems) if problems else 0.0,
            }
        )
    return rows
