#!/usr/bin/env python3
"""Merge all OPD preflight shards and apply the registered global gate."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.opd import (
    OPD_ARMS,
    preflight_summary,
    protocol_hash,
    read_gzip_jsonl,
    read_json,
    validate_opd_protocol,
    write_gzip_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    input_dir = _resolve(args.input_dir)
    output_dir = _resolve(args.output_dir)
    rows_path = output_dir / "preflight_rollouts.jsonl.gz"
    summary_path = output_dir / "preflight_summary.json"
    manifest_path = output_dir / "preflight_manifest.json"
    marker_path = output_dir / "PREFLIGHT_COMPLETE"
    final_paths = (rows_path, summary_path, manifest_path, marker_path)
    if all(path.is_file() for path in final_paths) and args.skip_complete:
        if _completed(protocol, rows_path, summary_path, manifest_path, marker_path):
            logging.info("opd_preflight_already_complete output=%s", output_dir)
            return
        raise ValueError("Existing merged OPD preflight evidence failed validation.")
    if any(path.exists() for path in final_paths):
        raise FileExistsError("Refusing to overwrite merged OPD preflight evidence.")

    num_shards = int(protocol["preflight"]["num_shards"])
    rows: List[Dict[str, Any]] = []
    shard_evidence = []
    for shard_index in range(num_shards):
        suffix = f"shard_{shard_index:05d}_of_{num_shards:05d}"
        shard_manifest_path = input_dir / "manifests" / f"{suffix}.json"
        shard = read_json(shard_manifest_path)
        expected = {
            "status": "complete",
            "stage": "preflight_shard",
            "protocol_hash": protocol_hash(protocol),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "source_sha256": file_sha256(
                PROJECT_ROOT / "scripts/17_3_preflight_opd_signal.py"
            ),
        }
        if any(shard.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Preflight shard manifest mismatch: {shard_manifest_path}")
        shard_path = Path(str(shard["rollout_path"]))
        if not shard_path.is_file() or shard.get("rollout_sha256") != file_sha256(shard_path):
            raise ValueError(f"Preflight shard data/hash mismatch: {shard_path}")
        shard_rows = list(read_gzip_jsonl(shard_path))
        if len(shard_rows) != int(shard["rollout_count"]):
            raise ValueError(f"Preflight shard record-count mismatch: {shard_path}")
        rows.extend(shard_rows)
        shard_evidence.append(
            {
                "manifest_path": str(shard_manifest_path),
                "manifest_sha256": file_sha256(shard_manifest_path),
                "rollout_path": str(shard_path),
                "rollout_sha256": file_sha256(shard_path),
                "prompt_count": int(shard["prompt_count"]),
                "rollout_count": len(shard_rows),
            }
        )

    prompt_count = int(protocol["preflight"]["prompt_count"])
    rollouts_per_prompt = int(protocol["training"]["rollouts_per_prompt"])
    expected_rollouts = prompt_count * len(OPD_ARMS) * rollouts_per_prompt
    if len(rows) != expected_rollouts:
        raise ValueError(
            f"Preflight merged count mismatch: expected={expected_rollouts} actual={len(rows)}"
        )
    identities = [
        (str(row["problem_id"]), str(row["arm"]), int(row["candidate_index"]))
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Merged preflight contains duplicate rollout identities.")
    split = protocol["splits"]["calibration"]
    expected_indices = set(
        range(int(split["start_index"]), int(split["start_index"]) + prompt_count)
    )
    observed_indices = {int(row["source_index"]) for row in rows}
    if observed_indices != expected_indices:
        raise ValueError("Merged preflight source-index support is incomplete or unexpected.")
    problem_ids = {str(row["problem_id"]) for row in rows}
    if len(problem_ids) != prompt_count:
        raise ValueError("Merged preflight problem support is incomplete.")
    multiplicities = Counter((str(row["problem_id"]), str(row["arm"])) for row in rows)
    if set(multiplicities.values()) != {rollouts_per_prompt}:
        raise ValueError("Merged preflight per-problem/arm rollout multiplicity mismatch.")
    candidates: Dict[tuple[str, str], set[int]] = {}
    topk_count = 0
    for row in rows:
        problem_arm = (str(row["problem_id"]), str(row["arm"]))
        candidates.setdefault(problem_arm, set()).add(int(row["candidate_index"]))
        completion = row.get("completion_token_ids", [])
        old = row.get("old_student_logprobs", [])
        teacher = row.get("teacher_logprobs", [])
        advantages = row.get("advantages", [])
        if not completion or not len(completion) == len(old) == len(teacher) == len(advantages):
            raise ValueError("Preflight sampled-token arrays are empty or misaligned.")
        if not all(
            math.isfinite(float(value))
            for values in (old, teacher, advantages)
            for value in values
        ) or not math.isfinite(float(row["mean_advantage"])):
            raise ValueError("Preflight contains a non-finite teacher signal.")
        if (
            row.get("teacher_context_mode") != "common_standard_prompt"
            or int(row.get("valid_vocab_size", -1))
            != int(protocol["models"]["tokenizer"]["expected_length"])
            or row.get("scalar_reward_used") is not False
            or row.get("value_head_used") is not False
            or row.get("correctness_is_diagnostic_only") is not True
            or row.get("length_is_diagnostic_only") is not True
        ):
            raise ValueError("Preflight pure-OPD method flags are invalid.")
        topk_count += int("topk_diagnostic" in row)
    expected_candidates = set(range(rollouts_per_prompt))
    if any(values != expected_candidates for values in candidates.values()):
        raise ValueError("Preflight candidate-index support is incomplete or unexpected.")
    expected_topk = len(OPD_ARMS) * int(protocol["diagnostics"]["top_k_rollouts"])
    if topk_count != expected_topk:
        raise ValueError(
            f"Preflight top-k diagnostic count mismatch: expected={expected_topk} actual={topk_count}"
        )
    rows.sort(
        key=lambda row: (
            int(row["source_index"]),
            OPD_ARMS.index(str(row["arm"])),
            int(row["candidate_index"]),
        )
    )
    summary = preflight_summary(rows)
    summary.update(
        {
            "protocol_hash": protocol_hash(protocol),
            "split": dict(split),
            "prompt_count": prompt_count,
            "problem_ids_sha256": canonical_sha256(sorted(problem_ids)),
            "finite_teacher_signal": True,
            "topk_diagnostic_rollouts": topk_count,
            "answer_extraction_failure_rate": sum(
                row.get("predicted_answer") is None for row in rows
            )
            / len(rows),
            "answer_only_proxy_rate": sum(
                int(row["output_token_count"]) <= 16 for row in rows
            )
            / len(rows),
        }
    )
    write_gzip_jsonl(rows_path, rows)
    write_json(summary_path, summary)
    manifest = {
        "status": summary["status"],
        "stage": "preflight_merge",
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "prompt_count": prompt_count,
        "rollout_path": str(rows_path),
        "rollout_sha256": file_sha256(rows_path),
        "rollout_count": len(rows),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "generation_source_sha256": file_sha256(
            PROJECT_ROOT / "scripts/17_3_preflight_opd_signal.py"
        ),
        "merge_source_sha256": file_sha256(Path(__file__).resolve()),
        "shards": shard_evidence,
    }
    write_json(manifest_path, manifest)
    if summary["status"] != "passed":
        raise SystemExit(
            "OPD preflight failed the registered global concise in-band gate; training remains blocked."
        )
    write_json(
        marker_path,
        {
            "status": "passed",
            "protocol_hash": protocol_hash(protocol),
            "manifest_sha256": file_sha256(manifest_path),
            "summary_sha256": file_sha256(summary_path),
            "rollout_sha256": file_sha256(rows_path),
        },
    )
    logging.info("opd_preflight_complete prompts=%d rollouts=%d", prompt_count, len(rows))


def _completed(
    protocol: Dict[str, Any],
    rows_path: Path,
    summary_path: Path,
    manifest_path: Path,
    marker_path: Path,
) -> bool:
    marker = read_json(marker_path)
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    expected_count = (
        int(protocol["preflight"]["prompt_count"])
        * len(OPD_ARMS)
        * int(protocol["training"]["rollouts_per_prompt"])
    )
    expected_marker = {
        "status": "passed",
        "protocol_hash": protocol_hash(protocol),
        "manifest_sha256": file_sha256(manifest_path),
        "summary_sha256": file_sha256(summary_path),
        "rollout_sha256": file_sha256(rows_path),
    }
    return (
        all(marker.get(key) == value for key, value in expected_marker.items())
        and manifest.get("status") == "passed"
        and manifest.get("stage") == "preflight_merge"
        and manifest.get("protocol_hash") == protocol_hash(protocol)
        and manifest.get("rollout_sha256") == file_sha256(rows_path)
        and manifest.get("summary_sha256") == file_sha256(summary_path)
        and manifest.get("merge_source_sha256") == file_sha256(Path(__file__).resolve())
        and manifest.get("generation_source_sha256")
        == file_sha256(PROJECT_ROOT / "scripts/17_3_preflight_opd_signal.py")
        and int(manifest.get("rollout_count", -1)) == expected_count
        and sum(1 for _ in read_gzip_jsonl(rows_path)) == expected_count
        and summary.get("status") == "passed"
    )


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
