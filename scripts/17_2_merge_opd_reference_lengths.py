#!/usr/bin/env python3
"""Merge and audit all frozen base-student OPD reference-length shards."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256
from length_budget_distill.opd import (
    build_bounded_concise_prompt,
    protocol_hash,
    read_json,
    reference_length_bounds,
    validate_opd_protocol,
    write_json,
)
from length_budget_distill.records import read_jsonl, write_jsonl


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
    reference_path = output_dir / "training_references.jsonl"
    manifest_path = output_dir / "reference_manifest.json"
    marker_path = output_dir / "REFERENCES_COMPLETE"
    if all(path.is_file() for path in (reference_path, manifest_path, marker_path)) and args.skip_complete:
        from length_budget_distill.opd import validate_reference_manifest

        existing, _rows = validate_reference_manifest(protocol, manifest_path)
        marker = read_json(marker_path)
        expected_marker = {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "reference_manifest_sha256": file_sha256(manifest_path),
            "reference_sha256": existing["reference_sha256"],
            "record_count": int(protocol["splits"]["training"]["limit"]),
        }
        if all(marker.get(key) == value for key, value in expected_marker.items()):
            logging.info("opd_references_already_complete output=%s", reference_path)
            return
        raise ValueError("Existing merged OPD reference evidence failed validation.")
    if any(path.exists() for path in (reference_path, manifest_path, marker_path)):
        raise FileExistsError("Refusing to overwrite merged OPD reference evidence.")

    num_shards = int(protocol["reference_generation"]["num_shards"])
    rows: List[Dict[str, Any]] = []
    shard_evidence = []
    for shard_index in range(num_shards):
        suffix = f"shard_{shard_index:05d}_of_{num_shards:05d}"
        shard_manifest_path = input_dir / "manifests" / f"{suffix}.json"
        if not shard_manifest_path.is_file():
            raise FileNotFoundError(f"Missing reference shard manifest: {shard_manifest_path}")
        shard = read_json(shard_manifest_path)
        if shard.get("status") != "complete" or shard.get("is_smoke"):
            raise ValueError(f"Reference shard is incomplete or smoke-only: {shard_manifest_path}")
        if shard.get("protocol_hash") != protocol_hash(protocol):
            raise ValueError(f"Reference shard protocol mismatch: {shard_manifest_path}")
        data_path = Path(str(shard["output_path"]))
        if not data_path.is_file() or file_sha256(data_path) != shard.get("output_sha256"):
            raise ValueError(f"Reference shard data/hash mismatch: {data_path}")
        shard_rows = list(read_jsonl(data_path))
        if len(shard_rows) != int(shard["record_count"]):
            raise ValueError(f"Reference shard count mismatch: {data_path}")
        rows.extend(shard_rows)
        shard_evidence.append(
            {
                "manifest_path": str(shard_manifest_path),
                "manifest_sha256": file_sha256(shard_manifest_path),
                "data_path": str(data_path),
                "data_sha256": file_sha256(data_path),
                "record_count": len(shard_rows),
            }
        )

    expected_n = int(protocol["splits"]["training"]["limit"])
    if len(rows) != expected_n:
        raise ValueError(f"Merged reference count mismatch: expected={expected_n} actual={len(rows)}")
    rows.sort(key=lambda row: int(row["source_index"]))
    ids = [str(row["problem_id"]) for row in rows]
    indices = [int(row["source_index"]) for row in rows]
    if len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        raise ValueError("Merged references contain duplicate problem IDs or source indices.")
    split = protocol["splits"]["training"]
    expected_indices = list(range(int(split["start_index"]), int(split["start_index"]) + expected_n))
    if indices != expected_indices:
        raise ValueError("Merged references have missing or unexpected source indices.")
    for row in rows:
        bounds = reference_length_bounds(
            int(row["reference_output_tokens"]),
            lower_ratio=float(protocol["concise_prompt"]["lower_ratio"]),
            upper_ratio=float(protocol["concise_prompt"]["upper_ratio"]),
            minimum_tokens=int(protocol["concise_prompt"]["minimum_tokens"]),
            maximum_tokens=int(protocol["concise_prompt"]["maximum_tokens"]),
        )
        if bounds != (int(row["concise_lower_tokens"]), int(row["concise_upper_tokens"])):
            raise ValueError(f"Reference bounds mismatch: {row['problem_id']}")
        expected_prompt = build_bounded_concise_prompt(str(row["question"]), *bounds)
        if row["concise_prompt"] != expected_prompt:
            raise ValueError(f"Concise prompt mismatch: {row['problem_id']}")

    write_jsonl(reference_path, rows)
    manifest = {
        "status": "complete",
        "stage": "training_reference_merge",
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "record_count": len(rows),
        "problem_ids_sha256": canonical_sha256(ids),
        "source_indices_sha256": canonical_sha256(indices),
        "reference_path": str(reference_path),
        "reference_sha256": file_sha256(reference_path),
        "shards": shard_evidence,
        "source_sha256": file_sha256(Path(__file__).resolve()),
    }
    write_json(manifest_path, manifest)
    marker = {
        "status": "complete",
        "protocol_hash": protocol_hash(protocol),
        "reference_manifest_sha256": file_sha256(manifest_path),
        "reference_sha256": file_sha256(reference_path),
        "record_count": len(rows),
    }
    write_json(marker_path, marker)
    logging.info("opd_references_complete records=%d output=%s", len(rows), reference_path)


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
