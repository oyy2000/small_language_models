#!/usr/bin/env python3
"""Freeze a deterministic 1,000-problem MATH train source pool for the pilot."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.factorial import canonical_sha256, file_sha256, runtime_metadata
from length_budget_distill.math_mix import (
    count_by_fields,
    normalized_question_sha256,
    parse_math_level,
    proportional_stratified_sample,
    stable_text_sha256,
)
from length_budget_distill.records import write_jsonl
from length_budget_distill.verifiers import (
    MATH_VERIFIER_VERSION,
    extract_last_boxed,
    verify_math_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_math_mix_pilot_v1.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config_hash = canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})
    source = dict(config.get("math_train_source", {}))
    dataset_name = str(source["dataset_name"])
    revision = str(source["revision"])
    split = str(source.get("split", "train"))
    dataset_configs = [str(item) for item in source["dataset_configs"]]
    sample_count = int(source.get("sample_count", 1000))
    sampling_seed = int(source.get("sampling_seed", 20260823))

    output_dir = Path(args.output_dir)
    source_path = output_dir / "math_train_source_1000.jsonl"
    manifest_path = output_dir / "math_train_source_manifest.json"
    marker_path = output_dir / "MATH_SOURCE_COMPLETE"
    existing = [path for path in (source_path, manifest_path, marker_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing source artifacts: {existing}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets before preparing the MATH source pool.") from exc

    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_question_hashes: Dict[str, str] = {}
    total_rows = 0
    for dataset_config in dataset_configs:
        loaded = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            revision=revision,
        )
        logging.info("loaded config=%s rows=%d", dataset_config, len(loaded))
        for row_index, row in enumerate(loaded):
            total_rows += 1
            problem_id = f"hendrycks_math/{split}/{dataset_config}/{row_index:05d}"
            question = str(row["problem"])
            solution = str(row["solution"])
            answer = extract_last_boxed(solution)
            rejection_reason = None
            try:
                level = parse_math_level(row["level"])
            except ValueError:
                level = None
                rejection_reason = "invalid_difficulty_level"
            if rejection_reason is None and answer is None:
                rejection_reason = "missing_balanced_boxed_answer"
            elif rejection_reason is None and not verify_math_answer(answer, answer):
                rejection_reason = "gold_answer_not_self_verifiable"
            question_hash = normalized_question_sha256(question)
            if rejection_reason is None and question_hash in seen_question_hashes:
                rejection_reason = "duplicate_normalized_question"
            if rejection_reason is not None:
                rejected.append({"id": problem_id, "reason": rejection_reason})
                continue
            seen_question_hashes[question_hash] = problem_id
            eligible.append(
                {
                    "id": problem_id,
                    "question": question,
                    "answer": str(answer),
                    "subject": dataset_config,
                    "level": level,
                    "type": str(row.get("type", dataset_config)),
                    "question_sha256": question_hash,
                    "official_solution_sha256": stable_text_sha256(solution),
                    "dataset_name": dataset_name,
                    "dataset_revision": revision,
                    "dataset_split": split,
                    "dataset_row_index": row_index,
                }
            )

    if len(eligible) < sample_count:
        raise ValueError(
            f"Not enough eligible MATH train rows: eligible={len(eligible)} requested={sample_count}"
        )
    selected = proportional_stratified_sample(
        eligible,
        sample_count,
        stratum_fields=("subject", "level"),
        seed=sampling_seed,
        minimum_per_stratum=1,
    )
    write_jsonl(source_path, selected)
    manifest = {
        "status": "complete",
        "config_path": args.config,
        "config_hash": config_hash,
        "dataset_name": dataset_name,
        "dataset_revision": revision,
        "dataset_configs": dataset_configs,
        "split": split,
        "total_source_rows": total_rows,
        "eligible_rows": len(eligible),
        "rejected_rows": len(rejected),
        "rejection_counts": count_by_fields(rejected, ("reason",)),
        "sample_count": len(selected),
        "sampling_seed": sampling_seed,
        "stratum_counts": count_by_fields(selected, ("subject", "level")),
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "verifier_version": MATH_VERIFIER_VERSION,
        "runtime": runtime_metadata(packages=("datasets", "math-verify", "sympy")),
    }
    _write_json(manifest_path, manifest)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"config_hash={config_hash}\nsource_sha256={file_sha256(source_path)}\n"
        f"manifest_sha256={file_sha256(manifest_path)}\nsample_count={len(selected)}\n",
        encoding="utf-8",
    )
    logging.info("math_source_complete sample_count=%d output=%s", len(selected), source_path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
