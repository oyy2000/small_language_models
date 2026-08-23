#!/usr/bin/env python3
"""Freeze GSM8K-200, MATH-500-100, and AIME-2025-30 pilot evaluation sets."""

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
    proportional_stratified_sample,
    stable_rank,
)
from length_budget_distill.records import read_jsonl, write_jsonl
from length_budget_distill.verifiers import extract_final_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_math_mix_pilot_v1.json")
    parser.add_argument("--math-train-source", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config_hash = canonical_sha256({key: value for key, value in config.items() if key != "_config_path"})
    suite = dict(config["evaluation_suite"])
    seed = int(suite["sampling_seed"])
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "eval_suite_manifest.json"
    marker_path = output_dir / "EVAL_SUITE_COMPLETE"
    if manifest_path.exists() or marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation suite in {output_dir}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets before preparing evaluation subsets.") from exc

    gsm_rows = _prepare_gsm8k(load_dataset, suite["gsm8k"], seed)
    math_rows = _prepare_math500(load_dataset, suite["math500"], seed)
    aime_rows = _prepare_aime2025(load_dataset, suite["aime2025"])

    train_question_hashes = {
        str(row.get("question_sha256") or normalized_question_sha256(str(row["question"])))
        for row in read_jsonl(Path(args.math_train_source))
    }
    math_eval_hashes = {row["question_sha256"] for row in math_rows}
    overlap = sorted(train_question_hashes & math_eval_hashes)
    if overlap:
        raise ValueError(f"MATH train/MATH-500 normalized-question overlap detected: {overlap[:10]}")

    datasets = {
        "gsm8k": gsm_rows,
        "math500": math_rows,
        "aime2025": aime_rows,
    }
    entries = []
    for dataset_name, rows in datasets.items():
        path = output_dir / f"{dataset_name}.jsonl"
        write_jsonl(path, rows)
        dataset_config = dict(suite[dataset_name])
        entries.append(
            {
                "dataset_name": dataset_name,
                "path": str(path),
                "sha256": file_sha256(path),
                "n": len(rows),
                "verifier": dataset_config["verifier"],
                "max_new_tokens": int(dataset_config["max_new_tokens"]),
                "source_dataset": dataset_config["dataset_name"],
                "source_revision": dataset_config["revision"],
                "source_split": dataset_config["split"],
                "selected_ids": [str(row["id"]) for row in rows],
            }
        )

    manifest = {
        "status": "complete",
        "stage": "pilot",
        "config_hash": config_hash,
        "sampling_seed": seed,
        "temperature": float(suite["temperature"]),
        "top_p": float(suite["top_p"]),
        "batch_size": int(suite["batch_size"]),
        "math_train_source": {
            "path": args.math_train_source,
            "sha256": file_sha256(args.math_train_source),
        },
        "datasets": entries,
        "runtime": runtime_metadata(packages=("datasets",)),
    }
    _write_json(manifest_path, manifest)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"config_hash={config_hash}\nmanifest_sha256={file_sha256(manifest_path)}\n"
        "dataset_count=3\ntotal_examples=330\n",
        encoding="utf-8",
    )
    logging.info("evaluation_suite_complete datasets=3 total_examples=330 output=%s", output_dir)


def _prepare_gsm8k(load_dataset: Any, cfg: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    loaded = load_dataset(
        cfg["dataset_name"],
        cfg.get("dataset_config"),
        split=cfg["split"],
        revision=cfg["revision"],
    )
    start = int(cfg["window_start"])
    stop = int(cfg["window_stop"])
    sample_count = int(cfg["sample_count"])
    if start != 50 or stop != 1319 or len(loaded) < stop:
        raise ValueError(f"Unexpected GSM8K formal window or dataset size: start={start} stop={stop} n={len(loaded)}")
    candidates = []
    for index in range(start, stop):
        row = loaded[index]
        answer = extract_final_answer(str(row["answer"]))
        if answer is None:
            raise ValueError(f"Could not extract GSM8K answer at test index {index}")
        candidates.append(
            {
                "id": f"hf-{index:06d}",
                "question": str(row["question"]),
                "answer": answer,
                "dataset_name": "gsm8k",
                "verifier": cfg["verifier"],
                "source_index": index,
                "question_sha256": normalized_question_sha256(str(row["question"])),
            }
        )
    candidates.sort(key=lambda row: stable_rank(seed, "gsm8k_eval", row["id"]))
    selected = candidates[:sample_count]
    selected.sort(key=lambda row: int(row["source_index"]))
    return selected


def _prepare_math500(load_dataset: Any, cfg: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
    loaded = load_dataset(
        cfg["dataset_name"],
        split=cfg["split"],
        revision=cfg["revision"],
    )
    candidates = [
        {
            "id": str(row["unique_id"]),
            "question": str(row["problem"]),
            "answer": str(row["answer"]),
            "dataset_name": "math500",
            "verifier": cfg["verifier"],
            "subject": str(row["subject"]),
            "level": int(row["level"]),
            "question_sha256": normalized_question_sha256(str(row["problem"])),
        }
        for row in loaded
    ]
    selected = proportional_stratified_sample(
        candidates,
        int(cfg["sample_count"]),
        stratum_fields=("subject", "level"),
        seed=seed,
        minimum_per_stratum=1,
    )
    logging.info("math500_strata=%s", count_by_fields(selected, ("subject", "level")))
    return selected


def _prepare_aime2025(load_dataset: Any, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    loaded = load_dataset(
        cfg["dataset_name"],
        split=cfg["split"],
        revision=cfg["revision"],
    )
    expected = int(cfg["sample_count"])
    if len(loaded) != expected:
        raise ValueError(f"AIME 2025 cardinality mismatch: expected={expected} actual={len(loaded)}")
    return [
        {
            "id": f"aime2025-{int(row['id']):03d}",
            "question": str(row["problem"]),
            "answer": str(row["answer"]),
            "dataset_name": "aime2025",
            "verifier": cfg["verifier"],
            "source_index": int(row["id"]),
            "question_sha256": normalized_question_sha256(str(row["problem"])),
        }
        for row in loaded
    ]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
