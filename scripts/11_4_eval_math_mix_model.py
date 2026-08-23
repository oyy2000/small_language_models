#!/usr/bin/env python3
"""Evaluate one base/adapter model on all frozen mixed-domain pilot datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import file_sha256, runtime_metadata, validated_adapter_evidence
from length_budget_distill.pilot_evaluation import (
    evaluate_frozen_dataset,
    load_causal_lm_bundle,
    summarize_predictions,
    write_prediction_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--model-metadata-json", required=True)
    parser.add_argument("--eval-suite-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    model_manifest_path = output_dir / "model_manifests" / f"{args.model_id}.json"
    if args.skip_complete and _model_eval_complete(model_manifest_path):
        logging.info("skip_complete_model=%s", args.model_id)
        return
    if model_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite model evaluation manifest: {model_manifest_path}")

    suite_path = Path(args.eval_suite_manifest)
    suite = _read_json(suite_path)
    if suite.get("status") != "complete":
        raise ValueError(f"Evaluation suite is incomplete: {suite_path}")
    metadata = json.loads(args.model_metadata_json)
    if not isinstance(metadata, dict):
        raise ValueError("--model-metadata-json must decode to an object")
    adapter_evidence = None
    if args.adapter_path:
        adapter_evidence = validated_adapter_evidence(args.adapter_path)
        if adapter_evidence is None:
            raise FileNotFoundError(f"Adapter lacks valid completion evidence: {args.adapter_path}")

    bundle = load_causal_lm_bundle(
        args.model_name,
        args.adapter_path,
        torch_dtype=args.torch_dtype,
    )
    artifact_entries = []
    for dataset in suite["datasets"]:
        dataset_name = str(dataset["dataset_name"])
        dataset_path = Path(str(dataset["path"]))
        if file_sha256(dataset_path) != dataset["sha256"]:
            raise ValueError(f"Frozen evaluation dataset hash mismatch: {dataset_path}")
        prediction_path = output_dir / "predictions" / dataset_name / f"{args.model_id}.jsonl"
        summary_path = output_dir / "summaries" / dataset_name / f"{args.model_id}.json"
        existing = [path for path in (prediction_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite incomplete model artifacts: {existing}")
        rows = evaluate_frozen_dataset(
            dataset_path,
            bundle,
            verifier=str(dataset["verifier"]),
            max_new_tokens=int(dataset["max_new_tokens"]),
            batch_size=int(suite["batch_size"]),
            temperature=float(suite["temperature"]),
            top_p=float(suite["top_p"]),
        )
        if len(rows) != int(dataset["n"]):
            raise RuntimeError(
                f"Evaluation cardinality mismatch dataset={dataset_name} "
                f"expected={dataset['n']} actual={len(rows)}"
            )
        summary = summarize_predictions(
            rows,
            model_id=args.model_id,
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            dataset_name=dataset_name,
            verifier=str(dataset["verifier"]),
            max_new_tokens=int(dataset["max_new_tokens"]),
        )
        summary["model_metadata"] = metadata
        summary["dataset_sha256"] = dataset["sha256"]
        write_prediction_artifacts(prediction_path, summary_path, rows, summary)
        artifact_entries.append(
            {
                "dataset_name": dataset_name,
                "n": len(rows),
                "prediction_path": str(prediction_path),
                "prediction_sha256": file_sha256(prediction_path),
                "summary_path": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
            }
        )

    model_manifest = {
        "status": "complete",
        "model_id": args.model_id,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "adapter_evidence": adapter_evidence,
        "model_metadata": metadata,
        "eval_suite_manifest": str(suite_path),
        "eval_suite_manifest_sha256": file_sha256(suite_path),
        "artifacts": artifact_entries,
        "runtime": runtime_metadata(
            packages=("torch", "transformers", "peft", "math-verify", "sympy")
        ),
    }
    _write_json(model_manifest_path, model_manifest)
    logging.info("model_evaluation_complete model=%s datasets=%d", args.model_id, len(artifact_entries))


def _model_eval_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    manifest = _read_json(path)
    if manifest.get("status") != "complete" or len(manifest.get("artifacts", [])) != 3:
        return False
    for artifact in manifest["artifacts"]:
        prediction_path = Path(artifact["prediction_path"])
        summary_path = Path(artifact["summary_path"])
        if not prediction_path.is_file() or not summary_path.is_file():
            return False
        if file_sha256(prediction_path) != artifact["prediction_sha256"]:
            return False
        if file_sha256(summary_path) != artifact["summary_sha256"]:
            return False
    return True


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
