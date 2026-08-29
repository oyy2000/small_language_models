#!/usr/bin/env python3
"""Evaluate one base or OPD adapter with the common standard prompt."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.factorial import file_sha256, runtime_metadata
from length_budget_distill.logit_kd import load_and_validate_tokenizers
from length_budget_distill.model_loading import resolve_model_load_spec
from length_budget_distill.opd import (
    OPD_ARMS,
    protocol_hash,
    validate_opd_protocol,
    validated_opd_adapter,
    write_json,
)
from length_budget_distill.opd_analysis import summarize_opd_predictions
from length_budget_distill.records import ProblemRecord, write_jsonl
from length_budget_distill.student_prompts import build_student_math_prompt
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument(
        "--split-name",
        choices=["primary_evaluation", "secondary_evaluation"],
        required=True,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    if args.model_id == "base":
        if args.adapter_path is not None:
            raise ValueError("The base evaluation must not load an adapter.")
    elif args.model_id in OPD_ARMS:
        if args.adapter_path is None:
            raise ValueError(f"OPD arm evaluation requires an adapter: {args.model_id}")
        marker = validated_opd_adapter(
            protocol,
            arm=args.model_id,
            adapter_dir=_resolve(args.adapter_path),
            stage="pilot",
        )
        if marker is None:
            raise ValueError(f"Invalid OPD adapter for {args.model_id}: {args.adapter_path}")
    else:
        raise ValueError(f"Unknown OPD evaluation model ID: {args.model_id}")
    output_path = _resolve(args.output_jsonl)
    summary_path = _resolve(args.summary_json)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite OPD evaluation for {args.model_id}.")
    split = protocol["splits"][args.split_name]
    dataset_config = dict(protocol)
    dataset_config["dataset"] = dict(protocol["dataset"])
    dataset_config["dataset"]["split"] = split["dataset_split"]
    dataset_config["dataset"]["max_examples"] = int(split["start_index"]) + int(split["limit"])
    problems = load_problem_records(dataset_config)
    start = int(split["start_index"])
    problems = problems[start : start + int(split["limit"])]
    if len(problems) != int(split["limit"]):
        raise ValueError("OPD evaluation problem count mismatch.")

    model, tokenizer, torch, model_evidence = _load_model(protocol, args.adapter_path)
    rows: List[Dict[str, Any]] = []
    batch_size = int(protocol["evaluation"]["batch_size"])
    for batch_start in range(0, len(problems), batch_size):
        batch = problems[batch_start : batch_start + batch_size]
        generated = _generate_batch(model, tokenizer, torch, batch, protocol)
        for problem, output in zip(batch, generated):
            predicted = extract_final_answer(output["prediction_text"])
            rows.append(
                {
                    "model_id": args.model_id,
                    "split_name": args.split_name,
                    "problem_id": problem.problem_id,
                    "question": problem.question,
                    "gold_answer": problem.answer,
                    "prediction_text": output["prediction_text"],
                    "predicted_answer": predicted,
                    "is_correct": verify_answer(predicted, problem.answer),
                    "output_token_count": output["output_token_count"],
                    "eos_emitted": output["eos_emitted"],
                    "hit_max_new_tokens": output["hit_max_new_tokens"],
                    "prompt_mode": "common_standard_prompt",
                }
            )
        logging.info("opd_eval_progress model=%s rows=%d/%d", args.model_id, len(rows), len(problems))
    write_jsonl(output_path, rows)
    metrics = summarize_opd_predictions(rows)
    summary = {
        "status": "complete",
        "model_id": args.model_id,
        "split_name": args.split_name,
        "dataset_split": split["dataset_split"],
        "start_index": start,
        "limit": int(split["limit"]),
        "prompt_mode": "common_standard_prompt",
        "protocol_hash": protocol_hash(protocol),
        "prediction_path": str(output_path),
        "prediction_sha256": file_sha256(output_path),
        "adapter_path": str(_resolve(args.adapter_path)) if args.adapter_path else None,
        "model_evidence": model_evidence,
        "metrics": metrics,
        "runtime": runtime_metadata(),
    }
    write_json(summary_path, summary)
    logging.info("opd_evaluation_complete model=%s accuracy=%.4f", args.model_id, metrics["accuracy"])


def _load_model(protocol: Dict[str, Any], adapter_path: str | None) -> tuple[Any, Any, Any, Dict[str, Any]]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install torch, transformers, and peft before OPD evaluation.") from exc
    student = protocol["models"]["student"]
    tokenizer, _teacher_tokenizer, _valid_vocab, tokenizer_evidence = (
        load_and_validate_tokenizers(protocol)
    )
    tokenizer.padding_side = "left"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        student["torch_dtype"]
    ]
    model_source, revision, local_only = resolve_model_load_spec(
        student, override_env="LBD_STUDENT_MODEL_SOURCE"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    ).to("cuda")
    evidence: Dict[str, Any] = {
        **tokenizer_evidence,
        "student_model": student["model_name"],
        "student_revision": student["revision"],
        "adapter_path": None,
    }
    if adapter_path:
        resolved = _resolve(adapter_path)
        model = PeftModel.from_pretrained(model, resolved)
        evidence.update(
            {
                "adapter_path": str(resolved),
                "adapter_config_sha256": file_sha256(resolved / "adapter_config.json"),
                "adapter_model_sha256": file_sha256(resolved / "adapter_model.safetensors"),
            }
        )
    model.eval()
    return model, tokenizer, torch, evidence


def _generate_batch(
    model: Any,
    tokenizer: Any,
    torch: Any,
    problems: Sequence[ProblemRecord],
    protocol: Dict[str, Any],
) -> List[Dict[str, Any]]:
    prompts = [build_student_math_prompt(problem.question) for problem in problems]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    maximum = int(protocol["evaluation"]["max_new_tokens"])
    invalid_token_ids = list(range(len(tokenizer), int(model.config.vocab_size)))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=maximum,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            suppress_tokens=invalid_token_ids or None,
        )[:, inputs["input_ids"].shape[1] :]
    rows = []
    for token_ids in generated.detach().cpu().tolist():
        eos_emitted = tokenizer.eos_token_id is not None and tokenizer.eos_token_id in token_ids
        if eos_emitted:
            token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
        else:
            while token_ids and token_ids[-1] == tokenizer.pad_token_id:
                token_ids.pop()
        rows.append(
            {
                "prediction_text": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "output_token_count": len(token_ids),
                "eos_emitted": eos_emitted,
                "hit_max_new_tokens": not eos_emitted and len(token_ids) >= maximum,
            }
        )
    return rows


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
