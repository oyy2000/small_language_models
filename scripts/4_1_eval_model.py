#!/usr/bin/env python3
"""Evaluate a base model or SFT checkpoint on a math QA split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.records import ProblemRecord, write_jsonl
from length_budget_distill.verifiers import extract_final_answer, verify_answer


SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem correctly and end with "
    "a line in the form: Answer: <final answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment config with dataset fields.")
    parser.add_argument("--model-name", required=True, help="Base model name or full fine-tuned checkpoint path.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA/PEFT adapter path for SFT evaluation.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of eval examples.")
    parser.add_argument("--output-jsonl", required=True, help="Per-example prediction JSONL path.")
    parser.add_argument("--summary-json", required=True, help="Aggregate metrics JSON path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation token limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Generation nucleus sampling value.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype for loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    config["dataset"] = dict(config.get("dataset", {}))
    config["dataset"]["split"] = args.split
    if args.limit is not None:
        config["dataset"]["max_examples"] = args.limit

    problems = load_problem_records(config)
    logging.info("loaded_eval_examples=%d split=%s", len(problems), args.split)

    model_bundle = _load_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    rows = _evaluate(problems, model_bundle, args)
    correct = sum(1 for row in rows if row["is_correct"])
    summary = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "split": args.split,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "output_jsonl": args.output_jsonl,
    }

    write_jsonl(Path(args.output_jsonl), rows)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    logging.info("accuracy=%.4f correct=%d n=%d", summary["accuracy"], correct, len(rows))
    logging.info("wrote_predictions=%s", args.output_jsonl)
    logging.info("wrote_summary=%s", args.summary_json)


def _load_model(model_name: str, adapter_path: Optional[str], device_map: str, torch_dtype: str) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers before evaluation.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"device_map": device_map}
    dtype = _resolve_dtype(torch, torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft before evaluating a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


def _resolve_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return None
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype_name]


def _evaluate(problems: List[ProblemRecord], model_bundle: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    for index, problem in enumerate(problems, start=1):
        prompt = _build_eval_prompt(problem.question)
        prediction_text = _generate(model_bundle, prompt, args)
        predicted_answer = extract_final_answer(prediction_text)
        is_correct = verify_answer(predicted_answer, problem.answer)
        rows.append(
            {
                "problem_id": problem.problem_id,
                "question": problem.question,
                "gold_answer": problem.answer,
                "prediction_text": prediction_text,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "metadata": problem.metadata,
            }
        )
        if index % 25 == 0:
            logging.info("evaluated=%d", index)
    return rows


def _build_eval_prompt(question: str) -> str:
    return f"Problem:\n{question}\n\nReturn the solution and final answer."


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{SYSTEM_PROMPT}\n\n{prompt}"


def _generate(model_bundle: Dict[str, Any], prompt: str, args: argparse.Namespace) -> str:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    torch = model_bundle["torch"]

    text = _format_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    main()
