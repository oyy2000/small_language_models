"""Model inference and scoring helpers for the mixed-domain math pilot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .records import read_jsonl, write_jsonl
from .student_prompts import build_student_math_prompt
from .verifiers import extract_answer_for_verifier, verify_answer_for_verifier


def load_causal_lm_bundle(
    model_name: str,
    adapter_path: Optional[str],
    *,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers before pilot evaluation.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(torch_dtype)
    model_kwargs: Dict[str, Any] = {"device_map": device_map}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft before evaluating LoRA adapters.") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


def evaluate_frozen_dataset(
    dataset_path: Path,
    model_bundle: Mapping[str, Any],
    *,
    verifier: str,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    rows = list(read_jsonl(dataset_path))
    outputs: List[Dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [build_student_math_prompt(str(row["question"])) for row in batch]
        predictions = generate_batch(
            model_bundle,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for row, prediction_text in zip(batch, predictions):
            predicted_answer = extract_answer_for_verifier(prediction_text, verifier)
            output_token_count = len(
                model_bundle["tokenizer"].encode(prediction_text, add_special_tokens=False)
            )
            outputs.append(
                {
                    "problem_id": str(row["id"]),
                    "dataset_name": str(row["dataset_name"]),
                    "question": str(row["question"]),
                    "gold_answer": str(row["answer"]),
                    "prediction_text": prediction_text,
                    "predicted_answer": predicted_answer,
                    "is_correct": verify_answer_for_verifier(
                        predicted_answer,
                        str(row["answer"]),
                        verifier,
                    ),
                    "output_token_count": output_token_count,
                    "hit_max_new_tokens": output_token_count >= max_new_tokens - 1,
                    "metadata": {
                        key: value
                        for key, value in row.items()
                        if key not in {"id", "dataset_name", "question", "answer", "verifier"}
                    },
                }
            )
        logging.info("dataset=%s evaluated=%d/%d", dataset_path.name, len(outputs), len(rows))
    return outputs


def summarize_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    model_name: str,
    adapter_path: Optional[str],
    dataset_name: str,
    verifier: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    count = len(rows)
    correct = sum(bool(row["is_correct"]) for row in rows)
    extraction_failures = sum(row.get("predicted_answer") is None for row in rows)
    token_counts = [int(row["output_token_count"]) for row in rows]
    return {
        "status": "complete",
        "model_id": model_id,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "dataset_name": dataset_name,
        "verifier": verifier,
        "max_new_tokens": max_new_tokens,
        "n": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "extraction_failures": extraction_failures,
        "max_token_hits": sum(bool(row["hit_max_new_tokens"]) for row in rows),
        "mean_output_tokens": sum(token_counts) / count if count else 0.0,
    }


def generate_batch(
    model_bundle: Mapping[str, Any],
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    torch = model_bundle["torch"]
    texts = [_format_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    return [text.strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


def write_prediction_artifacts(
    prediction_path: Path,
    summary_path: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    import json

    write_jsonl(prediction_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
