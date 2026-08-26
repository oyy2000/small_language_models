"""Evaluation helpers for the paired-rewrite SFT pilot."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence

from .records import ProblemRecord
from .student_prompts import build_student_math_prompt
from .verifiers import extract_final_answer, verify_answer


def load_model_bundle(
    model_name: str,
    adapter_path: str | None,
    *,
    torch_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers before paired-rewrite evaluation") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(torch_dtype)
    model_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft before evaluating a paired-rewrite adapter") from exc
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


def evaluate_problems(
    problems: Sequence[ProblemRecord],
    model_bundle: Mapping[str, Any],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_new_tokens <= 0 or batch_size <= 0 or num_samples <= 0:
        raise ValueError("Generation limits, batch size, and sample count must be positive")
    torch = model_bundle["torch"]
    if hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows: List[Dict[str, Any]] = []
    for start in range(0, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        prompts = [build_student_math_prompt(problem.question) for problem in batch]
        generations = generate_batch(
            model_bundle,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
        )
        if len(generations) != len(batch) * num_samples:
            raise RuntimeError("Generation cardinality does not match batch times num_samples")
        for problem_index, problem in enumerate(batch):
            for sample_index in range(num_samples):
                generation = generations[problem_index * num_samples + sample_index]
                predicted = extract_final_answer(str(generation["prediction_text"]))
                rows.append(
                    {
                        "problem_id": problem.problem_id,
                        "sample_index": sample_index,
                        "question": problem.question,
                        "gold_answer": problem.answer,
                        "prediction_text": generation["prediction_text"],
                        "predicted_answer": predicted,
                        "is_correct": verify_answer(predicted, problem.answer),
                        "output_token_count": generation["output_token_count"],
                        "finish_reason": generation["finish_reason"],
                        "hit_max_new_tokens": generation["finish_reason"] == "length",
                        "metadata": problem.metadata,
                    }
                )
        logging.info("paired_eval_progress problems=%d/%d", min(start + len(batch), len(problems)), len(problems))
    return rows


def generate_batch(
    model_bundle: Mapping[str, Any],
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
) -> List[Dict[str, Any]]:
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
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
    if temperature > 0:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p})
    elif num_samples != 1:
        raise ValueError("Greedy decoding supports exactly one sample")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    outputs: List[Dict[str, Any]] = []
    for token_row in generated_ids:
        token_list = token_row.detach().cpu().tolist()
        eos_index = token_list.index(eos_token_id) if eos_token_id in token_list else None
        effective = token_list[:eos_index] if eos_index is not None else token_list
        prediction = tokenizer.decode(effective, skip_special_tokens=True).strip()
        outputs.append(
            {
                "prediction_text": prediction,
                "output_token_count": len(effective),
                "finish_reason": "eos" if eos_index is not None else "length",
            }
        )
    return outputs


def summarize_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    model_name: str,
    adapter_path: str | None,
    split: str,
    start_index: int,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    by_problem: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(str(row["problem_id"]), []).append(row)
    problem_count = len(by_problem)
    total_count = len(rows)
    pass_count = sum(any(bool(row["is_correct"]) for row in group) for group in by_problem.values())
    greedy_count = sum(
        bool(sorted(group, key=lambda row: int(row["sample_index"]))[0]["is_correct"])
        for group in by_problem.values()
    )
    token_counts = [int(row["output_token_count"]) for row in rows]
    return {
        "status": "complete",
        "model_id": model_id,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "split": split,
        "start_index": start_index,
        "n": problem_count,
        "prediction_count": total_count,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "greedy_accuracy": greedy_count / problem_count if problem_count else 0.0,
        "pass_at_k": pass_count / problem_count if problem_count else 0.0,
        "sample_accuracy": (
            sum(bool(row["is_correct"]) for row in rows) / total_count if total_count else 0.0
        ),
        "mean_output_tokens": sum(token_counts) / total_count if total_count else 0.0,
        "max_token_hits": sum(bool(row["hit_max_new_tokens"]) for row in rows),
        "max_token_hit_rate": (
            sum(bool(row["hit_max_new_tokens"]) for row in rows) / total_count if total_count else 0.0
        ),
        "eos_finish_rate": (
            sum(row["finish_reason"] == "eos" for row in rows) / total_count if total_count else 0.0
        ),
        "answer_extraction_failures": sum(row.get("predicted_answer") is None for row in rows),
    }


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt
