#!/usr/bin/env python3
"""Generate one shard of frozen base-student standard-prompt reference lengths."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.config import load_config
from length_budget_distill.datasets import load_problem_records
from length_budget_distill.factorial import canonical_sha256, file_sha256, runtime_metadata
from length_budget_distill.logit_kd import load_and_validate_tokenizers
from length_budget_distill.model_loading import resolve_model_load_spec
from length_budget_distill.opd import (
    build_bounded_concise_prompt,
    protocol_hash,
    reference_length_bounds,
    validate_opd_protocol,
    write_json,
)
from length_budget_distill.records import ProblemRecord, write_jsonl
from length_budget_distill.student_prompts import build_student_math_prompt
from length_budget_distill.verifiers import extract_final_answer, verify_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_opd_prompt_pilot_v1.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-only cap within this shard.")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = _resolve(args.config)
    protocol = load_config(str(config_path))
    validate_opd_protocol(protocol)
    configured_shards = int(protocol["reference_generation"]["num_shards"])
    num_shards = args.num_shards or configured_shards
    if num_shards <= 0 or not 0 <= args.shard_index < num_shards:
        raise ValueError("Invalid reference shard topology.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    split = protocol["splits"]["training"]
    dataset_config = dict(protocol)
    dataset_config["dataset"] = dict(protocol["dataset"])
    dataset_config["dataset"]["split"] = split["dataset_split"]
    dataset_config["dataset"]["max_examples"] = int(split["start_index"]) + int(split["limit"])
    problems = load_problem_records(dataset_config)
    start = int(split["start_index"])
    stop = start + int(split["limit"])
    selected = list(enumerate(problems[start:stop], start=start))
    assigned = [item for item in selected if item[0] % num_shards == args.shard_index]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    if not assigned:
        raise ValueError("No reference problems assigned to this shard.")

    output_dir = _resolve(args.output_dir)
    suffix = f"shard_{args.shard_index:05d}_of_{num_shards:05d}"
    output_path = output_dir / "shards" / f"{suffix}.jsonl"
    manifest_path = output_dir / "manifests" / f"{suffix}.json"
    if output_path.is_file() and manifest_path.is_file() and args.skip_complete:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        expected = {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "shard_index": args.shard_index,
            "num_shards": num_shards,
            "is_smoke": args.limit is not None,
            "record_count": len(assigned),
            "output_path": str(output_path),
            "output_sha256": file_sha256(output_path),
            "source_sha256": file_sha256(Path(__file__).resolve()),
        }
        if all(existing.get(key) == value for key, value in expected.items()):
            logging.info("reference_shard_already_complete output=%s", output_path)
            return
        raise ValueError(f"Existing reference shard failed validation: {suffix}")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite reference shard {suffix}.")

    model, tokenizer, torch, model_evidence = _load_student(protocol)
    records: List[Dict[str, Any]] = []
    batch_size = int(protocol["reference_generation"]["batch_size"])
    for batch_start in range(0, len(assigned), batch_size):
        batch = assigned[batch_start : batch_start + batch_size]
        prompts = [build_student_math_prompt(problem.question) for _, problem in batch]
        generated = _generate_batch(model, tokenizer, torch, prompts, protocol)
        for (source_index, problem), standard_prompt, output in zip(batch, prompts, generated):
            output_tokens = int(output["output_token_count"])
            lower, upper = reference_length_bounds(
                output_tokens,
                lower_ratio=float(protocol["concise_prompt"]["lower_ratio"]),
                upper_ratio=float(protocol["concise_prompt"]["upper_ratio"]),
                minimum_tokens=int(protocol["concise_prompt"]["minimum_tokens"]),
                maximum_tokens=int(protocol["concise_prompt"]["maximum_tokens"]),
            )
            predicted = extract_final_answer(output["response_text"])
            records.append(
                {
                    "problem_id": problem.problem_id,
                    "source_index": source_index,
                    "question": problem.question,
                    "gold_answer": problem.answer,
                    "source_hash": canonical_sha256(
                        {
                            "problem_id": problem.problem_id,
                            "question": problem.question,
                            "answer": problem.answer,
                        }
                    ),
                    "standard_prompt": standard_prompt,
                    "reference_response": output["response_text"],
                    "reference_output_tokens": output_tokens,
                    "reference_eos_emitted": output["eos_emitted"],
                    "reference_hit_max_new_tokens": output["hit_max_new_tokens"],
                    "reference_predicted_answer": predicted,
                    "reference_is_correct": verify_answer(predicted, problem.answer),
                    "concise_lower_tokens": lower,
                    "concise_upper_tokens": upper,
                    "concise_prompt": build_bounded_concise_prompt(problem.question, lower, upper),
                    "reference_policy": "frozen_qwen2p5_1p5b_instruct_base",
                    "reference_decoding": "greedy",
                    "protocol_hash": protocol_hash(protocol),
                }
            )
        logging.info("reference_progress shard=%d records=%d/%d", args.shard_index, len(records), len(assigned))

    write_jsonl(output_path, records)
    manifest = {
        "status": "complete",
        "stage": "training_reference_generation",
        "protocol_hash": protocol_hash(protocol),
        "config_path": str(config_path),
        "config_file_sha256": file_sha256(config_path),
        "shard_index": args.shard_index,
        "num_shards": num_shards,
        "is_smoke": args.limit is not None,
        "record_count": len(records),
        "problem_ids_sha256": canonical_sha256([row["problem_id"] for row in records]),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "model_evidence": model_evidence,
        "runtime": runtime_metadata(),
    }
    write_json(manifest_path, manifest)
    logging.info("reference_shard_complete output=%s", output_path)


def _load_student(protocol: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install torch and transformers before reference generation.") from exc
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
    model.eval()
    return model, tokenizer, torch, {
        **tokenizer_evidence,
        "student_model_name": student["model_name"],
        "student_revision": student["revision"],
    }


def _generate_batch(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompts: Sequence[str],
    protocol: Mapping[str, Any],
) -> List[Dict[str, Any]]:
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
    maximum = int(protocol["reference_generation"]["max_new_tokens"])
    with torch.inference_mode():
        invalid_token_ids = list(range(len(tokenizer), int(model.config.vocab_size)))
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=maximum,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            suppress_tokens=invalid_token_ids or None,
        )
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    rows = []
    for values in generated.detach().cpu().tolist():
        eos_emitted = tokenizer.eos_token_id is not None and tokenizer.eos_token_id in values
        if eos_emitted:
            values = values[: values.index(tokenizer.eos_token_id)]
        else:
            while values and values[-1] == tokenizer.pad_token_id:
                values.pop()
        if not values:
            raise RuntimeError("Frozen student produced an empty reference response.")
        rows.append(
            {
                "response_text": tokenizer.decode(values, skip_special_tokens=True).strip(),
                "output_token_count": len(values),
                "eos_emitted": eos_emitted,
                "hit_max_new_tokens": not eos_emitted and len(values) >= maximum,
            }
        )
    return rows


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
