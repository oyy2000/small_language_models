"""Pure sampled-token on-policy distillation for dual prompt contexts.

The student samples responses from either the standard or bounded-concise
prompt.  A fixed teacher scores the exact sampled token IDs under the common
standard prompt.  Correctness and response length are recorded for diagnostics
only and never enter the optimization objective.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .factorial import canonical_sha256, file_sha256, runtime_metadata, stable_generation_seed
from .logit_kd import (
    causal_completion_logits,
    load_teacher_and_student,
    publish_adapter,
    validated_training_marker,
)
from .records import read_jsonl
from .student_prompts import build_student_math_prompt
from .verifiers import extract_final_answer, verify_answer


OPD_ARMS = ("standard_prompt", "bounded_concise_prompt")
OBJECTIVE_NAME = "sampled_token_reverse_kl_ppo_clipped"
TEACHER_CONTEXT_MODE = "common_standard_prompt"


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    """Hash a loaded protocol without its path-only loader metadata."""

    return canonical_sha256({key: value for key, value in protocol.items() if key != "_config_path"})


def validate_opd_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the invariants that make the experiment pure OPD."""

    if protocol.get("evidence_level") != "exploratory_single_seed_pilot":
        raise ValueError("The OPD v1 evidence level must remain exploratory_single_seed_pilot.")
    if protocol.get("scope") != "GSM8K only":
        raise ValueError("The OPD v1 scope is restricted to GSM8K only.")
    dataset = protocol.get("dataset", {})
    if (
        dataset.get("source") != "hf_dataset"
        or dataset.get("dataset_name") != "openai/gsm8k"
        or dataset.get("dataset_config") != "main"
        or dataset.get("verifier") != "gsm8k_numeric"
    ):
        raise ValueError("The OPD v1 dataset must remain openai/gsm8k main with gsm8k_numeric.")
    if tuple(protocol.get("arms", ())) != OPD_ARMS:
        raise ValueError(f"arms must equal {OPD_ARMS}.")
    models = protocol.get("models", {})
    for name in ("teacher", "student", "tokenizer"):
        if not isinstance(models.get(name), Mapping):
            raise ValueError(f"models.{name} must be configured.")
    expected_models = {
        "teacher": (
            "Qwen/Qwen2.5-7B-Instruct",
            "a09a35458c702b33eeacc393d103063234e8bc28",
        ),
        "student": (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        ),
        "tokenizer": (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        ),
    }
    for name, expected in expected_models.items():
        observed = (models[name].get("model_name"), models[name].get("revision"))
        if observed != expected:
            raise ValueError(f"models.{name} name/revision mismatch: {observed}")
    if (
        int(models["tokenizer"].get("expected_length", -1)) != 151665
        or models["tokenizer"].get("expected_tokenizer_json_sha256")
        != "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    ):
        raise ValueError("The registered OPD tokenizer length or file hash changed.")
    lora = models["student"].get("lora", {})
    if (
        models["student"].get("use_lora") is not True
        or int(lora.get("r", -1)) != 4
        or int(lora.get("alpha", -1)) != 16
        or float(lora.get("dropout", -1.0)) != 0.05
        or lora.get("target_modules") != "all-linear"
    ):
        raise ValueError("The OPD v1 student must use the registered rank-4 LoRA setup.")
    objective = protocol.get("objective", {})
    if objective.get("name") != OBJECTIVE_NAME:
        raise ValueError(f"objective.name must be {OBJECTIVE_NAME!r}.")
    if objective.get("teacher_context_mode") != TEACHER_CONTEXT_MODE:
        raise ValueError(f"teacher context must be {TEACHER_CONTEXT_MODE!r}.")
    forbidden = (
        "scalar_teacher_reward",
        "correctness_reward",
        "length_reward",
        "hard_ce_loss",
        "value_head",
    )
    enabled = [key for key in forbidden if bool(objective.get(key, False))]
    if enabled:
        raise ValueError(f"Pure OPD forbids enabled auxiliary objectives: {enabled}")
    clip_ratio = float(objective.get("clip_ratio", 0.0))
    if clip_ratio != 0.2:
        raise ValueError("The OPD v1 clip ratio must be 0.2.")
    if objective.get("loss_aggregation") != "token_mean":
        raise ValueError("Pure OPD v1 requires token_mean loss aggregation.")

    bounds = protocol.get("concise_prompt", {})
    lower_ratio = float(bounds.get("lower_ratio", 0.0))
    upper_ratio = float(bounds.get("upper_ratio", 0.0))
    minimum = int(bounds.get("minimum_tokens", 0))
    maximum = int(bounds.get("maximum_tokens", 0))
    if not 0.0 < lower_ratio <= upper_ratio <= 1.0:
        raise ValueError("Concise ratios must satisfy 0 < lower <= upper <= 1.")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Concise absolute token bounds are invalid.")

    training = protocol.get("training", {})
    if int(training.get("seed", -1)) != 17:
        raise ValueError("The v1 OPD pilot is restricted to seed 17.")
    for key in ("rollouts_per_prompt", "prompts_per_batch", "mini_batch_rollouts", "num_epochs"):
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    if int(training["num_epochs"]) != 1:
        raise ValueError("The registered OPD pilot uses exactly one training epoch.")
    if float(training.get("sampling_temperature", 0.0)) != 1.0:
        raise ValueError("The registered OPD sampling temperature is 1.0.")
    if float(training.get("top_p", 0.0)) != 1.0:
        raise ValueError("The registered OPD top-p is 1.0.")
    if int(training.get("max_new_tokens", 0)) != 512:
        raise ValueError("The registered OPD maximum response length is 512 tokens.")
    if float(training.get("learning_rate", 0.0)) != 1e-6:
        raise ValueError("The registered OPD learning rate is 1e-6.")
    registered_training = {
        "rollouts_per_prompt": 4,
        "prompts_per_batch": 16,
        "mini_batch_rollouts": 8,
        "max_length": 2048,
    }
    for key, expected in registered_training.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"training.{key} must remain {expected}.")
    if (
        float(training.get("warmup_ratio", -1.0)) != 0.03
        or float(training.get("max_grad_norm", -1.0)) != 1.0
        or training.get("gradient_checkpointing") is not True
    ):
        raise ValueError("The OPD v1 warmup, gradient norm, and checkpointing settings changed.")

    reference_generation = protocol.get("reference_generation", {})
    if (
        reference_generation.get("decoding") != "greedy"
        or int(reference_generation.get("num_shards", -1)) != 3
        or int(reference_generation.get("max_new_tokens", -1)) != 512
        or int(reference_generation.get("batch_size", -1)) != 16
    ):
        raise ValueError("The OPD reference-generation recipe changed.")
    preflight = protocol.get("preflight", {})
    if (
        int(preflight.get("prompt_count", -1)) != 100
        or int(preflight.get("num_shards", -1)) != 2
        or float(preflight.get("minimum_concise_in_band_rate", -1.0)) != 0.70
        or preflight.get("stop_training_on_failure") is not True
    ):
        raise ValueError("The OPD preflight gate changed.")
    evaluation = protocol.get("evaluation", {})
    if (
        evaluation.get("prompt_mode") != "common_standard_prompt"
        or int(evaluation.get("max_new_tokens", -1)) != 512
        or float(evaluation.get("temperature", -1.0)) != 0.0
        or float(evaluation.get("top_p", -1.0)) != 1.0
    ):
        raise ValueError("The OPD common-prompt evaluation recipe changed.")
    gate = protocol.get("advancement_gate", {})
    expected_gate = {
        "accuracy_noninferiority_margin_pp": 1.0,
        "maximum_mean_output_token_ratio": 0.90,
        "maximum_extraction_failure_increase_pp": 1.0,
        "maximum_truncation_increase_pp": 1.0,
    }
    if any(float(gate.get(key, -1.0)) != value for key, value in expected_gate.items()):
        raise ValueError("The OPD advancement gate changed.")

    splits = protocol.get("splits", {})
    required_splits = {
        "calibration": ("train", 2000, 500),
        "training": ("train", 3000, 3500),
        "primary_evaluation": ("train", 6500, 973),
        "secondary_evaluation": ("test", 50, 1269),
    }
    for name, expected in required_splits.items():
        split = splits.get(name, {})
        actual = (
            str(split.get("dataset_split", "")),
            int(split.get("start_index", -1)),
            int(split.get("limit", -1)),
        )
        if actual != expected:
            raise ValueError(f"splits.{name} must equal {expected}; observed={actual}.")


def reference_length_bounds(
    reference_tokens: int,
    *,
    lower_ratio: float = 0.70,
    upper_ratio: float = 0.90,
    minimum_tokens: int = 96,
    maximum_tokens: int = 256,
) -> Tuple[int, int]:
    """Return the registered inclusive concise target interval."""

    if reference_tokens <= 0:
        raise ValueError("reference_tokens must be positive.")
    if not 0.0 < lower_ratio <= upper_ratio <= 1.0:
        raise ValueError("Ratios must satisfy 0 < lower <= upper <= 1.")
    if minimum_tokens <= 0 or maximum_tokens < minimum_tokens:
        raise ValueError("Absolute token bounds are invalid.")
    lower = min(maximum_tokens, max(minimum_tokens, math.floor(lower_ratio * reference_tokens)))
    upper = min(maximum_tokens, max(lower, math.ceil(upper_ratio * reference_tokens)))
    return int(lower), int(upper)


def build_bounded_concise_prompt(question: str, lower_tokens: int, upper_tokens: int) -> str:
    """Build the student-only paired relative-length rollout prompt."""

    if lower_tokens <= 0 or upper_tokens < lower_tokens:
        raise ValueError("Invalid bounded-concise interval.")
    return (
        f"Problem:\n{question}\n\n"
        "Solve the problem with the shortest complete and independently checkable reasoning you can. "
        "Include every necessary intermediate calculation; do not give an answer-only response and do "
        "not pad the solution with repetition. "
        f"Aim for {lower_tokens} to {upper_tokens} solution tokens. If exact length compliance conflicts "
        "with correct, complete reasoning, preserve the reasoning. "
        "End with a line in the form: Answer: <final answer>."
    )


def prompt_for_arm(reference: Mapping[str, Any], arm: str) -> str:
    if arm not in OPD_ARMS:
        raise ValueError(f"Unknown OPD arm: {arm}")
    if arm == "standard_prompt":
        return str(reference["standard_prompt"])
    return str(reference["concise_prompt"])


def sampled_token_advantage(teacher_logprobs: Any, old_student_logprobs: Any) -> Any:
    """Return detached sampled-token reverse-KL rewards."""

    import torch

    if teacher_logprobs.shape != old_student_logprobs.shape:
        raise ValueError("Teacher and old-student log-prob shapes must match.")
    if teacher_logprobs.ndim != 1 or teacher_logprobs.numel() == 0:
        raise ValueError("Sampled-token log-probs must be non-empty vectors.")
    if not bool(torch.isfinite(teacher_logprobs).all()) or not bool(
        torch.isfinite(old_student_logprobs).all()
    ):
        raise ValueError("Sampled-token log-probs must be finite.")
    return (teacher_logprobs - old_student_logprobs).detach()


def clipped_opd_loss(
    new_student_logprobs: Any,
    old_student_logprobs: Any,
    advantages: Any,
    *,
    clip_ratio: float,
    reduction: str = "mean",
) -> Tuple[Any, Dict[str, Any]]:
    """Compute the PPO-style clipped sampled-token OPD policy loss."""

    import torch

    if new_student_logprobs.shape != old_student_logprobs.shape or (
        new_student_logprobs.shape != advantages.shape
    ):
        raise ValueError("New, old, and advantage tensors must have identical shapes.")
    if new_student_logprobs.ndim != 1 or new_student_logprobs.numel() == 0:
        raise ValueError("OPD loss expects non-empty token vectors.")
    if not 0.0 < float(clip_ratio) < 1.0:
        raise ValueError("clip_ratio must lie in (0, 1).")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")
    if not bool(torch.isfinite(new_student_logprobs).all()) or not bool(
        torch.isfinite(old_student_logprobs).all()
    ) or not bool(torch.isfinite(advantages).all()):
        raise ValueError("OPD loss inputs must be finite.")
    ratio = torch.exp(new_student_logprobs - old_student_logprobs.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
    objective = torch.minimum(ratio * advantages.detach(), clipped_ratio * advantages.detach())
    loss = -objective.mean() if reduction == "mean" else -objective.sum()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("OPD clipped surrogate became non-finite.")
    clipped = (ratio < 1.0 - float(clip_ratio)) | (ratio > 1.0 + float(clip_ratio))
    metrics = {
        "loss": loss.detach(),
        "mean_ratio": ratio.detach().mean(),
        "clip_fraction": clipped.float().detach().mean(),
        "mean_advantage": advantages.detach().mean(),
        "token_count": int(advantages.numel()),
    }
    return loss, metrics


def topk_overlap(student_ids: Any, teacher_ids: Any) -> float:
    """Mean per-position overlap ratio for equally shaped top-k token IDs."""

    import torch

    if student_ids.shape != teacher_ids.shape or student_ids.ndim != 2:
        raise ValueError("Top-k token ID tensors must have identical [tokens, k] shapes.")
    if student_ids.shape[0] == 0 or student_ids.shape[1] == 0:
        raise ValueError("Top-k token ID tensors must be non-empty.")
    matches = (student_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)).any(dim=-1)
    return float(matches.float().mean().detach().cpu())


def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Compute tie-aware AUROC without external dependencies."""

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length.")
    positive = [float(score) for score, label in zip(scores, labels) if label]
    negative = [float(score) for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += 1.0 if left > right else 0.5 if left == right else 0.0
    return wins / (len(positive) * len(negative))


def preflight_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize prompt adherence and teacher-signal diagnostics."""

    if not rows:
        raise ValueError("Preflight rows must be non-empty.")
    rewards = [float(row["mean_advantage"]) for row in rows]
    labels = [bool(row["is_correct"]) for row in rows]
    concise = [row for row in rows if row.get("arm") == "bounded_concise_prompt"]
    standard = [row for row in rows if row.get("arm") == "standard_prompt"]
    if not concise or not standard:
        raise ValueError("Preflight must contain both OPD arms.")
    concise_compliant = sum(bool(row["in_length_band"]) for row in concise)
    correct_rewards = [score for score, label in zip(rewards, labels) if label]
    incorrect_rewards = [score for score, label in zip(rewards, labels) if not label]
    return {
        "status": "passed" if concise_compliant / len(concise) >= 0.70 else "failed",
        "n": len(rows),
        "n_per_arm": {"standard_prompt": len(standard), "bounded_concise_prompt": len(concise)},
        "concise_in_band_rate": concise_compliant / len(concise),
        "minimum_concise_in_band_rate": 0.70,
        "teacher_reward_auc_for_correctness": binary_auc(rewards, labels),
        "mean_teacher_reward_correct": mean(correct_rewards) if correct_rewards else None,
        "mean_teacher_reward_incorrect": mean(incorrect_rewards) if incorrect_rewards else None,
        "correctness_is_diagnostic_only": True,
        "length_is_diagnostic_only": True,
    }


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(destination, "xt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_gzip_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid gzip JSONL at {path}:{line_number}") from exc


def validate_reference_manifest(
    protocol: Mapping[str, Any],
    manifest_path: str | Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and hash-check the frozen paired-prompt reference dataset."""

    path = Path(manifest_path)
    manifest = read_json(path)
    expected_n = int(protocol["splits"]["training"]["limit"])
    if manifest.get("status") != "complete":
        raise ValueError("Reference manifest is incomplete.")
    if manifest.get("protocol_hash") != protocol_hash(protocol):
        raise ValueError("Reference manifest protocol hash mismatch.")
    reference_path = Path(str(manifest["reference_path"]))
    if not reference_path.is_file() or file_sha256(reference_path) != manifest.get("reference_sha256"):
        raise ValueError("Reference data file/hash mismatch.")
    rows = list(read_jsonl(reference_path))
    if len(rows) != expected_n or int(manifest.get("record_count", -1)) != expected_n:
        raise ValueError(f"Reference record count mismatch: expected={expected_n} actual={len(rows)}")
    ids = [str(row["problem_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Reference data contains duplicate problem IDs.")
    required = {
        "question",
        "gold_answer",
        "source_index",
        "standard_prompt",
        "concise_prompt",
        "reference_output_tokens",
        "concise_lower_tokens",
        "concise_upper_tokens",
    }
    observed_indices = [int(row["source_index"]) for row in rows]
    split = protocol["splits"]["training"]
    expected_indices = list(
        range(int(split["start_index"]), int(split["start_index"]) + expected_n)
    )
    if observed_indices != expected_indices:
        raise ValueError("Reference source-index order/support mismatch.")
    for row in rows:
        if not required <= set(row):
            raise ValueError(f"Reference record is missing required fields: {row.get('problem_id')}")
        expected_bounds = reference_length_bounds(
            int(row["reference_output_tokens"]),
            lower_ratio=float(protocol["concise_prompt"]["lower_ratio"]),
            upper_ratio=float(protocol["concise_prompt"]["upper_ratio"]),
            minimum_tokens=int(protocol["concise_prompt"]["minimum_tokens"]),
            maximum_tokens=int(protocol["concise_prompt"]["maximum_tokens"]),
        )
        observed_bounds = (int(row["concise_lower_tokens"]), int(row["concise_upper_tokens"]))
        if observed_bounds != expected_bounds:
            raise ValueError(f"Concise reference bounds mismatch: {row['problem_id']}")
        if row["standard_prompt"] != build_student_math_prompt(str(row["question"])):
            raise ValueError(f"Standard reference prompt mismatch: {row['problem_id']}")
        if row["concise_prompt"] != build_bounded_concise_prompt(
            str(row["question"]), *expected_bounds
        ):
            raise ValueError(f"Concise reference prompt mismatch: {row['problem_id']}")
        if (
            row.get("protocol_hash") != protocol_hash(protocol)
            or row.get("reference_policy") != "frozen_qwen2p5_1p5b_instruct_base"
            or row.get("reference_decoding") != "greedy"
        ):
            raise ValueError(f"Reference policy evidence mismatch: {row['problem_id']}")
    return manifest, rows


def _prompt_ids(tokenizer: Any, prompt: str) -> List[int]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if not token_ids:
        raise ValueError("Chat-formatted prompt is empty.")
    return [int(value) for value in token_ids]


def _trim_completion(token_ids: Sequence[int], eos_token_id: int | None) -> Tuple[List[int], bool]:
    values = [int(value) for value in token_ids]
    if eos_token_id is None or eos_token_id not in values:
        return values, False
    index = values.index(int(eos_token_id))
    return values[: index + 1], True


def generate_completion_ids(
    student: Any,
    tokenizer: Any,
    prompt: str,
    *,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    valid_vocab_size: int,
) -> List[Dict[str, Any]]:
    """Sample raw student token IDs without materializing generation logits."""

    import torch

    prompt_ids = _prompt_ids(tokenizer, prompt)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    was_training = bool(student.training)
    student.eval()
    student.config.use_cache = True
    invalid_token_ids = list(range(int(valid_vocab_size), int(student.config.vocab_size)))
    with torch.inference_mode():
        generated = student.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(num_return_sequences),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            suppress_tokens=invalid_token_ids or None,
            use_cache=True,
        )
    student.config.use_cache = False
    if was_training:
        student.train()
    results = []
    for sequence in generated:
        completion, eos_emitted = _trim_completion(
            sequence[len(prompt_ids) :].detach().cpu().tolist(),
            tokenizer.eos_token_id,
        )
        if not completion:
            raise RuntimeError("Student generation produced an empty completion.")
        results.append(
            {
                "completion_token_ids": completion,
                "eos_emitted": eos_emitted,
                "hit_max_new_tokens": not eos_emitted and len(completion) >= int(max_new_tokens),
            }
        )
    if len(results) != int(num_return_sequences):
        raise RuntimeError("Student generation returned the wrong number of sequences.")
    return results


def generate_greedy_completion_ids(
    student: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    valid_vocab_size: int,
) -> Dict[str, Any]:
    """Generate one deterministic completion for frozen reference lengths."""

    import torch

    prompt_ids = _prompt_ids(tokenizer, prompt)
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    was_training = bool(student.training)
    student.eval()
    student.config.use_cache = True
    invalid_token_ids = list(range(int(valid_vocab_size), int(student.config.vocab_size)))
    with torch.inference_mode():
        generated = student.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            suppress_tokens=invalid_token_ids or None,
            use_cache=True,
        )[0]
    student.config.use_cache = False
    if was_training:
        student.train()
    completion, eos_emitted = _trim_completion(
        generated[len(prompt_ids) :].detach().cpu().tolist(),
        tokenizer.eos_token_id,
    )
    if not completion:
        raise RuntimeError("Student generation produced an empty greedy completion.")
    return {
        "completion_token_ids": completion,
        "eos_emitted": eos_emitted,
        "hit_max_new_tokens": not eos_emitted and len(completion) >= int(max_new_tokens),
    }


def score_completion_tokens(
    model: Any,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    *,
    top_k: int = 0,
    requires_grad: bool = False,
    valid_vocab_size: int,
) -> Dict[str, Any]:
    """Score exact completion token IDs under one prompt context."""

    import torch
    import torch.nn.functional as functional

    full_ids = [int(value) for value in prompt_ids] + [int(value) for value in completion_ids]
    input_tensor = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    context = torch.enable_grad() if requires_grad else torch.inference_mode()
    with context:
        logits = causal_completion_logits(model, input_tensor, len(prompt_ids))
        if valid_vocab_size <= 0 or valid_vocab_size > logits.shape[-1]:
            raise ValueError("valid_vocab_size is outside the model output vocabulary.")
        if any(int(token_id) < 0 or int(token_id) >= valid_vocab_size for token_id in completion_ids):
            raise ValueError("Completion contains a token outside the shared valid vocabulary.")
        logits = logits[..., : int(valid_vocab_size)]
        log_probs = functional.log_softmax(logits.float(), dim=-1)
        targets = torch.tensor(completion_ids, dtype=torch.long, device=log_probs.device)
        sampled = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        result: Dict[str, Any] = {"sampled_logprobs": sampled}
        if top_k:
            if top_k <= 0 or top_k > log_probs.shape[-1]:
                raise ValueError("top_k is outside the model vocabulary.")
            values, ids = torch.topk(log_probs, k=int(top_k), dim=-1)
            result.update({"topk_logprobs": values, "topk_ids": ids})
    return result


def collect_scored_rollouts(
    protocol: Mapping[str, Any],
    *,
    arm: str,
    references: Sequence[Mapping[str, Any]],
    teacher: Any,
    student: Any,
    tokenizer: Any,
    diagnostic_top_k: int,
    diagnostic_rollouts: int,
) -> List[Dict[str, Any]]:
    """Generate and score one fresh on-policy rollout batch."""

    if arm not in OPD_ARMS:
        raise ValueError(f"Unknown OPD arm: {arm}")
    training = protocol["training"]
    valid_vocab_size = len(tokenizer)
    if valid_vocab_size != int(protocol["models"]["tokenizer"]["expected_length"]):
        raise ValueError("Runtime tokenizer length does not match the registered valid vocabulary.")
    rows: List[Dict[str, Any]] = []
    teacher.eval()
    student.eval()
    for reference in references:
        student_prompt = prompt_for_arm(reference, arm)
        teacher_prompt = str(reference["standard_prompt"])
        seed = stable_generation_seed(
            int(training["generation_seed"]),
            arm,
            "opd_rollout",
            str(reference["problem_id"]),
        )
        generated = generate_completion_ids(
            student,
            tokenizer,
            student_prompt,
            num_return_sequences=int(training["rollouts_per_prompt"]),
            max_new_tokens=int(training["max_new_tokens"]),
            temperature=float(training["sampling_temperature"]),
            top_p=float(training["top_p"]),
            seed=seed,
            valid_vocab_size=valid_vocab_size,
        )
        student_prompt_ids = _prompt_ids(tokenizer, student_prompt)
        teacher_prompt_ids = _prompt_ids(tokenizer, teacher_prompt)
        for candidate_index, generated_row in enumerate(generated):
            completion_ids = generated_row["completion_token_ids"]
            maximum_sequence_tokens = int(training["max_length"])
            if len(student_prompt_ids) + len(completion_ids) > maximum_sequence_tokens:
                raise ValueError(
                    "Student prompt plus sampled completion exceeds training.max_length: "
                    f"problem={reference['problem_id']} arm={arm}"
                )
            if len(teacher_prompt_ids) + len(completion_ids) > maximum_sequence_tokens:
                raise ValueError(
                    "Teacher prompt plus sampled completion exceeds training.max_length: "
                    f"problem={reference['problem_id']} arm={arm}"
                )
            capture_topk = len(rows) < int(diagnostic_rollouts)
            old_score = score_completion_tokens(
                student,
                student_prompt_ids,
                completion_ids,
                top_k=diagnostic_top_k if capture_topk else 0,
                valid_vocab_size=valid_vocab_size,
            )
            teacher_score = score_completion_tokens(
                teacher,
                teacher_prompt_ids,
                completion_ids,
                top_k=diagnostic_top_k if capture_topk else 0,
                valid_vocab_size=valid_vocab_size,
            )
            advantages = sampled_token_advantage(
                teacher_score["sampled_logprobs"],
                old_score["sampled_logprobs"],
            )
            response_ids_without_eos = list(completion_ids)
            if tokenizer.eos_token_id is not None and response_ids_without_eos[-1] == tokenizer.eos_token_id:
                response_ids_without_eos = response_ids_without_eos[:-1]
            response_text = tokenizer.decode(response_ids_without_eos, skip_special_tokens=True).strip()
            predicted = extract_final_answer(response_text)
            output_tokens = len(response_ids_without_eos)
            lower = int(reference["concise_lower_tokens"])
            upper = int(reference["concise_upper_tokens"])
            row: Dict[str, Any] = {
                "problem_id": str(reference["problem_id"]),
                "source_index": int(reference["source_index"]),
                "arm": arm,
                "candidate_index": candidate_index,
                "student_prompt": student_prompt,
                "teacher_prompt": teacher_prompt,
                "teacher_context_mode": TEACHER_CONTEXT_MODE,
                "valid_vocab_size": valid_vocab_size,
                "completion_token_ids": [int(value) for value in completion_ids],
                "response_text": response_text,
                "predicted_answer": predicted,
                "is_correct": verify_answer(predicted, str(reference["gold_answer"])),
                "output_token_count": output_tokens,
                "eos_emitted": bool(generated_row["eos_emitted"]),
                "hit_max_new_tokens": bool(generated_row["hit_max_new_tokens"]),
                "concise_lower_tokens": lower,
                "concise_upper_tokens": upper,
                "in_length_band": lower <= output_tokens <= upper,
                "old_student_logprobs": [float(value) for value in old_score["sampled_logprobs"].cpu()],
                "teacher_logprobs": [float(value) for value in teacher_score["sampled_logprobs"].cpu()],
                "advantages": [float(value) for value in advantages.cpu()],
                "mean_advantage": float(advantages.mean().cpu()),
                "reward_definition": "logp_teacher_minus_logp_old_student",
                "correctness_is_diagnostic_only": True,
                "length_is_diagnostic_only": True,
                "scalar_reward_used": False,
                "value_head_used": False,
            }
            if capture_topk:
                student_topk = old_score["topk_ids"].detach().cpu()
                teacher_topk = teacher_score["topk_ids"].detach().cpu()
                row["topk_diagnostic"] = {
                    "k": int(diagnostic_top_k),
                    "overlap_ratio": topk_overlap(student_topk, teacher_topk),
                    "student_ids": student_topk.tolist(),
                    "student_logprobs": old_score["topk_logprobs"].detach().cpu().tolist(),
                    "teacher_ids": teacher_topk.tolist(),
                    "teacher_logprobs": teacher_score["topk_logprobs"].detach().cpu().tolist(),
                }
            rows.append(row)
            del old_score, teacher_score, advantages
    student.train()
    return rows


def optimize_rollout_batch(
    protocol: Mapping[str, Any],
    *,
    student: Any,
    tokenizer: Any,
    rollouts: Sequence[Mapping[str, Any]],
    optimizer: Any,
    scheduler: Any,
) -> Dict[str, Any]:
    """Apply one registered PPO epoch over a fresh rollout batch."""

    import torch

    if not rollouts:
        raise ValueError("Cannot optimize an empty rollout batch.")
    training = protocol["training"]
    objective = protocol["objective"]
    mini_batch = int(training["mini_batch_rollouts"])
    ordered = list(rollouts)
    random.Random(int(training["seed"]) + int(rollouts[0].get("batch_index", 0))).shuffle(ordered)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer_steps = 0
    total_clipped = 0.0
    total_tokens = 0
    loss_token_sum = 0.0
    grad_norms: List[float] = []
    # Keep the parent model in training mode so Transformers activates gradient
    # checkpointing, but disable every dropout module so policy probabilities
    # remain comparable to deterministic rollout-time probabilities.
    student.train()
    disabled_dropout_modules = 0
    for module in student.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
            disabled_dropout_modules += 1
    for start in range(0, len(ordered), mini_batch):
        subset = ordered[start : start + mini_batch]
        mini_tokens = sum(len(row["completion_token_ids"]) for row in subset)
        if mini_tokens <= 0:
            raise ValueError("OPD mini-batch has no completion tokens.")
        optimizer.zero_grad(set_to_none=True)
        mini_loss_sum = 0.0
        mini_clipped_sum = 0.0
        for row in subset:
            prompt_ids = _prompt_ids(tokenizer, str(row["student_prompt"]))
            score = score_completion_tokens(
                student,
                prompt_ids,
                row["completion_token_ids"],
                requires_grad=True,
                valid_vocab_size=len(tokenizer),
            )
            device = score["sampled_logprobs"].device
            old = torch.tensor(row["old_student_logprobs"], dtype=torch.float32, device=device)
            advantages = torch.tensor(row["advantages"], dtype=torch.float32, device=device)
            loss_sum, metrics = clipped_opd_loss(
                score["sampled_logprobs"],
                old,
                advantages,
                clip_ratio=float(objective["clip_ratio"]),
                reduction="sum",
            )
            (loss_sum / mini_tokens).backward()
            token_count = int(metrics["token_count"])
            mini_loss_sum += float(loss_sum.detach())
            mini_clipped_sum += float(metrics["clip_fraction"]) * token_count
            del score, old, advantages, loss_sum, metrics
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            float(training["max_grad_norm"]),
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise RuntimeError("Non-finite OPD gradient norm.")
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        grad_norms.append(float(grad_norm))
        total_tokens += mini_tokens
        loss_token_sum += mini_loss_sum
        total_clipped += mini_clipped_sum
    return {
        "optimizer_steps": optimizer_steps,
        "token_count": total_tokens,
        "mean_loss": loss_token_sum / total_tokens,
        "clip_fraction": total_clipped / total_tokens,
        "mean_grad_norm": mean(grad_norms),
        "max_grad_norm_observed": max(grad_norms),
        "disabled_dropout_modules": disabled_dropout_modules,
    }


def _save_resume_state(
    runtime_dir: Path,
    student: Any,
    optimizer: Any,
    scheduler: Any,
    state: Mapping[str, Any],
) -> None:
    import torch

    resume_root = runtime_dir / "resume"
    next_bundle = resume_root / "next_bundle"
    current_bundle = resume_root / "current_bundle"
    previous_bundle = resume_root / "previous_bundle"
    resume_root.mkdir(parents=True, exist_ok=True)
    if next_bundle.exists():
        shutil.rmtree(next_bundle)
    next_bundle.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(next_bundle / "adapter", safe_serialization=True)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        next_bundle / "optimizer.pt",
    )
    write_json(next_bundle / "state.json", state)
    if previous_bundle.exists():
        shutil.rmtree(previous_bundle)
    if current_bundle.exists():
        current_bundle.rename(previous_bundle)
    next_bundle.rename(current_bundle)
    if previous_bundle.exists():
        shutil.rmtree(previous_bundle)


def _recover_resume_bundle(runtime_dir: Path) -> Path:
    """Return the last complete resume bundle, recovering an interrupted rename."""

    resume_root = runtime_dir / "resume"
    current_bundle = resume_root / "current_bundle"
    previous_bundle = resume_root / "previous_bundle"
    required = ("state.json", "optimizer.pt", "adapter")

    def complete(bundle: Path) -> bool:
        return all((bundle / name).exists() for name in required)

    if complete(current_bundle):
        return current_bundle
    if complete(previous_bundle):
        if current_bundle.exists():
            shutil.rmtree(current_bundle)
        previous_bundle.rename(current_bundle)
        return current_bundle
    raise FileNotFoundError(f"No complete resume bundle exists under {resume_root}")


def train_opd_arm(
    protocol: Mapping[str, Any],
    *,
    arm: str,
    reference_manifest_path: str | Path,
    runtime_dir: str | Path,
    rollout_dir: str | Path,
    resume: bool,
    max_prompt_batches: int | None = None,
    logger: Any,
) -> Dict[str, Any]:
    """Train one OPD arm and persist auditable rollout shards."""

    import torch
    from transformers import get_linear_schedule_with_warmup

    validate_opd_protocol(protocol)
    if arm not in OPD_ARMS:
        raise ValueError(f"Unknown OPD arm: {arm}")
    _manifest, references = validate_reference_manifest(protocol, reference_manifest_path)
    training = protocol["training"]
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    order = list(range(len(references)))
    random.Random(seed).shuffle(order)
    prompt_batches = [
        order[start : start + int(training["prompts_per_batch"])]
        for start in range(0, len(order), int(training["prompts_per_batch"]))
    ]
    if max_prompt_batches is not None:
        if max_prompt_batches <= 0:
            raise ValueError("max_prompt_batches must be positive.")
        prompt_batches = prompt_batches[: int(max_prompt_batches)]
    runtime_path = Path(runtime_dir)
    rollout_path = Path(rollout_dir)
    if runtime_path.exists() and not resume:
        raise FileExistsError(f"Runtime directory already exists: {runtime_path}")
    runtime_path.mkdir(parents=True, exist_ok=True)
    rollout_path.mkdir(parents=True, exist_ok=True)
    resume_bundle: Path | None = None
    completed_batches = 0
    prior_elapsed_seconds = 0.0
    batch_metrics: List[Dict[str, Any]] = []
    rollout_artifacts: List[Dict[str, Any]] = []
    adapter_path = None
    if resume:
        resume_bundle = _recover_resume_bundle(runtime_path)
        state = read_json(resume_bundle / "state.json")
        if state.get("protocol_hash") != protocol_hash(protocol) or state.get("arm") != arm:
            raise ValueError("Resume state protocol/arm mismatch.")
        completed_batches = int(state["completed_prompt_batches"])
        if not 0 <= completed_batches <= len(prompt_batches):
            raise ValueError("Resume state completed-batch count is invalid.")
        if int(state.get("total_prompt_batches", -1)) != len(prompt_batches):
            raise ValueError("Resume state target prompt-batch count mismatch.")
        batch_metrics = [dict(row) for row in state.get("batch_metrics", [])]
        if len(batch_metrics) != completed_batches:
            raise ValueError("Resume state batch metrics are incomplete.")
        if [int(row.get("batch_index", -1)) for row in batch_metrics] != list(
            range(completed_batches)
        ):
            raise ValueError("Resume state batch-metric indices are not contiguous.")
        prior_elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
        adapter_path = resume_bundle / "adapter"
        expected_resume_shards = {
            rollout_path / f"batch_{batch_index:05d}.jsonl.gz"
            for batch_index in range(completed_batches)
        }
        observed_resume_shards = set(rollout_path.glob("batch_*.jsonl.gz"))
        if observed_resume_shards != expected_resume_shards:
            raise ValueError(
                "Resume rollout directory has missing or uncommitted extra shards; "
                "audit it before retrying."
            )
        for batch_index in range(completed_batches):
            expected = rollout_path / f"batch_{batch_index:05d}.jsonl.gz"
            if not expected.is_file():
                raise FileNotFoundError(f"Resume rollout shard is missing: {expected}")
            expected_records = (
                len(prompt_batches[batch_index]) * int(training["rollouts_per_prompt"])
            )
            existing_rows = list(read_gzip_jsonl(expected))
            actual_records = len(existing_rows)
            if actual_records != expected_records:
                raise ValueError(f"Resume rollout shard count mismatch: {expected}")
            rollout_artifacts.append(
                {
                    "path": str(expected),
                    "sha256": file_sha256(expected),
                    "records": actual_records,
                    "tokens": sum(
                        len(row["completion_token_ids"]) for row in existing_rows
                    ),
                }
            )
        if state.get("rollout_artifacts") != rollout_artifacts:
            raise ValueError("Resume rollout artifact hashes do not match the saved state.")
    teacher, student, tokenizer, _valid_vocab, model_evidence = load_teacher_and_student(
        protocol,
        student_adapter=adapter_path,
        train_student=True,
    )
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
    )
    total_optimizer_steps = sum(
        math.ceil(
            len(batch) * int(training["rollouts_per_prompt"])
            / int(training["mini_batch_rollouts"])
        )
        for batch in prompt_batches
    )
    warmup_steps = math.ceil(float(training["warmup_ratio"]) * total_optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)
    if resume:
        if resume_bundle is None:
            raise RuntimeError("Resume bundle was not initialized.")
        checkpoint = torch.load(resume_bundle / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    started = time.time()
    for batch_index in range(completed_batches, len(prompt_batches)):
        batch_references = [references[index] for index in prompt_batches[batch_index]]
        rollouts = collect_scored_rollouts(
            protocol,
            arm=arm,
            references=batch_references,
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            diagnostic_top_k=int(protocol["diagnostics"]["top_k"]),
            diagnostic_rollouts=(
                int(protocol["diagnostics"]["top_k_rollouts"])
                if batch_index == 0
                else 0
            ),
        )
        for row in rollouts:
            row["batch_index"] = batch_index
            row["protocol_hash"] = protocol_hash(protocol)
        optimization = optimize_rollout_batch(
            protocol,
            student=student,
            tokenizer=tokenizer,
            rollouts=rollouts,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        shard_path = rollout_path / f"batch_{batch_index:05d}.jsonl.gz"
        write_gzip_jsonl(shard_path, rollouts)
        rollout_artifacts.append(
            {
                "path": str(shard_path),
                "sha256": file_sha256(shard_path),
                "records": len(rollouts),
                "tokens": sum(len(row["completion_token_ids"]) for row in rollouts),
            }
        )
        metric = {
            "batch_index": batch_index,
            "prompt_count": len(batch_references),
            "rollout_count": len(rollouts),
            "mean_advantage": mean(float(row["mean_advantage"]) for row in rollouts),
            "mean_output_tokens": mean(float(row["output_token_count"]) for row in rollouts),
            "diagnostic_accuracy": mean(float(bool(row["is_correct"])) for row in rollouts),
            "concise_in_band_rate": mean(float(bool(row["in_length_band"])) for row in rollouts),
            **optimization,
        }
        batch_metrics.append(metric)
        state = {
            "status": "running",
            "arm": arm,
            "protocol_hash": protocol_hash(protocol),
            "completed_prompt_batches": batch_index + 1,
            "total_prompt_batches": len(prompt_batches),
            "last_rollout_path": str(shard_path),
            "last_rollout_sha256": file_sha256(shard_path),
            "elapsed_seconds": prior_elapsed_seconds + time.time() - started,
            "batch_metrics": batch_metrics,
            "rollout_artifacts": rollout_artifacts,
        }
        _save_resume_state(runtime_path, student, optimizer, scheduler, state)
        logger.info(
            "opd_progress arm=%s batch=%d/%d prompts=%d rollouts=%d mean_advantage=%.6f mean_tokens=%.1f",
            arm,
            batch_index + 1,
            len(prompt_batches),
            len(batch_references),
            len(rollouts),
            metric["mean_advantage"],
            metric["mean_output_tokens"],
        )

    all_shards = sorted(rollout_path.glob("batch_*.jsonl.gz"))
    rollout_artifacts = [
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "records": sum(1 for _ in read_gzip_jsonl(path)),
        }
        for path in all_shards
    ]
    total_rollouts = sum(int(item["records"]) for item in rollout_artifacts)
    expected_rollouts = sum(len(batch) for batch in prompt_batches) * int(training["rollouts_per_prompt"])
    if total_rollouts != expected_rollouts:
        raise RuntimeError(
            f"Rollout count mismatch: expected={expected_rollouts} actual={total_rollouts}"
        )
    student.save_pretrained(runtime_path, safe_serialization=True)
    metrics = {
        "status": "trained",
        "method": "pure_sampled_token_opd",
        "arm": arm,
        "seed": seed,
        "protocol_hash": protocol_hash(protocol),
        "reference_manifest_path": str(reference_manifest_path),
        "reference_manifest_sha256": file_sha256(reference_manifest_path),
        "prompt_batches": len(prompt_batches),
        "prompts": sum(len(batch) for batch in prompt_batches),
        "rollouts": total_rollouts,
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in batch_metrics),
        "sampled_tokens": sum(
            sum(len(row["completion_token_ids"]) for row in read_gzip_jsonl(path))
            for path in all_shards
        ),
        "elapsed_seconds": prior_elapsed_seconds + time.time() - started,
        "model_evidence": model_evidence,
        "objective": dict(protocol["objective"]),
        "gold_labels_used_in_loss": False,
        "length_used_in_loss": False,
        "batch_metrics": batch_metrics,
        "runtime": runtime_metadata(),
    }
    write_json(runtime_path / "training_metrics.json", metrics)
    rollout_manifest = {
        "status": "complete",
        "arm": arm,
        "protocol_hash": protocol_hash(protocol),
        "record_count": total_rollouts,
        "expected_record_count": expected_rollouts,
        "shards": rollout_artifacts,
        "gold_labels_used_in_loss": False,
        "length_used_in_loss": False,
    }
    write_json(rollout_path / "rollout_manifest.json", rollout_manifest)
    return metrics


def publish_opd_adapter(
    protocol: Mapping[str, Any],
    *,
    arm: str,
    runtime_dir: str | Path,
    publish_dir: str | Path,
    rollout_manifest_path: str | Path,
    reference_manifest_path: str | Path,
    source_paths: Sequence[str | Path],
    stage: str = "pilot",
) -> Dict[str, Any]:
    """Publish one OPD LoRA adapter with hash-bound pure-loss evidence."""

    evidence = {
        "status": "complete",
        "stage": stage,
        "method": "pure_sampled_token_opd",
        "objective": OBJECTIVE_NAME,
        "arm": arm,
        "seed": int(protocol["training"]["seed"]),
        "protocol_hash": protocol_hash(protocol),
        "rollout_manifest_path": str(rollout_manifest_path),
        "rollout_manifest_sha256": file_sha256(rollout_manifest_path),
        "reference_manifest_path": str(reference_manifest_path),
        "reference_manifest_sha256": file_sha256(reference_manifest_path),
        "gold_labels_used_in_loss": False,
        "length_used_in_loss": False,
        "scalar_reward_used": False,
        "value_head_used": False,
        "source_sha256": {str(path): file_sha256(path) for path in source_paths},
    }
    return publish_adapter(runtime_dir, publish_dir, evidence=evidence)


def validated_opd_adapter(
    protocol: Mapping[str, Any],
    *,
    arm: str,
    adapter_dir: str | Path,
    stage: str = "pilot",
) -> Dict[str, Any] | None:
    marker = validated_training_marker(adapter_dir)
    if marker is None:
        return None
    checks = {
        "status": "complete",
        "method": "pure_sampled_token_opd",
        "stage": stage,
        "objective": OBJECTIVE_NAME,
        "arm": arm,
        "protocol_hash": protocol_hash(protocol),
        "gold_labels_used_in_loss": False,
        "length_used_in_loss": False,
        "scalar_reward_used": False,
        "value_head_used": False,
    }
    if any(marker.get(key) != value for key, value in checks.items()):
        return None
    rollout_manifest = Path(str(marker.get("rollout_manifest_path", "")))
    if not rollout_manifest.is_file() or file_sha256(rollout_manifest) != marker.get(
        "rollout_manifest_sha256"
    ):
        return None
    reference_manifest = Path(str(marker.get("reference_manifest_path", "")))
    if not reference_manifest.is_file() or file_sha256(reference_manifest) != marker.get(
        "reference_manifest_sha256"
    ):
        return None
    return marker


def validated_temporary_runtime_path(runtime_dir: str | Path) -> Path:
    """Resolve and validate the explicitly scoped node-local runtime directory."""

    path = Path(runtime_dir)
    allowed_prefixes = (Path("/var/tmp"), Path("/tmp"))
    resolved = path.resolve()
    if not any(prefix == resolved or prefix in resolved.parents for prefix in allowed_prefixes):
        raise ValueError(f"OPD runtime directory must be under /var/tmp or /tmp: {resolved}")
    if resolved in allowed_prefixes:
        raise ValueError(f"OPD runtime directory cannot be a broad temporary root: {resolved}")
    return resolved


def remove_runtime_after_publish(runtime_dir: str | Path) -> None:
    """Remove only the explicit node-local runtime directory after publication."""

    resolved = validated_temporary_runtime_path(runtime_dir)
    if resolved.exists():
        shutil.rmtree(resolved)
