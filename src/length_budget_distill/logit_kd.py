"""Reusable utilities for exact online logit distillation and logit artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_protocol(path: str | Path) -> Dict[str, Any]:
    protocol = read_json(path)
    required = {"experiment_name", "parent", "models", "budgets", "kd", "training", "validation", "formal", "outputs"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"KD protocol is missing keys: {sorted(missing)}")
    if set(protocol["budgets"]) != {"short_128", "medium_256", "long_512"}:
        raise ValueError("KD protocol must contain exactly short_128, medium_256, and long_512.")
    if int(protocol["training"].get("seed", -1)) != 17:
        raise ValueError("The registered KD protocol is restricted to seed 17.")
    if not bool(protocol["kd"].get("completion_only")):
        raise ValueError("The registered KD protocol requires completion-only loss.")
    return protocol


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return canonical_sha256(protocol)


def float_slug(value: float) -> str:
    rendered = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return rendered


def kd_run_name(budget_name: str, alpha: float, temperature: float) -> str:
    return f"{budget_name}__a{float_slug(alpha)}__t{float_slug(temperature)}__seed_17"


def validate_budget_dataset(protocol: Mapping[str, Any], budget_name: str) -> Tuple[Path, List[Dict[str, Any]]]:
    if budget_name not in protocol["budgets"]:
        raise ValueError(f"Unknown budget: {budget_name}")
    budget = protocol["budgets"][budget_name]
    path = resolve_project_path(str(budget["train_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing registered KD training data: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != budget["train_sha256"]:
        raise ValueError(
            f"KD training-data hash mismatch for {budget_name}: expected={budget['train_sha256']} actual={actual_hash}"
        )
    rows = read_jsonl(path)
    if len(rows) != int(budget["expected_records"]):
        raise ValueError(
            f"KD training-data count mismatch for {budget_name}: expected={budget['expected_records']} actual={len(rows)}"
        )
    seen = set()
    for row in rows:
        record_id = str(row.get("id"))
        if not record_id or record_id in seen:
            raise ValueError(f"Missing or duplicate KD record id in {path}: {record_id!r}")
        seen.add(record_id)
        metadata = row.get("metadata", {})
        if metadata.get("budget_name") != budget_name:
            raise ValueError(f"KD record budget mismatch: {record_id}")
        if metadata.get("generator_name") != "qwen2p5_7b":
            raise ValueError(f"KD record does not come from the registered 7B teacher: {record_id}")
        if not bool(metadata.get("is_correct")) or not bool(metadata.get("budget_compliant")):
            raise ValueError(f"KD record is not verified-correct and budget-compliant: {record_id}")
    return path, rows


def load_and_validate_tokenizers(protocol: Mapping[str, Any]) -> Tuple[Any, Any, int, Dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
        from transformers.utils.hub import cached_file
    except ImportError as exc:
        raise ImportError("Install transformers before running logit KD.") from exc

    tokenizer_config = protocol["models"]["tokenizer"]
    teacher_config = protocol["models"]["teacher"]
    student_config = protocol["models"]["student"]
    common_kwargs = {"local_files_only": bool(int(os.environ.get("LBD_LOCAL_FILES_ONLY", "0")))}
    student_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_name"],
        revision=tokenizer_config["revision"],
        **common_kwargs,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_config["model_name"],
        revision=teacher_config["revision"],
        **common_kwargs,
    )
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher and student tokenizer vocabularies are not identical.")
    valid_vocab_size = len(student_tokenizer)
    if valid_vocab_size != int(tokenizer_config["expected_length"]):
        raise ValueError(
            f"Tokenizer length mismatch: expected={tokenizer_config['expected_length']} actual={valid_vocab_size}"
        )
    try:
        tokenizer_json = cached_file(
            tokenizer_config["model_name"],
            "tokenizer.json",
            revision=tokenizer_config["revision"],
            local_files_only=common_kwargs["local_files_only"],
        )
    except Exception as exc:  # pragma: no cover - environment-specific download errors
        raise RuntimeError("Could not resolve the registered tokenizer.json.") from exc
    actual_tokenizer_hash = file_sha256(tokenizer_json)
    if actual_tokenizer_hash != tokenizer_config["expected_tokenizer_json_sha256"]:
        raise ValueError(
            "Tokenizer file hash mismatch: "
            f"expected={tokenizer_config['expected_tokenizer_json_sha256']} actual={actual_tokenizer_hash}"
        )
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    evidence = {
        "valid_vocab_size": valid_vocab_size,
        "tokenizer_json_path": str(tokenizer_json),
        "tokenizer_json_sha256": actual_tokenizer_hash,
        "teacher_tokenizer_length": len(teacher_tokenizer),
        "student_tokenizer_length": len(student_tokenizer),
    }
    return student_tokenizer, teacher_tokenizer, valid_vocab_size, evidence


def tokenize_completion_record(tokenizer: Any, row: Mapping[str, Any], max_length: int) -> Dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    completion = str(row.get("completion", ""))
    if not prompt or not completion:
        raise ValueError(f"KD record is missing prompt/completion: {row.get('id')}")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Chat-template prompt is not a prefix of the full sequence: {row.get('id')}")
    if len(full_ids) > max_length:
        raise ValueError(
            f"KD sequence exceeds registered max_length for {row.get('id')}: length={len(full_ids)} max={max_length}"
        )
    targets = full_ids[len(prompt_ids) :]
    if not targets:
        raise ValueError(f"KD record has no completion tokens after formatting: {row.get('id')}")
    return {
        "record_id": str(row.get("id")),
        "problem_id": str(row.get("metadata", {}).get("problem_id", "")),
        "input_ids": full_ids,
        "prompt_token_count": len(prompt_ids),
        "completion_token_count": len(targets),
        "target_ids": targets,
    }


def hybrid_kd_loss(
    student_logits: Any,
    teacher_logits: Any,
    target_ids: Any,
    *,
    alpha: float,
    temperature: float,
    valid_vocab_size: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Compute hard CE plus exact forward KL over the shared valid vocabulary."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("Install torch before computing KD loss.") from exc
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if student_logits.ndim != 2 or teacher_logits.ndim != 2 or target_ids.ndim != 1:
        raise ValueError("Expected [tokens, vocab] logits and [tokens] targets.")
    if student_logits.shape[0] != teacher_logits.shape[0] or student_logits.shape[0] != target_ids.shape[0]:
        raise ValueError("Teacher, student, and target token counts do not match.")
    if student_logits.shape[-1] < valid_vocab_size or teacher_logits.shape[-1] < valid_vocab_size:
        raise ValueError("A model output head is smaller than the shared valid vocabulary.")
    if bool(torch.any(target_ids < 0)) or bool(torch.any(target_ids >= valid_vocab_size)):
        raise ValueError("A target token lies outside the shared valid vocabulary.")

    ce = functional.cross_entropy(student_logits.float(), target_ids, reduction="mean")
    scaled_teacher = teacher_logits[:, :valid_vocab_size].float() / float(temperature)
    scaled_student = student_logits[:, :valid_vocab_size].float() / float(temperature)
    teacher_log_probs = functional.log_softmax(scaled_teacher, dim=-1)
    student_log_probs = functional.log_softmax(scaled_student, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl_by_token = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    kd = kl_by_token.mean()
    loss = (1.0 - float(alpha)) * ce + float(alpha) * (float(temperature) ** 2) * kd
    metrics = {
        "ce": ce.detach(),
        "kd": kd.detach(),
        "loss": loss.detach(),
        "kl_by_token": kl_by_token.detach(),
    }
    return loss, metrics


def invalid_vocab_probability_mass(logits: Any, valid_vocab_size: int) -> Any:
    """Return probability mass assigned to padded output-head rows."""

    import torch

    full_lse = torch.logsumexp(logits.float(), dim=-1)
    valid_lse = torch.logsumexp(logits[..., :valid_vocab_size].float(), dim=-1)
    return torch.clamp(1.0 - torch.exp(valid_lse - full_lse), min=0.0, max=1.0)


def load_teacher_and_student(
    protocol: Mapping[str, Any],
    *,
    student_adapter: str | Path | None = None,
    train_student: bool = False,
) -> Tuple[Any, Any, Any, int, Dict[str, Any]]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install torch, transformers, and peft before running logit KD.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Logit KD requires a CUDA GPU.")
    student_tokenizer, _teacher_tokenizer, valid_vocab_size, tokenizer_evidence = load_and_validate_tokenizers(protocol)
    teacher_cfg = protocol["models"]["teacher"]
    student_cfg = protocol["models"]["student"]
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    local_only = bool(int(os.environ.get("LBD_LOCAL_FILES_ONLY", "0")))
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_cfg["model_name"],
        revision=teacher_cfg["revision"],
        torch_dtype=dtype_by_name[teacher_cfg["torch_dtype"]],
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    ).to("cuda")
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.config.use_cache = False

    student = AutoModelForCausalLM.from_pretrained(
        student_cfg["model_name"],
        revision=student_cfg["revision"],
        torch_dtype=dtype_by_name[student_cfg["torch_dtype"]],
        local_files_only=local_only,
        low_cpu_mem_usage=True,
    ).to("cuda")
    student.config.use_cache = False
    if student_adapter is not None:
        adapter_path = resolve_project_path(student_adapter)
        student = PeftModel.from_pretrained(student, adapter_path, is_trainable=train_student)
    elif train_student:
        lora = student_cfg["lora"]
        student = get_peft_model(
            student,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=lora["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
    if train_student:
        student.train()
        if bool(protocol["training"].get("gradient_checkpointing", True)):
            student.gradient_checkpointing_enable()
            enable_input_grads = getattr(student, "enable_input_require_grads", None)
            if callable(enable_input_grads):
                enable_input_grads()
        if not any(parameter.requires_grad for parameter in student.parameters()):
            raise ValueError("KD student has no trainable parameters.")
    else:
        student.requires_grad_(False)
        student.eval()
    evidence = {
        **tokenizer_evidence,
        "teacher_output_vocab_size": int(teacher.config.vocab_size),
        "student_output_vocab_size": int(student.config.vocab_size),
        "teacher_model_name": teacher_cfg["model_name"],
        "teacher_revision": teacher_cfg["revision"],
        "student_model_name": student_cfg["model_name"],
        "student_revision": student_cfg["revision"],
        "student_adapter": str(student_adapter) if student_adapter is not None else None,
    }
    return teacher, student, student_tokenizer, valid_vocab_size, evidence


def causal_completion_logits(model: Any, input_ids: Any, prompt_token_count: int) -> Any:
    """Return logits aligned to completion targets, excluding the unused final logit."""

    completion_count = int(input_ids.shape[1]) - int(prompt_token_count)
    if completion_count <= 0:
        raise ValueError("Input sequence has no completion tokens.")
    outputs = model(
        input_ids=input_ids,
        attention_mask=input_ids.new_ones(input_ids.shape),
        use_cache=False,
        num_logits_to_keep=completion_count + 1,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    if logits.shape[1] != completion_count:
        raise RuntimeError(
            f"Completion-logit alignment failed: expected={completion_count} actual={logits.shape[1]}"
        )
    return logits[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_logit_kd_adapter(
    protocol: Mapping[str, Any],
    *,
    budget_name: str,
    alpha: float,
    temperature: float,
    runtime_output_dir: str | Path,
    logger: Any,
) -> Dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    training = protocol["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    data_path, rows = validate_budget_dataset(protocol, budget_name)
    teacher, student, tokenizer, valid_vocab_size, model_evidence = load_teacher_and_student(
        protocol,
        train_student=True,
    )
    max_length = int(training["max_length"])
    encoded = [tokenize_completion_record(tokenizer, row, max_length) for row in rows]
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    accumulation = int(training["gradient_accumulation_steps"])
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    optimizer_steps = math.ceil(len(order) / accumulation) * int(training["num_train_epochs"])
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]), weight_decay=0.0)
    warmup_steps = math.ceil(float(training["warmup_ratio"]) * optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, optimizer_steps)
    report_every = int(training.get("report_every_optimizer_steps", 10))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    totals = {
        "loss_sum": 0.0,
        "ce_sum": 0.0,
        "kd_sum": 0.0,
        "token_count": 0,
        "teacher_invalid_mass_sum": 0.0,
        "student_invalid_mass_sum": 0.0,
    }
    optimizer_step = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(training["num_train_epochs"])):
        if epoch > 0:
            random.Random(seed + epoch).shuffle(order)
        for group_start in range(0, len(order), accumulation):
            group = order[group_start : group_start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                item = encoded[index]
                input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device="cuda")
                targets = torch.tensor(item["target_ids"], dtype=torch.long, device="cuda")
                with torch.inference_mode():
                    teacher_logits = causal_completion_logits(teacher, input_ids, item["prompt_token_count"])
                student_logits = causal_completion_logits(student, input_ids, item["prompt_token_count"])
                loss, metrics = hybrid_kd_loss(
                    student_logits,
                    teacher_logits,
                    targets,
                    alpha=alpha,
                    temperature=temperature,
                    valid_vocab_size=valid_vocab_size,
                )
                (loss / len(group)).backward()
                token_count = int(targets.numel())
                totals["loss_sum"] += float(metrics["loss"]) * token_count
                totals["ce_sum"] += float(metrics["ce"]) * token_count
                totals["kd_sum"] += float(metrics["kd"]) * token_count
                totals["token_count"] += token_count
                totals["teacher_invalid_mass_sum"] += float(
                    invalid_vocab_probability_mass(teacher_logits, valid_vocab_size).sum()
                )
                totals["student_invalid_mass_sum"] += float(
                    invalid_vocab_probability_mass(student_logits.detach(), valid_vocab_size).sum()
                )
                del teacher_logits, student_logits, loss, metrics, input_ids, targets
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            if optimizer_step % report_every == 0 or optimizer_step == optimizer_steps:
                logger.info(
                    "kd_progress budget=%s alpha=%s temperature=%s optimizer_step=%d/%d mean_loss=%.6f peak_mib=%.1f",
                    budget_name,
                    alpha,
                    temperature,
                    optimizer_step,
                    optimizer_steps,
                    totals["loss_sum"] / max(1, totals["token_count"]),
                    torch.cuda.max_memory_allocated() / 1024**2,
                )
    output_dir = Path(runtime_output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    student.save_pretrained(output_dir, safe_serialization=True)
    token_count = int(totals["token_count"])
    metrics = {
        "status": "trained",
        "budget_name": budget_name,
        "alpha": float(alpha),
        "temperature": float(temperature),
        "seed": seed,
        "records": len(rows),
        "supervised_tokens": token_count,
        "optimizer_steps": optimizer_step,
        "mean_loss": totals["loss_sum"] / token_count,
        "mean_ce": totals["ce_sum"] / token_count,
        "mean_kd": totals["kd_sum"] / token_count,
        "teacher_invalid_vocab_mass": totals["teacher_invalid_mass_sum"] / token_count,
        "student_invalid_vocab_mass": totals["student_invalid_mass_sum"] / token_count,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "elapsed_seconds": time.time() - started,
        "train_path": str(data_path),
        "train_sha256": file_sha256(data_path),
        "model_evidence": model_evidence,
    }
    write_json(output_dir / "training_metrics.json", metrics)
    return metrics


def runtime_metadata() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_unix": time.time(),
    }
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "CUDA_VISIBLE_DEVICES"):
        if key in os.environ:
            payload[key.lower()] = os.environ[key]
    try:
        import torch

        payload.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        pass
    return payload


def copy_with_retries(source: str | Path, destination: str | Path, attempts: int = 5) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            if file_sha256(source_path) != file_sha256(destination_path):
                raise OSError(f"Hash mismatch after copy: {source_path} -> {destination_path}")
            return
        except OSError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(30, attempt * 5))
    raise OSError(f"Could not publish {source_path} after {attempts} attempts") from last_error


def publish_adapter(
    runtime_dir: str | Path,
    publish_dir: str | Path,
    *,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    runtime_path = Path(runtime_dir)
    destination = resolve_project_path(publish_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite adapter destination: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    required = ["adapter_config.json", "adapter_model.safetensors", "training_metrics.json"]
    for filename in required:
        source = runtime_path / filename
        if not source.is_file():
            raise FileNotFoundError(f"Runtime adapter is missing {filename}: {runtime_path}")
        copy_with_retries(source, destination / filename)
    marker_payload = {
        **dict(evidence),
        "adapter_config_sha256": file_sha256(destination / "adapter_config.json"),
        "adapter_model_sha256": file_sha256(destination / "adapter_model.safetensors"),
        "training_metrics_sha256": file_sha256(destination / "training_metrics.json"),
        "runtime": runtime_metadata(),
    }
    write_json(destination / "train_manifest.json", marker_payload)
    marker_payload["train_manifest_sha256"] = file_sha256(destination / "train_manifest.json")
    write_json(destination / "TRAIN_COMPLETE", marker_payload)
    return marker_payload


def validated_training_marker(path: str | Path) -> Dict[str, Any] | None:
    directory = resolve_project_path(path)
    marker_path = directory / "TRAIN_COMPLETE"
    manifest_path = directory / "train_manifest.json"
    required = [directory / "adapter_config.json", directory / "adapter_model.safetensors", directory / "training_metrics.json"]
    if not marker_path.is_file() or not manifest_path.is_file() or not all(item.is_file() for item in required):
        return None
    marker = read_json(marker_path)
    manifest = read_json(manifest_path)
    checks = {
        "train_manifest_sha256": file_sha256(manifest_path),
        "adapter_config_sha256": file_sha256(required[0]),
        "adapter_model_sha256": file_sha256(required[1]),
        "training_metrics_sha256": file_sha256(required[2]),
    }
    if any(marker.get(key) != value for key, value in checks.items()):
        return None
    for key, value in manifest.items():
        if marker.get(key) != value:
            return None
    return marker


def iter_groups(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("group size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]
