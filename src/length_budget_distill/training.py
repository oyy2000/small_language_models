"""Student SFT entrypoint helpers.

This module is a thin wrapper around common open-source training packages
rather than a custom trainer.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_trl_sft(config: Dict[str, Any]) -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Install datasets, transformers, trl, peft, and accelerate before running student SFT."
        ) from exc

    data_config = config.get("data", {})
    student_config = config.get("student", {})
    train_path = data_config.get("train_path")
    if not train_path:
        raise ValueError("data.train_path is required.")
    data_files = {"train": _require_existing_project_file("data.train_path", train_path)}
    if data_config.get("eval_path"):
        data_files["eval"] = _require_existing_project_file("data.eval_path", data_config["eval_path"])

    model_name = student_config.get("model_name")
    if not model_name or model_name == "REQUIRES_USER_APPROVAL":
        raise ValueError("student.model_name must be set to a configured model name.")

    tokenizer_name = student_config.get("tokenizer_name", model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **_tokenizer_kwargs(student_config))
    _ensure_pad_token(tokenizer)
    _validate_loss_format(data_config, config.get("training", {}), tokenizer)

    dataset = load_dataset("json", data_files=data_files, keep_in_memory=True)
    text_format = data_config.get("text_format", "prompt_completion")
    dataset = _select_sft_columns(dataset, text_format)

    peft_config = None
    if student_config.get("use_lora", False):
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise ImportError("Install peft before using LoRA.") from exc
        lora = student_config.get("lora", {})
        peft_config = LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=lora.get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
        )

    training = dict(config.get("training", {}))
    training["output_dir"] = str(_resolve_project_path(training.get("output_dir", "checkpoints/student_sft")))
    args = _make_sft_config(SFTConfig, training, student_config)

    trainer_kwargs = {
        "model": model_name,
        "args": args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("eval"),
        "peft_config": peft_config,
    }
    trainer_kwargs.update(_tokenizer_trainer_kwargs(SFTTrainer, tokenizer))

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(training["output_dir"])


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    legacy_path = _resolve_legacy_code_path(path)
    if legacy_path is not None:
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path
        return legacy_path
    return PROJECT_ROOT / path


def _resolve_legacy_code_path(path: Path) -> Path | None:
    parts = path.parts
    if not parts or parts[0] != "code" or (PROJECT_ROOT / "code").exists():
        return None
    if len(parts) == 1:
        return PROJECT_ROOT
    return PROJECT_ROOT / Path(*parts[1:])


def _require_existing_project_file(label: str, path_value: str) -> str:
    path = _resolve_project_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} does not exist: {path} "
            f"(from {path_value!r}; relative paths are resolved from project root {PROJECT_ROOT})"
        )
    return str(path)


def _select_sft_columns(dataset: Any, text_format: str) -> Any:
    if text_format == "prompt_completion":
        keep_columns = {"prompt", "completion"}
    elif text_format == "messages":
        keep_columns = {"messages"}
    else:
        raise ValueError("data.text_format must be either 'prompt_completion' or 'messages'.")

    def select_columns(example: Dict[str, Any]) -> Dict[str, Any]:
        missing = keep_columns - set(example)
        if missing:
            raise ValueError(f"SFT record is missing required columns: {sorted(missing)}")
        return {key: example[key] for key in keep_columns}

    for split_name in dataset:
        remove_columns = [column for column in dataset[split_name].column_names if column not in keep_columns]
        dataset[split_name] = dataset[split_name].map(
            select_columns,
            remove_columns=remove_columns,
            keep_in_memory=True,
            load_from_cache_file=False,
        )
    return dataset


def _validate_loss_format(data_config: Dict[str, Any], training_config: Dict[str, Any], tokenizer: Any) -> None:
    text_format = data_config.get("text_format", "prompt_completion")
    assistant_only_loss = bool(training_config.get("assistant_only_loss", False))
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    if text_format == "messages" and assistant_only_loss and "{% generation %}" not in chat_template:
        raise ValueError(
            "assistant_only_loss=True requires a tokenizer chat_template with a "
            "{% generation %} block. Qwen chat templates often do not provide this "
            "assistant mask. Use data.text_format='prompt_completion' with "
            "training.completion_only_loss=true, or set assistant_only_loss=false."
        )


def _tokenizer_kwargs(student_config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(student_config.get("tokenizer_kwargs", {}))
    if "trust_remote_code" in student_config:
        kwargs.setdefault("trust_remote_code", bool(student_config["trust_remote_code"]))
    return kwargs


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token


def _model_init_kwargs(training_config: Dict[str, Any], student_config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(training_config.get("model_init_kwargs", {}))
    kwargs.setdefault("device_map", "auto")
    if "trust_remote_code" in student_config:
        kwargs.setdefault("trust_remote_code", bool(student_config["trust_remote_code"]))
    if "torch_dtype" in student_config:
        kwargs.setdefault("torch_dtype", student_config["torch_dtype"])
    return kwargs


def _make_sft_config(sft_config_cls: Any, training_config: Dict[str, Any], student_config: Dict[str, Any]) -> Any:
    parameters = inspect.signature(sft_config_cls).parameters
    kwargs: Dict[str, Any] = {
        "output_dir": training_config.get("output_dir", "checkpoints/student_sft"),
        "num_train_epochs": float(training_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training_config.get("learning_rate", 2e-5)),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.03)),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_steps": int(training_config.get("save_steps", 200)),
        "bf16": bool(training_config.get("bf16", True)),
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", True)),
        "report_to": training_config.get("report_to", "none"),
        "completion_only_loss": training_config.get("completion_only_loss", None),
        "assistant_only_loss": bool(training_config.get("assistant_only_loss", False)),
        "packing": bool(training_config.get("packing", False)),
        "model_init_kwargs": _model_init_kwargs(training_config, student_config),
        "dataset_kwargs": training_config.get(
            "dataset_kwargs",
            {
                "keep_in_memory": True,
                "load_from_cache_file": False,
            },
        ),
    }
    max_length = training_config.get("max_length", training_config.get("max_seq_length", 2048))
    if "max_length" in parameters:
        kwargs["max_length"] = max_length
    elif "max_seq_length" in parameters:
        kwargs["max_seq_length"] = max_length

    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    supported_kwargs = kwargs if accepts_arbitrary_kwargs else {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return sft_config_cls(**supported_kwargs)


def _tokenizer_trainer_kwargs(sft_trainer_cls: Any, tokenizer: Any) -> Dict[str, Any]:
    parameters = inspect.signature(sft_trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}
