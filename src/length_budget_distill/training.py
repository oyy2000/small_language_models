"""Student SFT entrypoint helpers.

This module is a thin wrapper around common open-source training packages
rather than a custom trainer.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_trl_sft(config: Dict[str, Any]) -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
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

    resume_adapter_path = student_config.get("resume_adapter_path") or config.get("training", {}).get(
        "resume_adapter_path"
    )
    peft_config = None
    model: Any = model_name
    if resume_adapter_path:
        if not student_config.get("use_lora", False):
            raise ValueError("resume_adapter_path is only supported when student.use_lora=true.")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Install peft before resuming from a LoRA adapter.") from exc
        adapter_path = _require_complete_adapter_dir("student.resume_adapter_path", resume_adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **_model_load_kwargs(config.get("training", {}), student_config),
        )
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        _prepare_resumed_adapter_for_training(model, config.get("training", {}))
    elif student_config.get("use_lora", False):
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
    configured_output_dir = _resolve_project_path(training.get("output_dir", "checkpoints/student_sft"))
    runtime_output_override = os.environ.get("LBD_RUNTIME_OUTPUT_DIR")
    training["output_dir"] = str(
        Path(runtime_output_override).resolve() if runtime_output_override else configured_output_dir
    )
    logging.info(
        "configured_output_dir=%s runtime_output_dir=%s",
        configured_output_dir,
        training["output_dir"],
    )
    args = _make_sft_config(
        SFTConfig,
        training,
        student_config,
        include_model_init_kwargs=not bool(resume_adapter_path),
    )
    data_collator = _make_completion_only_collator(SFTConfig, training, tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset.get("eval"),
        "peft_config": peft_config,
    }
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator
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


def _require_complete_adapter_dir(label: str, path_value: str) -> str:
    path = _resolve_project_path(path_value)
    missing = [
        filename
        for filename in ("adapter_config.json", "adapter_model.safetensors")
        if not (path / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{label} is not a complete LoRA adapter directory: {path}; "
            f"missing={missing} (from {path_value!r})"
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


def _model_load_kwargs(training_config: Dict[str, Any], student_config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = _model_init_kwargs(training_config, student_config)
    dtype = kwargs.get("torch_dtype")
    if isinstance(dtype, str) and dtype != "auto":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install torch before loading a model for adapter resume.") from exc
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in dtype_by_name:
            raise ValueError(f"Unsupported torch_dtype for model loading: {dtype!r}")
        kwargs["torch_dtype"] = dtype_by_name[dtype]
    return kwargs


def _prepare_resumed_adapter_for_training(model: Any, training_config: Dict[str, Any]) -> None:
    if hasattr(model, "train"):
        model.train()
    if _count_trainable_parameters(model) <= 0:
        raise ValueError("Resumed LoRA adapter has no trainable parameters; check adapter loading.")
    if not bool(training_config.get("gradient_checkpointing", True)):
        return

    enable_input_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_grads):
        enable_input_grads()
        return

    get_embeddings = getattr(model, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    if embeddings is None or not hasattr(embeddings, "register_forward_hook"):
        return

    def make_inputs_require_grad(_module: Any, _inputs: Any, output: Any) -> None:
        if hasattr(output, "requires_grad_"):
            output.requires_grad_(True)

    embeddings.register_forward_hook(make_inputs_require_grad)


def _count_trainable_parameters(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(
        int(parameter.numel())
        for parameter in parameters()
        if bool(getattr(parameter, "requires_grad", False))
    )


def _make_sft_config(
    sft_config_cls: Any,
    training_config: Dict[str, Any],
    student_config: Dict[str, Any],
    include_model_init_kwargs: bool = True,
) -> Any:
    parameters = inspect.signature(sft_config_cls).parameters
    if bool(training_config.get("assistant_only_loss", False)) and "assistant_only_loss" not in parameters:
        raise ValueError(
            "assistant_only_loss was requested, but the installed TRL SFTConfig does not support it."
        )
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
        "dataset_kwargs": training_config.get(
            "dataset_kwargs",
            {
                "keep_in_memory": True,
                "load_from_cache_file": False,
            },
        ),
    }
    if include_model_init_kwargs:
        kwargs["model_init_kwargs"] = _model_init_kwargs(training_config, student_config)
    max_length = training_config.get("max_length", training_config.get("max_seq_length", 2048))
    if "max_length" in parameters:
        kwargs["max_length"] = max_length
    elif "max_seq_length" in parameters:
        kwargs["max_seq_length"] = max_length
    if "max_steps" in training_config:
        kwargs["max_steps"] = int(training_config["max_steps"])
    if "seed" in training_config:
        kwargs["seed"] = int(training_config["seed"])
    if "data_seed" in training_config:
        kwargs["data_seed"] = int(training_config["data_seed"])

    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    supported_kwargs = kwargs if accepts_arbitrary_kwargs else {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return sft_config_cls(**supported_kwargs)


def _make_completion_only_collator(
    sft_config_cls: Any,
    training_config: Dict[str, Any],
    tokenizer: Any,
) -> Any | None:
    if not bool(training_config.get("completion_only_loss", False)):
        logging.info("completion_only_loss_backend=disabled")
        return None
    parameters = inspect.signature(sft_config_cls).parameters
    if "completion_only_loss" in parameters:
        logging.info("completion_only_loss_backend=native_sft_config")
        return None
    if bool(training_config.get("packing", False)):
        raise ValueError("Legacy TRL completion-only masking is incompatible with packing=True.")
    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError as exc:
        raise ImportError(
            "The installed TRL lacks native completion_only_loss and DataCollatorForCompletionOnlyLM."
        ) from exc
    response_template = str(
        training_config.get("completion_response_template", "<|im_start|>assistant\n")
    )
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    if not response_token_ids:
        raise ValueError("completion_response_template tokenized to an empty sequence.")
    probe = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "completion-mask-user-probe"},
            {"role": "assistant", "content": "completion-mask-assistant-probe"},
        ],
        tokenize=True,
    )
    if not any(
        probe[index : index + len(response_token_ids)] == response_token_ids
        for index in range(len(probe) - len(response_token_ids) + 1)
    ):
        raise ValueError(
            "completion_response_template is absent from the tokenizer chat template; "
            "refusing to train without a verified completion mask."
        )
    logging.info(
        "completion_only_loss_backend=trl_data_collator response_template=%r response_token_ids=%s",
        response_template,
        response_token_ids,
    )
    return DataCollatorForCompletionOnlyLM(
        response_template=response_token_ids,
        tokenizer=tokenizer,
        mlm=False,
    )


def _tokenizer_trainer_kwargs(sft_trainer_cls: Any, tokenizer: Any) -> Dict[str, Any]:
    parameters = inspect.signature(sft_trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}
