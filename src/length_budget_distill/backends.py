"""Teacher generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

from .records import ProblemRecord


class TeacherBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, problem: ProblemRecord, budget: Dict[str, Any], prompt: str) -> str:
        raise NotImplementedError("TeacherBackend implementations must define generate().")


@dataclass
class LocalRuleTeacherBackend:
    model_name: str = "local-rule-teacher"
    backend_name: str = "local_rule"

    def generate(self, problem: ProblemRecord, budget: Dict[str, Any], prompt: str) -> str:
        name = budget["name"]
        answer = problem.answer
        if name in {"short", "small"}:
            return f"Compute directly. Answer: {answer}"
        if name == "medium":
            return (
                "Use the given quantities and perform the needed arithmetic. "
                f"The computation gives {answer}. Answer: {answer}"
            )
        return (
            "We identify the quantities in the problem, choose the required arithmetic operation, "
            "and compute it carefully. The resulting value satisfies the question's condition. "
            f"Therefore the final value is {answer}. Answer: {answer}"
        )


def _max_new_tokens_for_budget(generation_config: Dict[str, Any], budget: Dict[str, Any]) -> int:
    configured = generation_config.get("max_new_tokens")
    max_new_tokens = int(budget.get("generation_max_new_tokens", configured or budget["max_solution_tokens"]))
    if bool(generation_config.get("cap_max_new_tokens_by_budget", False)):
        max_new_tokens = min(max_new_tokens, int(budget["max_solution_tokens"]))
    return max_new_tokens


def _tokenizer_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(config.get("tokenizer_kwargs", {}))
    if "trust_remote_code" in config:
        kwargs.setdefault("trust_remote_code", bool(config["trust_remote_code"]))
    return kwargs


def _model_kwargs(config: Dict[str, Any], torch_module: Any) -> Dict[str, Any]:
    kwargs = dict(config.get("model_kwargs", {}))
    if "trust_remote_code" in config:
        kwargs.setdefault("trust_remote_code", bool(config["trust_remote_code"]))
    kwargs.setdefault("device_map", config.get("device_map", "auto"))

    dtype_name = config.get("torch_dtype", "auto")
    if dtype_name == "auto":
        kwargs.setdefault("torch_dtype", torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32)
    elif dtype_name is not None:
        dtype_lookup = {
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        kwargs.setdefault("torch_dtype", dtype_lookup.get(str(dtype_name), dtype_name))
    return kwargs


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token


def _format_generation_prompt(tokenizer: Any, prompt: str, config: Dict[str, Any]) -> str:
    use_chat_template = bool(config.get("use_chat_template", True))
    if not use_chat_template or not getattr(tokenizer, "chat_template", None):
        return prompt

    messages = []
    system_prompt = config.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": prompt})
    chat_template_kwargs = dict(config.get("chat_template_kwargs", {}))
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


def _hf_generate_kwargs(generation_config: Dict[str, Any], tokenizer: Any, max_new_tokens: int) -> Dict[str, Any]:
    temperature = float(generation_config.get("temperature", 0.0))
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = float(generation_config.get("top_p", 1.0))
    for key in ("top_k", "min_p", "repetition_penalty", "no_repeat_ngram_size"):
        if key in generation_config:
            kwargs[key] = generation_config[key]
    return kwargs


class HFTransformersTeacherBackend:
    backend_name = "hf_transformers"

    def __init__(self, teacher_config: Dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers before using hf_transformers.") from exc

        model_name = teacher_config.get("model_name")
        if not model_name or model_name == "REQUIRES_USER_APPROVAL":
            raise ValueError("teacher.model_name must be set to a configured model name.")

        self.model_name = model_name
        self.teacher_config = teacher_config
        self.generation_config = teacher_config.get("generation", {})
        tokenizer_name = teacher_config.get("tokenizer_name", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **_tokenizer_kwargs(teacher_config))
        _ensure_pad_token(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **_model_kwargs(teacher_config, torch),
        )
        self.model.eval()

    def generate(self, problem: ProblemRecord, budget: Dict[str, Any], prompt: str) -> str:
        max_new_tokens = _max_new_tokens_for_budget(self.generation_config, budget)
        text = _format_generation_prompt(self.tokenizer, prompt, self.teacher_config)
        inputs = self.tokenizer(text, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generate_kwargs = _hf_generate_kwargs(self.generation_config, self.tokenizer, max_new_tokens)
        output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class VLLMTeacherBackend:
    backend_name = "vllm"

    def __init__(self, teacher_config: Dict[str, Any]) -> None:
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install vllm and transformers before using the vllm teacher backend.") from exc

        model_name = teacher_config.get("model_name")
        if not model_name or model_name == "REQUIRES_USER_APPROVAL":
            raise ValueError("teacher.model_name must be set to a configured model name.")

        self.model_name = model_name
        self.teacher_config = teacher_config
        self.generation_config = teacher_config.get("generation", {})
        self.sampling_params_cls = SamplingParams
        tokenizer_name = teacher_config.get("tokenizer_name", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **_tokenizer_kwargs(teacher_config))
        llm_kwargs = dict(teacher_config.get("llm_kwargs", {}))
        if "trust_remote_code" in teacher_config:
            llm_kwargs.setdefault("trust_remote_code", bool(teacher_config["trust_remote_code"]))
        if "dtype" in teacher_config:
            llm_kwargs.setdefault("dtype", teacher_config["dtype"])
        self.llm = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            tensor_parallel_size=int(teacher_config.get("tensor_parallel_size", 1)),
            **llm_kwargs,
        )

    def generate(self, problem: ProblemRecord, budget: Dict[str, Any], prompt: str) -> str:
        text = _format_generation_prompt(self.tokenizer, prompt, self.teacher_config)
        outputs = self.llm.generate([text], self._sampling_params_for_budget(budget))
        return outputs[0].outputs[0].text.strip()

    def _sampling_params_for_budget(self, budget: Dict[str, Any]) -> Any:
        kwargs: Dict[str, Any] = {
            "temperature": float(self.generation_config.get("temperature", 0.0)),
            "top_p": float(self.generation_config.get("top_p", 1.0)),
            "max_tokens": _max_new_tokens_for_budget(self.generation_config, budget),
        }
        for key in (
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "seed",
        ):
            if key in self.generation_config:
                kwargs[key] = self.generation_config[key]
        return self.sampling_params_cls(**kwargs)


def make_teacher_backend(config: Dict[str, Any]) -> TeacherBackend:
    teacher_config = config.get("teacher", {})
    backend = teacher_config.get("backend", "vllm")

    if backend == "local_rule":
        return LocalRuleTeacherBackend(model_name=teacher_config.get("model_name", "local-rule-teacher"))

    if backend == "hf_transformers":
        return HFTransformersTeacherBackend(teacher_config)
    if backend == "vllm":
        return VLLMTeacherBackend(teacher_config)

    raise ValueError(f"Unsupported teacher backend: {backend}")
