"""Token counting utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        raise NotImplementedError("TokenCounter implementations must define count().")


class WhitespaceTokenCounter:
    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class HFTokenCounter:
    def __init__(self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any] | None = None) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers before using hf_tokenizer token counting.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **(tokenizer_kwargs or {}))

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def make_token_counter(config: Dict[str, Any]) -> TokenCounter:
    token_config = config.get("token_counter", {})
    backend = token_config.get("backend", "whitespace")

    if backend == "whitespace":
        return WhitespaceTokenCounter()

    if backend == "hf_tokenizer":
        tokenizer_name = token_config.get("tokenizer_name")
        if not tokenizer_name or tokenizer_name == "REQUIRES_USER_APPROVAL":
            raise ValueError("token_counter.tokenizer_name must be set to a configured tokenizer.")
        tokenizer_kwargs = dict(token_config.get("tokenizer_kwargs", {}))
        if "trust_remote_code" in token_config:
            tokenizer_kwargs.setdefault("trust_remote_code", bool(token_config["trust_remote_code"]))
        return HFTokenCounter(tokenizer_name, tokenizer_kwargs)

    raise ValueError(f"Unsupported token counter backend: {backend}")
