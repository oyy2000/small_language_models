"""Simple final-answer extraction and verification."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
GSM8K_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
VERIFIER_VERSION = "gsm8k_answer_segment_first_numeric_v2"
MATH_VERIFIER_VERSION = "math_verify_0.9.0_latex_expr_v1"
SUPPORTED_VERIFIERS = {"gsm8k_numeric", "math_verify"}


def extract_final_answer(text: str) -> Optional[str]:
    for pattern in (GSM8K_RE, BOXED_RE, ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            return clean_answer(matches[-1])
    return None


def extract_last_boxed(text: str) -> Optional[str]:
    """Return the content of the last brace-balanced LaTeX boxed expression."""

    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    content_start = start + len(r"\boxed{")
    depth = 1
    for index in range(content_start, len(text)):
        character = text[index]
        if character == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif character == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def extract_math_final_answer(text: str) -> Optional[str]:
    """Extract a MATH-style final answer while preserving nested LaTeX."""

    answer_matches = ANSWER_RE.findall(text)
    if answer_matches:
        answer = clean_answer(answer_matches[-1])
        boxed = extract_last_boxed(answer)
        return clean_answer(boxed if boxed is not None else answer)
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return clean_answer(boxed)
    return extract_final_answer(text)


def clean_answer(answer: str) -> str:
    cleaned = answer.strip()
    cleaned = cleaned.strip("$")
    cleaned = cleaned.strip()
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.rstrip(",")
    return cleaned.strip()


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    normalized = clean_answer(answer)
    normalized = normalized.replace(",", "")
    numeric_match = NUMBER_RE.search(clean_answer(answer))
    if numeric_match:
        numeric_text = numeric_match.group(0).replace(",", "")
        try:
            numeric_value = Decimal(numeric_text)
        except InvalidOperation:
            pass
        else:
            if numeric_value == numeric_value.to_integral_value():
                return str(int(numeric_value))
            return format(numeric_value.normalize(), "f")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.lower()


def verify_answer(predicted: Optional[str], gold: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)


def verifier_name(config: dict) -> str:
    name = str(config.get("dataset", {}).get("verifier", "gsm8k_numeric"))
    if name not in SUPPORTED_VERIFIERS:
        raise ValueError(f"Unsupported verifier: {name!r}; expected one of {sorted(SUPPORTED_VERIFIERS)}")
    return name


def verifier_version(name: str) -> str:
    if name == "gsm8k_numeric":
        return VERIFIER_VERSION
    if name == "math_verify":
        return MATH_VERIFIER_VERSION
    raise ValueError(f"Unsupported verifier: {name!r}")


def extract_answer_for_verifier(text: str, name: str) -> Optional[str]:
    if name == "gsm8k_numeric":
        return extract_final_answer(text)
    if name == "math_verify":
        return extract_math_final_answer(text)
    raise ValueError(f"Unsupported verifier: {name!r}")


def verify_answer_for_verifier(predicted: Optional[str], gold: str, name: str) -> bool:
    if name == "gsm8k_numeric":
        return verify_answer(predicted, gold)
    if name == "math_verify":
        return verify_math_answer(predicted, gold)
    raise ValueError(f"Unsupported verifier: {name!r}")


def verify_math_answer(predicted: Optional[str], gold: str) -> bool:
    """Compare symbolic/numeric MATH answers with the pinned Math-Verify backend."""

    if predicted is None:
        return False
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as exc:
        raise ImportError(
            "Install math-verify[antlr4_13_2]==0.9.0 before MATH generation or evaluation."
        ) from exc

    gold_parsed = parse(
        _as_math_environment(gold),
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
        fallback_mode="no_fallback",
    )
    predicted_parsed = parse(
        _as_math_environment(predicted),
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
        fallback_mode="no_fallback",
    )
    return bool(gold_parsed and predicted_parsed and verify(gold_parsed, predicted_parsed))


def _as_math_environment(answer: str) -> str:
    stripped = clean_answer(answer)
    if any(marker in stripped for marker in ("$", r"\(", r"\[", r"\boxed{")):
        return stripped
    return f"${stripped}$"
