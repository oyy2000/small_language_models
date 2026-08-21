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


def extract_final_answer(text: str) -> Optional[str]:
    for pattern in (GSM8K_RE, BOXED_RE, ANSWER_RE):
        matches = pattern.findall(text)
        if matches:
            return clean_answer(matches[-1])
    return None


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
