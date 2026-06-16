"""Simple final-answer extraction and verification."""

from __future__ import annotations

import re
from typing import Optional


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
GSM8K_RE = re.compile(r"####\s*([^\n]+)")


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
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.lower()


def verify_answer(predicted: Optional[str], gold: str) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)

