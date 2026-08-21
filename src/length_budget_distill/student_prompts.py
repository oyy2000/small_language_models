"""Shared student prompts used by SFT construction and evaluation."""

from __future__ import annotations


def build_student_math_prompt(question: str) -> str:
    return (
        f"Problem:\n{question}\n\n"
        "Solve the problem and end with a line in the form: Answer: <final answer>."
    )
