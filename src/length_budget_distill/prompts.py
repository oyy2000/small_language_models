"""Prompt templates for length-budgeted teacher generation."""

from __future__ import annotations

from typing import Any, Dict

from .records import ProblemRecord


SYSTEM_INSTRUCTION = (
    "You are a careful math teacher. Solve the problem correctly. "
    "Write only the visible solution trace that a student should learn from. "
    "Always include the final answer."
)


def build_length_budget_prompt(problem: ProblemRecord, budget: Dict[str, Any]) -> str:
    max_tokens = int(budget["max_solution_tokens"])
    style_hint = budget.get("style_hint", "Use a concise but complete solution.")
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Length budget: solve in <= {max_tokens} solution tokens. This is a prompt-level length target.\n"
        "If the budget is tight, compress the reasoning first; do not omit the final answer.\n"
        f"Style: {style_hint}\n\n"
        f"Problem:\n{problem.question}\n\n"
        "Return exactly this format:\n"
        "Solution: <visible reasoning>\n"
        "Answer: <final answer>"
    )
