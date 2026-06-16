"""Conversion from verified teacher traces to SFT JSONL records."""

from __future__ import annotations

from typing import Any, Dict

from .records import TraceRecord


def trace_to_sft_record(trace: TraceRecord) -> Dict[str, Any]:
    assistant_text = trace.solution.strip()
    return {
        "id": trace.trace_id,
        "messages": [
            {"role": "user", "content": trace.question},
            {"role": "assistant", "content": assistant_text},
        ],
        "prompt": trace.question,
        "completion": assistant_text,
        "metadata": {
            "problem_id": trace.problem_id,
            "budget_name": trace.budget_name,
            "max_solution_tokens": trace.max_solution_tokens,
            "solution_token_count": trace.solution_token_count,
            "teacher_backend": trace.teacher_backend,
            "teacher_model": trace.teacher_model,
            "is_correct": trace.is_correct,
        },
    }

