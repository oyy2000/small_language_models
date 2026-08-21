"""Conversion from verified teacher traces to SFT JSONL records."""

from __future__ import annotations

from typing import Any, Dict

from .records import TraceRecord
from .student_prompts import build_student_math_prompt


def trace_to_sft_record(trace: TraceRecord) -> Dict[str, Any]:
    assistant_text = trace.solution.strip()
    student_prompt = build_student_math_prompt(trace.question)
    prompt_strategy = trace.metadata.get("prompt_strategy")
    problem_metadata = trace.metadata.get("problem_metadata", {})
    return {
        "id": trace.trace_id,
        "messages": [
            {"role": "user", "content": student_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "prompt": student_prompt,
        "teacher_prompt": trace.prompt,
        "completion": assistant_text,
        "metadata": {
            "problem_id": trace.problem_id,
            "budget_name": trace.budget_name,
            "max_solution_tokens": trace.max_solution_tokens,
            "solution_token_count": trace.solution_token_count,
            "teacher_backend": trace.teacher_backend,
            "teacher_model": trace.teacher_model,
            "generator_name": trace.generator_name,
            "generator_size_b": trace.generator_size_b,
            "candidate_index": trace.candidate_index,
            "generation_seed": trace.generation_seed,
            "budget_compliant": trace.budget_compliant,
            "selected_for_sft": trace.selected_for_sft,
            "is_correct": trace.is_correct,
            "prompt_strategy": prompt_strategy,
            "problem_metadata": problem_metadata,
        },
    }
