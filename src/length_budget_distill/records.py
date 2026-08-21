"""Dataclasses and JSONL helpers for trace records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class ProblemRecord:
    problem_id: str
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    trace_id: str
    problem_id: str
    question: str
    answer: str
    budget_name: str
    max_solution_tokens: int
    teacher_backend: str
    teacher_model: str
    prompt: str
    solution: str
    predicted_answer: Optional[str]
    is_correct: bool
    solution_token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    generator_name: Optional[str] = None
    generator_size_b: Optional[float] = None
    candidate_index: int = 0
    generation_seed: Optional[int] = None
    budget_compliant: Optional[bool] = None
    selected_for_sft: bool = False
    config_hash: Optional[str] = None
    source_hash: Optional[str] = None


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def trace_from_dict(record: Dict[str, Any]) -> TraceRecord:
    return TraceRecord(**record)


def trace_to_dict(record: TraceRecord) -> Dict[str, Any]:
    return asdict(record)


def traces_from_jsonl(path: Path) -> List[TraceRecord]:
    return [trace_from_dict(item) for item in read_jsonl(path)]
