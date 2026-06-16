"""Dataset loading utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .config import resolve_path
from .records import ProblemRecord, read_jsonl
from .verifiers import clean_answer, extract_final_answer


def load_problem_records(config: Dict[str, Any]) -> List[ProblemRecord]:
    dataset_config = config.get("dataset", {})
    source = dataset_config.get("source", "local_jsonl")

    if source == "local_jsonl":
        return _load_local_jsonl(config)

    if source == "hf_dataset":
        return _load_hf_dataset(dataset_config)

    raise ValueError(f"Unsupported dataset source: {source}")


def _load_local_jsonl(config: Dict[str, Any]) -> List[ProblemRecord]:
    dataset_config = config.get("dataset", {})
    path = resolve_path(dataset_config.get("path"), config)
    if path is None:
        raise ValueError("Local dataset config must define path.")
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {path}")

    question_field = dataset_config.get("question_field", "question")
    answer_field = dataset_config.get("answer_field", "answer")
    max_examples = dataset_config.get("max_examples")

    records: List[ProblemRecord] = []
    for index, item in enumerate(read_jsonl(path)):
        if question_field not in item or answer_field not in item:
            raise ValueError(
                f"Dataset row {index} must contain fields {question_field!r} and {answer_field!r}."
            )
        problem_id = str(item.get("id", f"local-{index:06d}"))
        answer = _extract_answer(item[answer_field], dataset_config)
        metadata = {key: value for key, value in item.items() if key not in {"id", question_field, answer_field}}
        if str(answer) != str(item[answer_field]):
            metadata["raw_answer"] = item[answer_field]
        records.append(
            ProblemRecord(
                problem_id=problem_id,
                question=str(item[question_field]),
                answer=str(answer),
                metadata=metadata,
            )
        )
        if max_examples is not None and len(records) >= int(max_examples):
            break
    return records


def _load_hf_dataset(dataset_config: Dict[str, Any]) -> List[ProblemRecord]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets before loading Hugging Face datasets.") from exc

    dataset_name = dataset_config.get("dataset_name")
    if not dataset_name or dataset_name == "REQUIRES_USER_APPROVAL":
        raise ValueError("dataset.dataset_name must be set to a configured dataset name.")

    dataset_config_name = dataset_config.get("dataset_config")
    split = dataset_config.get("split", "train")
    question_field = dataset_config.get("question_field", "question")
    answer_field = dataset_config.get("answer_field", "answer")
    max_examples = dataset_config.get("max_examples")

    loaded = load_dataset(dataset_name, dataset_config_name, split=split)
    if max_examples is not None:
        loaded = loaded.select(range(min(int(max_examples), len(loaded))))

    records: List[ProblemRecord] = []
    for index, item in enumerate(loaded):
        if question_field not in item or answer_field not in item:
            raise ValueError(
                f"HF dataset row {index} must contain fields {question_field!r} and {answer_field!r}."
            )
        problem_id = str(item.get("id", f"hf-{index:06d}"))
        answer = _extract_answer(item[answer_field], dataset_config)
        metadata = {key: value for key, value in item.items() if key not in {"id", question_field, answer_field}}
        if str(answer) != str(item[answer_field]):
            metadata["raw_answer"] = item[answer_field]
        records.append(
            ProblemRecord(
                problem_id=problem_id,
                question=str(item[question_field]),
                answer=str(answer),
                metadata=metadata,
            )
        )
    return records


def _extract_answer(raw_answer: Any, dataset_config: Dict[str, Any]) -> str:
    answer_text = str(raw_answer)
    answer_format = dataset_config.get("answer_format")
    if answer_format in {"gsm8k", "gsm8k_final"}:
        extracted = extract_final_answer(answer_text)
        if extracted is None:
            raise ValueError(f"Could not extract GSM8K-style final answer from: {answer_text!r}")
        return extracted

    answer_regex = dataset_config.get("answer_regex")
    if answer_regex:
        match = re.search(str(answer_regex), answer_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract answer with answer_regex={answer_regex!r}")
        return clean_answer(match.group(1))

    return answer_text
