"""Paired, structure-preserving compression of verified math rationales."""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .verifiers import extract_final_answer, normalize_answer, verify_answer


ANSWER_LINE_RE = re.compile(r"(?im)^\s*(?:answer|final answer)\s*[:=]")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
SIMPLE_FRACTION_RE = re.compile(r"\\(?:d)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
PERCENT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")


@dataclass(frozen=True)
class EquationCheck:
    text: str
    left: str
    right: str
    left_value: str
    right_value: str
    valid: bool


def stable_rewrite_seed(base_seed: int, source_id: str, ratio_name: str) -> int:
    """Return a deterministic generation seed for one source/ratio request."""

    digest = hashlib.sha256(f"{base_seed}|{source_id}|{ratio_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def target_token_count(source_tokens: int, ratio: float) -> int:
    """Compute a non-empty target that is always shorter than the source."""

    if source_tokens <= 1:
        raise ValueError("source_tokens must be greater than one")
    if not 0.0 < ratio < 1.0:
        raise ValueError("rewrite ratio must be strictly between zero and one")
    return max(1, min(source_tokens - 1, int(round(source_tokens * ratio))))


def minimum_target_token_count(target_tokens: int, minimum_fraction: float) -> int:
    """Return the minimum accepted length for an adaptive rewrite band."""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum target fraction must be in (0, 1]")
    return max(1, int(math.ceil(target_tokens * minimum_fraction)))


def source_problem_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    problem_id = metadata.get("problem_id") if isinstance(metadata, Mapping) else None
    if problem_id is None:
        problem_id = row.get("problem_id") or row.get("id")
    if not problem_id:
        raise ValueError("SFT source row is missing a problem ID")
    return str(problem_id)


def source_token_count(row: Mapping[str, Any], token_counter: Any) -> int:
    metadata = row.get("metadata", {})
    stored = metadata.get("solution_token_count") if isinstance(metadata, Mapping) else None
    completion = str(row.get("completion", "")).strip()
    if not completion:
        raise ValueError(f"Source row has an empty completion: {source_problem_id(row)}")
    measured = int(token_counter.count(completion))
    if stored is not None and int(stored) != measured:
        raise ValueError(
            f"Stored/measured source token mismatch for {source_problem_id(row)}: "
            f"stored={stored} measured={measured}"
        )
    return measured


def extract_equation_checks(text: str) -> List[EquationCheck]:
    """Check adjacent fully numeric expressions joined by equality signs.

    Non-numeric statements and symbolic equations are deliberately skipped. The
    checker is a conservative data-quality gate, not a general algebra system.
    """

    checks: List[EquationCheck] = []
    for raw_segment in _equation_segments(text):
        if "=" not in raw_segment or any(marker in raw_segment for marker in ("<=", ">=", "≠")):
            continue
        pieces = raw_segment.split("=")
        parsed = [_parse_numeric_expression(piece) for piece in pieces]
        for index in range(len(parsed) - 1):
            left = parsed[index]
            right = parsed[index + 1]
            if left is None or right is None:
                continue
            left_text, left_value = left
            right_text, right_value = right
            if not (_has_arithmetic_operator(left_text) or _has_arithmetic_operator(right_text)):
                continue
            if _is_semantic_left_quantity(pieces[index], left_text) and _has_arithmetic_operator(right_text):
                # A prose quantity or unit definition on the left is not a
                # numeric equality (for example, "half of 50 = 0.5 * 50").
                continue
            if _is_percent_scalar(pieces[index + 1]):
                # A bare percentage is a labelled quantity. Treating 25% as
                # 0.25 would incorrectly reject "... * 100 = 25%".
                continue
            checks.append(
                EquationCheck(
                    text=raw_segment.strip(),
                    left=left_text,
                    right=right_text,
                    left_value=_decimal_text(left_value),
                    right_value=_decimal_text(right_value),
                    valid=math.isclose(float(left_value), float(right_value), rel_tol=1e-9, abs_tol=1e-9),
                )
            )
    return checks


def _equation_segments(text: str) -> List[str]:
    segments: List[str] = []
    for sentence in re.split(r"[\n;]|\.\s+(?=[A-Z])", text):
        inline_math = re.findall(r"\\\((.*?)\\\)", sentence)
        display_math = re.findall(r"\\\[(.*?)\\\]", sentence)
        math_spans = [*inline_math, *display_math]
        segments.extend(math_spans if math_spans else [sentence])
    return segments


def essential_step_values(source_text: str, gold_answer: str) -> List[str]:
    """Extract equation results reused later in the verified source rationale."""

    checks = extract_equation_checks(source_text)
    normalized_source = _normalized_value_inventory(source_text, checks)
    gold_normalized = normalize_answer(gold_answer)
    required: List[str] = []
    for check in checks:
        if not check.valid:
            continue
        value = check.right_value
        if normalize_answer(value) == gold_normalized:
            continue
        if normalized_source.count(value) >= 2 and value not in required:
            required.append(value)
    return required


def assess_rewrite_candidate(
    solution: str,
    *,
    gold_answer: str,
    source_tokens: int,
    minimum_tokens: int,
    target_tokens: int,
    required_step_values: Sequence[str],
    token_counter: Any,
    candidate_index: int,
) -> Dict[str, Any]:
    """Return deterministic quality evidence for one rewrite candidate."""

    text = solution.strip()
    actual_tokens = int(token_counter.count(text)) if text else 0
    predicted_answer = extract_final_answer(text)
    checks = extract_equation_checks(text)
    invalid_checks = [check for check in checks if not check.valid]
    inventory = set(_normalized_value_inventory(text, checks))
    matched = [value for value in required_step_values if value in inventory]
    coverage = len(matched) / len(required_step_values) if required_step_values else 1.0
    answer_line_present = bool(ANSWER_LINE_RE.search(text))
    answer_valid = verify_answer(predicted_answer, gold_answer)
    equation_valid = not invalid_checks
    no_longer_than_source = 0 < actual_tokens <= source_tokens
    in_length_band = minimum_tokens <= actual_tokens <= target_tokens
    structurally_valid = (
        answer_line_present
        and answer_valid
        and equation_valid
        and coverage == 1.0
        and no_longer_than_source
    )
    within_target = structurally_valid and in_length_band
    return {
        "candidate_index": int(candidate_index),
        "solution": text,
        "predicted_answer": predicted_answer,
        "answer_line_present": answer_line_present,
        "answer_valid": answer_valid,
        "actual_tokens": actual_tokens,
        "source_tokens": int(source_tokens),
        "minimum_tokens": int(minimum_tokens),
        "target_tokens": int(target_tokens),
        "in_length_band": in_length_band,
        "within_target": within_target,
        "no_longer_than_source": no_longer_than_source,
        "equation_checked_count": len(checks),
        "equation_errors": [check.__dict__ for check in invalid_checks],
        "equation_valid": equation_valid,
        "required_step_values": list(required_step_values),
        "matched_step_values": matched,
        "step_coverage": coverage,
        "structurally_valid": structurally_valid,
    }


def select_rewrite_candidate(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    """Select a valid candidate closest to, but not beyond, its target length."""

    eligible = [candidate for candidate in candidates if bool(candidate.get("within_target"))]
    if not eligible:
        return None
    selected = min(
        eligible,
        key=lambda candidate: (
            abs(int(candidate["target_tokens"]) - int(candidate["actual_tokens"])),
            -int(candidate.get("equation_checked_count", 0)),
            int(candidate["candidate_index"]),
        ),
    )
    return dict(selected)


def select_adaptive_rewrite(
    by_ratio: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    preferred_ratio: str,
    fallback_ratios: Sequence[str],
) -> Dict[str, Any] | None:
    """Select at the requested ratio and then follow an explicit fallback order."""

    ordered = [preferred_ratio, *fallback_ratios]
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"Duplicate ratios in adaptive selection order: {ordered}")
    for fallback_level, ratio_name in enumerate(ordered):
        selected = select_rewrite_candidate(by_ratio.get(ratio_name, []))
        if selected is not None:
            selected["requested_ratio_name"] = preferred_ratio
            selected["selected_ratio_name"] = ratio_name
            selected["fallback_level"] = fallback_level
            return selected
    return None


def build_rewrite_prompt(
    *,
    question: str,
    gold_answer: str,
    source_solution: str,
    source_tokens: int,
    minimum_tokens: int,
    target_tokens: int,
    ratio_name: str,
    required_step_values: Sequence[str],
) -> str:
    """Build the paired-rewrite instruction for one source rationale."""

    required = ", ".join(required_step_values) if required_step_values else "none identified"
    return (
        "Rewrite the supplied verified solution; do not solve the problem from scratch.\n"
        "Preserve the original causal order, every necessary arithmetic operation, intermediate "
        "quantity, unit, percentage, and condition. Remove only repetition and explanatory filler.\n"
        "Every equation you write must be numerically correct. Correctness and complete logical "
        "coverage take priority over compression. End with exactly one line: Answer: <final answer>.\n\n"
        f"Rewrite condition: {ratio_name}\n"
        f"Original solution tokens: {source_tokens}\n"
        f"Target token range: {minimum_tokens} to {target_tokens}\n"
        f"Required reused intermediate values: {required}\n\n"
        f"Problem:\n{question.strip()}\n\n"
        f"Gold final answer:\n{gold_answer.strip()}\n\n"
        f"Verified original solution:\n{source_solution.strip()}\n\n"
        "Return exactly:\n"
        "Solution: <concise but complete reasoning>\n"
        "Answer: <final answer>"
    )


def paired_sft_record(
    source_row: Mapping[str, Any],
    *,
    condition: str,
    completion: str,
    source_sha256: str,
    selection: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Create one SFT record while preserving the source prompt and problem support."""

    problem_id = source_problem_id(source_row)
    metadata = dict(source_row.get("metadata", {}))
    metadata.update(
        {
            "problem_id": problem_id,
            "paired_rewrite_condition": condition,
            "source_trace_id": str(source_row.get("id", problem_id)),
            "source_sha256": source_sha256,
            "rewrite_selection": dict(selection) if selection is not None else None,
        }
    )
    return {
        "id": f"paired::{problem_id}::{condition}",
        "prompt": str(source_row["prompt"]),
        "completion": completion.strip(),
        "messages": [
            {"role": "user", "content": str(source_row["prompt"])},
            {"role": "assistant", "content": completion.strip()},
        ],
        "metadata": metadata,
    }


def _parse_numeric_expression(text: str) -> Tuple[str, Decimal] | None:
    expression_region = _expression_region(text)
    if expression_region is None:
        return None
    # Do not turn algebra into a different numeric expression by stripping its
    # variables below. Skipping a symbolic equation is safer than claiming it
    # is numerically inconsistent.
    if re.search(
        r"(?:\b[A-Za-z]\b|(?<=\d)[A-Za-z]\b|\b[A-Za-z](?=\d))",
        expression_region,
    ):
        return None
    stripped = expression_region.strip()
    explicit_math = any(marker in stripped for marker in (r"\(", r"\[", "$"))
    expression_leading = re.match(
        r"^(?:[({\[]\s*)*[-+]?\s*(?:\d|\\(?:d)?frac)",
        stripped,
    )
    if not explicit_math and expression_leading is None:
        return None
    normalized = _normalize_expression_text(expression_region)
    if not normalized or not re.search(r"\d", normalized):
        return None
    if re.search(r"[A-Za-z]", normalized):
        return None
    normalized = normalized.strip(" .,:;[]{}")
    if not normalized or not re.fullmatch(r"[\d\s.+\-*/()]+", normalized):
        return None
    try:
        tree = ast.parse(normalized, mode="eval")
        value = _eval_numeric_ast(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None
    if not value.is_finite():
        return None
    return normalized, value


def _expression_region(text: str) -> str | None:
    stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text).strip()
    math_starts = [stripped.rfind(marker) for marker in (r"\(", r"\[")]
    if max(math_starts) >= 0:
        return stripped[max(math_starts) :]
    if ":" in stripped:
        after_colon = stripped.rsplit(":", 1)[1].strip()
        if re.search(r"\d", after_colon):
            return after_colon
    if re.match(
        r"(?i)^(?:first\s+|next\s+|then\s+)?"
        r"(?:compute|calculate|add|subtract|multiply|divide)\b",
        stripped,
    ):
        match = re.search(r"(?:\d|\\(?:d)?frac)", stripped)
        return stripped[match.start() :] if match else None
    if re.match(r"(?i)^(?:then|thus|therefore)\s+(?=\d)", stripped):
        match = re.search(r"\d", stripped)
        return stripped[match.start() :] if match else None
    if re.match(r"^(?:[({\[]\s*)*[-+]?\s*(?:\d|\\(?:d)?frac)", stripped):
        return stripped
    return None


def _normalize_expression_text(text: str) -> str:
    # Remove Markdown/list punctuation before natural-language words are
    # stripped; otherwise a bullet can become an unintended unary minus.
    normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    math_starts = [normalized.rfind(marker) for marker in (r"\(", r"\[")]
    if max(math_starts) >= 0:
        normalized = normalized[max(math_starts) :]
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = SIMPLE_FRACTION_RE.sub(r"((\1)/(\2))", normalized)
    normalized = PERCENT_RE.sub(r"((\1)/100)", normalized)
    replacements = {
        "\\times": "*",
        "\\cdot": "*",
        "\\div": "/",
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "^": "**",
        ",": "",
        "$": "",
        "\\[": "",
        "\\]": "",
        "\\(": "",
        "\\)": "",
        "\\%": "/100",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\\(?:text|mathrm)\s*\{[^{}]*\}", "", normalized)
    normalized = re.sub(r"\\[A-Za-z]+", "", normalized)
    normalized = re.sub(r"[A-Za-z]+", "", normalized)
    normalized = re.sub(r"[^\d\s.+\-*/()]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _has_arithmetic_operator(expression: str) -> bool:
    return bool(re.search(r"(?:[+*/]|(?<=\d)\s*-\s*(?=\d))", expression))


def _is_semantic_left_quantity(raw_text: str, expression: str) -> bool:
    decorated_scalar = not _has_arithmetic_operator(expression) and bool(
        re.search(r"[A-Za-z]", raw_text)
    )
    semantic_of = bool(re.search(r"(?i)(?:\\text\s*\{[^{}]*\bof\b|\bof\b)", raw_text))
    return decorated_scalar or semantic_of


def _is_percent_scalar(raw_text: str) -> bool:
    stripped = raw_text
    for marker in (r"\[", r"\]", r"\(", r"\)", "$", " "):
        stripped = stripped.replace(marker, "")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?\\?%[.,]?", stripped))


def _eval_numeric_ast(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_numeric_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_numeric_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_ast(node.left)
        right = _eval_numeric_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"Unsupported numeric expression node: {type(node).__name__}")


def _normalized_value_inventory(text: str, checks: Iterable[EquationCheck]) -> List[str]:
    values: List[str] = []
    for number in NUMBER_RE.findall(text):
        normalized = _normalize_decimal_token(number)
        if normalized is not None:
            values.append(normalized)
    for check in checks:
        values.extend((check.left_value, check.right_value))
    return values


def _normalize_decimal_token(value: str) -> str | None:
    try:
        return _decimal_text(Decimal(value.replace(",", "")))
    except InvalidOperation:
        return None


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f")
