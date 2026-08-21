"""Prompt templates for teacher generation."""

from __future__ import annotations

import json
from typing import Any, Dict

from .config import resolve_path
from .records import ProblemRecord


SYSTEM_INSTRUCTION = (
    "You are a careful math teacher. Solve the problem correctly. "
    "Always include the final answer."
)


PROMPT_STRATEGY_ALIASES = {
    "length_budget": "length_budget",
    "budget": "length_budget",
    "standard": "chain_of_thought",
    "baseline": "chain_of_thought",
    "plain": "chain_of_thought",
    "cot": "chain_of_thought",
    "chain_of_thought": "chain_of_thought",
    "chain-of-thought": "chain_of_thought",
    "cod": "chain_of_draft",
    "chain_of_draft": "chain_of_draft",
    "chain-of-draft": "chain_of_draft",
}


GSM8K_COT_SYSTEM_PROMPT = (
    "Think step by step to answer the following question.\n"
    "Return the answer at the end of the response after a separator ####."
)


GSM8K_COD_SYSTEM_PROMPT = (
    "Think step by step, but only keep minimum draft for each thinking step, "
    "with 5 words at most.\n"
    "Return the answer at the end of the response after a separator ####."
)


GSM8K_COT_FEWSHOT = [
    {
        "question": (
            "There are 15 trees in the grove. Grove workers will plant trees in the\n"
            "grove today. After they are done, there will be 21 trees. How many trees did\n"
            "the grove workers plant today?"
        ),
        "answer": (
            "There are 15 trees originally. Then there were 21 trees after some more\n"
            "were planted. So there must have been 21 - 15 = 6. #### 6"
        ),
    },
    {
        "question": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many\n"
            "cars are in the parking lot?"
        ),
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "question": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many\n"
            "pieces do they have left in total?"
        ),
        "answer": (
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they\n"
            "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"
        ),
    },
    {
        "question": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12\n"
            "lollipops. How many lollipops did Jason give to Denny?"
        ),
        "answer": (
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny.\n"
            "So he gave Denny 20 - 12 = 8. #### 8"
        ),
    },
    {
        "question": (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and\n"
            "dad. How many toys does he have now?"
        ),
        "answer": (
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad,\n"
            "then that is 4 more toys. 5 + 4 = 9. #### 9"
        ),
    },
    {
        "question": (
            "There were nine computers in the server room. Five more computers were\n"
            "installed each day, from monday to thursday. How many computers are now in the\n"
            "server room?"
        ),
        "answer": (
            "There were originally 9 computers. For each of 4 days, 5 more computers\n"
            "were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29"
        ),
    },
    {
        "question": (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday,\n"
            "he lost 2 more. How many golf balls did he have at the end of wednesday?"
        ),
        "answer": (
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had\n"
            "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"
        ),
    },
    {
        "question": (
            "Olivia has $23. She bought five bagels for $3 each. How much money does\n"
            "she have left"
        ),
        "answer": (
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15\n"
            "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8"
        ),
    },
]


GSM8K_COD_FEWSHOT = [
    {
        "question": GSM8K_COT_FEWSHOT[0]["question"],
        "answer": "21 - 15 = 6. #### 6",
    },
    {
        "question": GSM8K_COT_FEWSHOT[1]["question"],
        "answer": "3 + 2 = 5. #### 5",
    },
    {
        "question": GSM8K_COT_FEWSHOT[2]["question"],
        "answer": "32 + 42 = 74; 74 - 35 = 39. #### 39",
    },
    {
        "question": GSM8K_COT_FEWSHOT[3]["question"],
        "answer": "20 - x = 12; x = 20 - 12 = 8. #### 8",
    },
    {
        "question": GSM8K_COT_FEWSHOT[4]["question"],
        "answer": "2 * 2 = 4; 5 + 4 = 9. #### 9",
    },
    {
        "question": GSM8K_COT_FEWSHOT[5]["question"],
        "answer": "5 * 4 = 20; 9 + 20 = 29. #### 29",
    },
    {
        "question": GSM8K_COT_FEWSHOT[6]["question"],
        "answer": "58 - 23 = 35; 35 - 2 = 33. #### 33",
    },
    {
        "question": GSM8K_COT_FEWSHOT[7]["question"],
        "answer": "5 * 3 = 15; 23 - 15 = 8. #### 8",
    },
]


def get_prompt_strategy(config: Dict[str, Any]) -> str:
    prompt_config = config.get("prompt", {})
    strategy = prompt_config.get("strategy", config.get("prompt_strategy", "length_budget"))
    normalized = PROMPT_STRATEGY_ALIASES.get(str(strategy).strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(PROMPT_STRATEGY_ALIASES))
        raise ValueError(f"Unsupported prompt strategy: {strategy}. Expected one of: {valid}")
    return normalized


def build_teacher_prompt(problem: ProblemRecord, budget: Dict[str, Any], config: Dict[str, Any]) -> str:
    strategy = get_prompt_strategy(config)
    if strategy == "length_budget":
        return build_length_budget_prompt(problem, budget)
    if strategy == "chain_of_thought":
        return build_standard_prompt(problem, config.get("prompt", {}), config)
    if strategy == "chain_of_draft":
        return build_chain_of_draft_prompt(problem, budget, config.get("prompt", {}), config)
    raise AssertionError(f"Unhandled prompt strategy: {strategy}")


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


def build_standard_prompt(
    problem: ProblemRecord,
    prompt_config: Dict[str, Any] | None = None,
    config: Dict[str, Any] | None = None,
) -> str:
    template = _resolve_fewshot_template(
        prompt_config=prompt_config or {},
        config=config,
        default_system_prompt=GSM8K_COT_SYSTEM_PROMPT,
        default_fewshot=GSM8K_COT_FEWSHOT,
    )
    return build_fewshot_prompt(problem=problem, **template)


def build_chain_of_draft_prompt(
    problem: ProblemRecord,
    _budget: Dict[str, Any],
    prompt_config: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> str:
    template = _resolve_fewshot_template(
        prompt_config=prompt_config,
        config=config,
        default_system_prompt=GSM8K_COD_SYSTEM_PROMPT,
        default_fewshot=GSM8K_COD_FEWSHOT,
    )
    return build_fewshot_prompt(problem=problem, **template)


def build_fewshot_prompt(
    problem: ProblemRecord,
    system_prompt: str,
    fewshot: list[Dict[str, str]],
    format_template: str = "Q: {question}\nA: {answer}",
) -> str:
    blocks = [system_prompt.strip()]
    for example in fewshot:
        blocks.append(_format_qa(example["question"], example["answer"], format_template))
    blocks.append(_format_qa(problem.question, "", format_template))
    return "\n\n".join(blocks)


def _resolve_fewshot_template(
    prompt_config: Dict[str, Any],
    config: Dict[str, Any] | None,
    default_system_prompt: str,
    default_fewshot: list[Dict[str, str]],
) -> Dict[str, Any]:
    template: Dict[str, Any] = {
        "system_prompt": default_system_prompt,
        "format_template": "Q: {question}\nA: {answer}",
        "fewshot": default_fewshot,
    }

    fewshot_path = prompt_config.get("fewshot_path")
    if fewshot_path:
        template.update(_load_fewshot_template(str(fewshot_path), config))

    if "system_prompt" in prompt_config:
        template["system_prompt"] = str(prompt_config["system_prompt"])
    if "format" in prompt_config:
        template["format_template"] = str(prompt_config["format"])
    if "fewshot" in prompt_config:
        template["fewshot"] = prompt_config["fewshot"]

    template["fewshot"] = _validate_fewshot(template["fewshot"])
    return template


def _load_fewshot_template(path_value: str, config: Dict[str, Any] | None) -> Dict[str, Any]:
    path = resolve_path(path_value, config)
    if path is None or not path.exists():
        raise FileNotFoundError(f"prompt.fewshot_path does not exist: {path_value}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            data = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "Install pyyaml before using a YAML prompt.fewshot_path."
                ) from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported prompt.fewshot_path extension: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Few-shot template must be a mapping: {path}")

    template: Dict[str, Any] = {}
    if "system_prompt" in data:
        template["system_prompt"] = str(data["system_prompt"])
    if "format" in data:
        template["format_template"] = str(data["format"])
    if "fewshot" in data:
        template["fewshot"] = data["fewshot"]
    return template


def _validate_fewshot(fewshot: Any) -> list[Dict[str, str]]:
    if not isinstance(fewshot, list):
        raise ValueError("Few-shot examples must be a list.")

    normalized = []
    for index, example in enumerate(fewshot):
        if not isinstance(example, dict) or "question" not in example or "answer" not in example:
            raise ValueError(f"Few-shot example {index} must contain question and answer.")
        normalized.append(
            {
                "question": str(example["question"]),
                "answer": str(example["answer"]),
            }
        )
    return normalized


def _format_qa(question: str, answer: str, format_template: str) -> str:
    return format_template.format(question=question.strip(), answer=answer.strip()).rstrip()
