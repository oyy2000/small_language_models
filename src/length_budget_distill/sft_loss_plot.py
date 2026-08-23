"""Validated parsing and plotting of factorial SFT training-loss logs."""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any


MODES = ("equal_example", "equal_token")
MODE_LABELS = {
    "equal_example": "Equal examples",
    "equal_token": "Equal supervision tokens",
}
BUDGET_LABELS = {
    128: "Small / short (128)",
    256: "Medium (256)",
    512: "Large / long (512)",
}
BUDGET_COLORS = {
    128: "#0072B2",
    256: "#E69F00",
    512: "#009E73",
}
_DICT_PATTERN = re.compile(r"\{[^{}\r\n]+\}")


def read_loss_runs(training_dir: Path, seed: int | None) -> tuple[list[dict[str, Any]], int]:
    """Read a complete factorial grid from saved configs and Trainer logs."""
    config_dir = training_dir / "configs"
    log_dir = training_dir / "logs"
    if not config_dir.is_dir() or not log_dir.is_dir():
        raise FileNotFoundError(
            f"Expected configs/ and logs/ below training directory: {training_dir}"
        )

    runs: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for config_path in sorted(config_dir.glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = config.get("factorial_metadata", {})
        mode = metadata.get("mode")
        if mode not in MODES:
            continue
        run_seed = int(metadata["seed"])
        if seed is not None and run_seed != seed:
            continue
        observed_seeds.add(run_seed)

        run_name = str(metadata["run_name"])
        if config_path.stem != run_name:
            raise ValueError(
                f"Training config filename/run-name mismatch: {config_path.name} != {run_name}"
            )
        log_path = log_dir / f"{run_name}.log"
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing training log for {run_name}: {log_path}")

        training = config.get("training", {})
        n_examples = int(metadata["n"])
        batch_size = int(training["per_device_train_batch_size"])
        gradient_accumulation = int(training["gradient_accumulation_steps"])
        epochs = float(training["num_train_epochs"])
        logging_steps = int(training["logging_steps"])
        if min(n_examples, batch_size, gradient_accumulation, logging_steps) <= 0 or epochs <= 0:
            raise ValueError(f"Invalid training dimensions in {config_path}")
        batches_per_epoch = math.ceil(n_examples / batch_size)
        updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation)
        total_steps = math.ceil(updates_per_epoch * epochs)

        log_dicts = _read_python_dicts(log_path)
        logged_losses = [item for item in log_dicts if "loss" in item]
        summaries = [item for item in log_dicts if "train_loss" in item]
        if not logged_losses or len(summaries) != 1:
            raise ValueError(
                f"Expected logged losses and exactly one train summary in {log_path}; "
                f"loss_points={len(logged_losses)} summaries={len(summaries)}"
            )

        points = []
        for index, item in enumerate(logged_losses, start=1):
            global_step = index * logging_steps
            if global_step > total_steps:
                raise ValueError(
                    f"Inferred logged step exceeds total steps for {run_name}: "
                    f"step={global_step} total={total_steps}"
                )
            points.append(
                {
                    "step": global_step,
                    "epoch": float(item["epoch"]),
                    "loss": float(item["loss"]),
                    "learning_rate": float(item["learning_rate"]),
                    "grad_norm": float(item["grad_norm"]),
                }
            )

        budget_name = str(metadata["budget_name"])
        runs.append(
            {
                "run_name": run_name,
                "mode": mode,
                "generator_name": str(metadata["generator_name"]),
                "generator_size_b": _generator_size_b(str(metadata["generator_name"])),
                "budget_name": budget_name,
                "budget_tokens": _budget_tokens(budget_name),
                "seed": run_seed,
                "n_examples": n_examples,
                "supervised_tokens": int(metadata["supervised_tokens"]),
                "total_steps": total_steps,
                "logging_steps": logging_steps,
                "train_loss": float(summaries[0]["train_loss"]),
                "points": points,
                "config_path": str(config_path),
                "log_path": str(log_path),
            }
        )

    if not runs:
        raise ValueError(f"No factorial SFT loss runs found in {training_dir} for seed={seed}")
    if len(observed_seeds) != 1:
        raise ValueError(f"Expected exactly one training seed, found {sorted(observed_seeds)}")
    plotted_seed = next(iter(observed_seeds))
    _validate_grid(runs)
    runs.sort(
        key=lambda item: (
            item["generator_size_b"],
            MODES.index(item["mode"]),
            item["budget_tokens"],
        )
    )
    return runs, plotted_seed


def plot_loss_curves(
    runs: list[dict[str, Any]],
    seed: int,
    output_prefix: Path,
    dpi: int,
) -> None:
    """Write publication-ready PNG and PDF loss-curve comparisons."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    generators = sorted(
        {(run["generator_name"], run["generator_size_b"]) for run in runs},
        key=lambda item: item[1],
    )
    budgets = sorted({run["budget_tokens"] for run in runs})
    lookup = {
        (run["mode"], run["generator_name"], run["budget_tokens"]): run
        for run in runs
    }
    max_step = max(run["total_steps"] for run in runs)
    max_loss = max(point["loss"] for run in runs for point in run["points"])
    logging_intervals = {run["logging_steps"] for run in runs}
    if len(logging_intervals) != 1:
        raise ValueError(
            f"Expected one logging interval for a shared figure note, found {logging_intervals}"
        )
    logging_steps = next(iter(logging_intervals))

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.titlesize": 15,
        }
    )
    fig, axes = plt.subplots(
        len(generators),
        len(MODES),
        figsize=(12.0, 12.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for column, mode in enumerate(MODES):
        axes[0, column].set_title(MODE_LABELS[mode], pad=10)
    for row, (generator_name, generator_size_b) in enumerate(generators):
        for column, mode in enumerate(MODES):
            axis = axes[row, column]
            for budget in budgets:
                run = lookup[(mode, generator_name, budget)]
                steps = [point["step"] for point in run["points"]]
                losses = [point["loss"] for point in run["points"]]
                axis.plot(
                    steps,
                    losses,
                    color=BUDGET_COLORS[budget],
                    linewidth=1.7,
                    marker="o",
                    markersize=3.2,
                    label=BUDGET_LABELS.get(budget, str(budget)),
                    zorder=3,
                )
            axis.grid(color="#D9D9D9", linewidth=0.8, alpha=0.8, zorder=0)
            axis.set_axisbelow(True)
            axis.text(
                0.985,
                0.94,
                _generator_label(generator_size_b),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                color="#333333",
            )
            if column == 0:
                axis.set_ylabel("Logged completion-token CE loss")
            if row == len(generators) - 1:
                axis.set_xlabel("Optimizer step")

    for axis in np.asarray(axes).ravel():
        axis.set_xlim(0, max_step * 1.03)
        axis.set_ylim(0, max_loss * 1.08)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BUDGET_COLORS[budget],
            linewidth=2,
            marker="o",
            markersize=4,
            label=BUDGET_LABELS.get(budget, str(budget)),
        )
        for budget in budgets
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.944),
        ncol=len(legend_handles),
        frameon=False,
    )
    fig.suptitle("Completion-only SFT training loss by teacher solution length", y=0.988)
    fig.text(
        0.5,
        0.012,
        (
            f"Seed {seed}. Each point is the Trainer mean over the preceding "
            f"{logging_steps} optimizer steps. "
            "Loss is next-token cross-entropy over unmasked assistant-completion tokens only; "
            "equal-token curve endpoints differ because run step counts differ."
        ),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.048, 0.985, 0.925), h_pad=1.25, w_pad=1.15)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Wrote %s", png_path)
    logging.info("Wrote %s", pdf_path)


def _read_python_dicts(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    for match in _DICT_PATTERN.finditer(text):
        try:
            value = ast.literal_eval(match.group())
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _budget_tokens(budget_name: str) -> int:
    match = re.search(r"_(\d+)$", budget_name)
    if not match:
        raise ValueError(f"Could not parse token budget from {budget_name!r}")
    return int(match.group(1))


def _generator_size_b(generator_name: str) -> float:
    decimal_match = re.search(r"_(\d+)p(\d+)b$", generator_name)
    if decimal_match:
        return float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")
    integer_match = re.search(r"_(\d+)b$", generator_name)
    if integer_match:
        return float(integer_match.group(1))
    raise ValueError(f"Could not parse generator size from {generator_name!r}")


def _generator_label(size_b: float) -> str:
    label = f"Teacher {size_b:g}B"
    if math.isclose(size_b, 1.5):
        return f"{label} (self-distillation control)"
    return label


def _validate_grid(runs: list[dict[str, Any]]) -> None:
    keys = [(run["mode"], run["generator_name"], run["budget_tokens"]) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate mode/generator/budget loss runs found")
    generators = {run["generator_name"] for run in runs}
    budgets = {run["budget_tokens"] for run in runs}
    expected = {
        (mode, generator, budget)
        for mode in MODES
        for generator in generators
        for budget in budgets
    }
    missing = expected - set(keys)
    extra = set(keys) - expected
    if missing or extra:
        raise ValueError(f"Incomplete loss grid: missing={sorted(missing)} extra={sorted(extra)}")
