"""Validated loading and plotting for factorial SFT accuracy comparisons."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any


MODES = ("equal_example", "equal_token")
MODE_LABELS = {
    "equal_example": "Equal examples",
    "equal_token": "Equal supervision tokens",
}
MODE_COLORS = {
    "equal_example": "#0072B2",
    "equal_token": "#D55E00",
}


def read_metrics(path: Path, seed: int | None) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Read one complete two-mode factorial grid plus its base-model row."""
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    required = {
        "run_name",
        "mode",
        "generator_name",
        "generator_size_b",
        "budget_tokens",
        "seed",
        "n",
        "accuracy",
    }
    missing_columns = required - set(raw_rows[0] if raw_rows else {})
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    factorial_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if raw["mode"] not in MODES:
            continue
        row_seed = int(raw["seed"])
        if seed is not None and row_seed != seed:
            continue
        factorial_rows.append(
            {
                **raw,
                "generator_size_b": float(raw["generator_size_b"]),
                "budget_tokens": int(raw["budget_tokens"]),
                "seed": row_seed,
                "n": int(raw["n"]),
                "accuracy": float(raw["accuracy"]),
            }
        )
    if not factorial_rows:
        raise ValueError(f"No factorial rows found in {path} for seed={seed}")

    observed_seeds = {row["seed"] for row in factorial_rows}
    if len(observed_seeds) != 1:
        raise ValueError(f"Expected exactly one training seed, found {sorted(observed_seeds)}")
    plotted_seed = next(iter(observed_seeds))

    keys = [
        (row["mode"], row["generator_name"], row["budget_tokens"])
        for row in factorial_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate mode/generator/budget rows found")
    generators = {row["generator_name"] for row in factorial_rows}
    budgets = {row["budget_tokens"] for row in factorial_rows}
    expected = {
        (mode, generator, budget)
        for mode in MODES
        for generator in generators
        for budget in budgets
    }
    missing_cells = expected - set(keys)
    extra_cells = set(keys) - expected
    if missing_cells or extra_cells:
        raise ValueError(
            f"Incomplete factorial grid: missing={sorted(missing_cells)} extra={sorted(extra_cells)}"
        )

    sample_sizes = {row["n"] for row in factorial_rows}
    if len(sample_sizes) != 1:
        raise ValueError(f"Expected a common evaluation size, found {sorted(sample_sizes)}")

    base_rows = [row for row in raw_rows if row["mode"] == "base"]
    if len(base_rows) != 1:
        raise ValueError(f"Expected exactly one base-model row, found {len(base_rows)}")
    base_row = {
        **base_rows[0],
        "n": int(base_rows[0]["n"]),
        "accuracy": float(base_rows[0]["accuracy"]),
    }
    return factorial_rows, base_row, plotted_seed


def _generator_label(size_b: float) -> str:
    size = f"{size_b:g}B"
    if math.isclose(size_b, 1.5):
        return f"Teacher generator: {size} (self-distillation control)"
    return f"Teacher generator: {size}"


def plot_accuracy(
    rows: list[dict[str, Any]],
    base_row: dict[str, Any],
    seed: int,
    output_prefix: Path,
    dpi: int,
) -> None:
    """Write publication-ready PNG and PDF comparisons for one training seed."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    generators = sorted(
        {(row["generator_name"], row["generator_size_b"]) for row in rows},
        key=lambda item: item[1],
    )
    budgets = sorted({row["budget_tokens"] for row in rows})
    lookup = {
        (row["mode"], row["generator_name"], row["budget_tokens"]): row["accuracy"]
        for row in rows
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        }
    )
    ncols = 2
    nrows = math.ceil(len(generators) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 7.6), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()
    x = np.arange(len(budgets), dtype=float)
    width = 0.36

    for axis, (generator_name, size_b) in zip(axes_flat, generators):
        for mode, offset in zip(MODES, (-width / 2, width / 2)):
            values = [lookup[(mode, generator_name, budget)] for budget in budgets]
            bars = axis.bar(
                x + offset,
                values,
                width=width,
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.7,
                label=MODE_LABELS[mode],
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[f"{value * 100:.1f}" for value in values],
                padding=2,
                fontsize=8.5,
            )
        axis.axhline(
            base_row["accuracy"],
            color="#555555",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            zorder=2,
        )
        axis.set_title(_generator_label(size_b))
        axis.set_xticks(x, [str(budget) for budget in budgets])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8, zorder=0)
        axis.set_axisbelow(True)

    for axis in axes_flat[len(generators) :]:
        axis.set_visible(False)
    for axis in axes_flat:
        axis.set_ylim(0.0, 0.80)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for axis in axes_flat[-ncols:]:
        if axis.get_visible():
            axis.set_xlabel("Teacher solution budget (tokens)")
    for row_index in range(nrows):
        axes_flat[row_index * ncols].set_ylabel("GSM8K accuracy")

    legend_handles = [
        Patch(facecolor=MODE_COLORS[mode], label=MODE_LABELS[mode]) for mode in MODES
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            label=f"Base student ({base_row['accuracy'] * 100:.1f}%)",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("SFT accuracy under equal-example and equal-token supervision", y=0.985)
    sample_size = rows[0]["n"]
    fig.text(
        0.5,
        0.012,
        (
            f"Locked GSM8K test[50:1319]; n={sample_size:,} per condition; training seed={seed}. "
            "Revised single-seed protocol; no estimate of training-seed variability."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.895), h_pad=2.0, w_pad=1.5)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Wrote %s", png_path)
    logging.info("Wrote %s", pdf_path)
