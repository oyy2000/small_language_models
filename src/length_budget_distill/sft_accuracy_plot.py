"""Validated loading and plotting for factorial SFT accuracy comparisons."""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MODES = ("equal_example", "equal_token")
MODE_LABELS = {
    "equal_example": "Equal examples",
    "equal_token": "Equal supervision tokens",
}
MODE_COLORS = {
    "equal_example": "#0072B2",
    "equal_token": "#D55E00",
}
COMPARISON_COLORS = (MODE_COLORS["equal_example"], MODE_COLORS["equal_token"])
BASELINE_COLOR = "#555555"
BASELINE_LINESTYLE = (0, (4, 3))


def plot_grouped_accuracy_bars(
    axis: Any,
    *,
    x_labels: Sequence[str],
    series_values: Mapping[str, Sequence[float]],
    series_colors: Mapping[str, str],
    series_intervals: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    baseline_accuracy: float | None = None,
    show_value_labels: bool = True,
) -> None:
    """Draw the shared paired-comparison bar style using proportion-scale values."""
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    series_names = list(series_values)
    if not series_names:
        raise ValueError("At least one accuracy series is required")
    if set(series_colors) != set(series_names):
        raise ValueError("series_colors must have exactly the same keys as series_values")
    if series_intervals is not None and set(series_intervals) != set(series_names):
        raise ValueError("series_intervals must have exactly the same keys as series_values")

    x = np.arange(len(x_labels), dtype=float)
    width = min(0.36, 0.80 / len(series_names))
    for series_index, series_name in enumerate(series_names):
        values = [float(value) for value in series_values[series_name]]
        if len(values) != len(x_labels):
            raise ValueError(f"Series {series_name!r} does not match x_labels")
        yerr = None
        if series_intervals is not None:
            intervals = list(series_intervals[series_name])
            if len(intervals) != len(values):
                raise ValueError(f"Intervals for {series_name!r} do not match its values")
            yerr = [
                [value - float(interval[0]) for value, interval in zip(values, intervals)],
                [float(interval[1]) - value for value, interval in zip(values, intervals)],
            ]
            if any(error < 0.0 for side in yerr for error in side):
                raise ValueError(f"Intervals for {series_name!r} do not contain their values")
        offset = (series_index - (len(series_names) - 1) / 2.0) * width
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            color=series_colors[series_name],
            edgecolor="white",
            linewidth=0.7,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
            label=series_name,
            zorder=3,
        )
        if show_value_labels:
            axis.bar_label(
                bars,
                labels=[f"{value * 100:.1f}" for value in values],
                padding=2,
                fontsize=8.5,
            )
    if baseline_accuracy is not None:
        axis.axhline(
            float(baseline_accuracy),
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
            linewidth=1.2,
            zorder=2,
        )
    axis.set_xticks(x, x_labels)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


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

    for axis, (generator_name, size_b) in zip(axes_flat, generators):
        plot_grouped_accuracy_bars(
            axis,
            x_labels=[str(budget) for budget in budgets],
            series_values={
                MODE_LABELS[mode]: [
                    lookup[(mode, generator_name, budget)] for budget in budgets
                ]
                for mode in MODES
            },
            series_colors={MODE_LABELS[mode]: MODE_COLORS[mode] for mode in MODES},
            baseline_accuracy=base_row["accuracy"],
        )
        axis.set_title(_generator_label(size_b))

    for axis in axes_flat[len(generators) :]:
        axis.set_visible(False)
    for axis in axes_flat:
        axis.set_ylim(0.0, 0.80)
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
            color=BASELINE_COLOR,
            linestyle=BASELINE_LINESTYLE,
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
