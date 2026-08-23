#!/usr/bin/env python3
"""Tests for shared SFT accuracy plotting primitives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.sft_accuracy_plot import BASELINE_COLOR, plot_grouped_accuracy_bars


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class GroupedAccuracyBarsTest(unittest.TestCase):
    def test_grouped_bars_support_intervals_and_baseline(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        plot_grouped_accuracy_bars(
            axis,
            x_labels=("128", "256"),
            series_values={"Reference": (0.4, 0.5), "Mixed": (0.45, 0.55)},
            series_colors={"Reference": "#0072B2", "Mixed": "#D55E00"},
            series_intervals={
                "Reference": ((0.3, 0.5), (0.4, 0.6)),
                "Mixed": ((0.35, 0.55), (0.45, 0.65)),
            },
            baseline_accuracy=0.42,
        )
        self.assertEqual(len(axis.patches), 4)
        self.assertTrue(any(line.get_color() == BASELINE_COLOR for line in axis.lines))
        plt.close(figure)

    def test_grouped_bars_reject_mismatched_series_lengths(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        with self.assertRaisesRegex(ValueError, "does not match x_labels"):
            plot_grouped_accuracy_bars(
                axis,
                x_labels=("128", "256"),
                series_values={"Reference": (0.4,)},
                series_colors={"Reference": "#0072B2"},
            )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
