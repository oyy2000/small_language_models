#!/usr/bin/env python3
"""Tests for validated factorial SFT loss parsing and plotting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.sft_loss_plot import plot_loss_curves, read_loss_runs


class SftLossPlotTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        training_dir = root / "training"
        config_dir = training_dir / "configs"
        log_dir = training_dir / "logs"
        config_dir.mkdir(parents=True)
        log_dir.mkdir(parents=True)
        for mode in ("equal_example", "equal_token"):
            for budget_name, supervised_tokens in (
                ("short_128", 100),
                ("medium_256", 200),
                ("long_512", 300),
            ):
                run_name = f"{mode}__qwen2p5_1p5b__{budget_name}__seed_17"
                config = {
                    "training": {
                        "num_train_epochs": 1,
                        "per_device_train_batch_size": 4,
                        "gradient_accumulation_steps": 1,
                        "logging_steps": 1,
                    },
                    "factorial_metadata": {
                        "run_name": run_name,
                        "mode": mode,
                        "generator_name": "qwen2p5_1p5b",
                        "budget_name": budget_name,
                        "seed": 17,
                        "n": 8,
                        "supervised_tokens": supervised_tokens,
                    },
                }
                (config_dir / f"{run_name}.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                (log_dir / f"{run_name}.log").write_text(
                    "{'loss': 1.0, 'grad_norm': 2.0, 'learning_rate': 2e-5, 'epoch': 0.5}\n"
                    "{'loss': 0.5, 'grad_norm': 1.0, 'learning_rate': 0.0, 'epoch': 1.0}\n"
                    "{'train_runtime': 1.0, 'train_loss': 0.75, 'epoch': 1.0}\n",
                    encoding="utf-8",
                )
        return training_dir

    def test_read_loss_runs_validates_and_parses_grid(self) -> None:
        with TemporaryDirectory() as tmpdir:
            runs, seed = read_loss_runs(self._write_fixture(Path(tmpdir)), seed=17)
            self.assertEqual(seed, 17)
            self.assertEqual(len(runs), 6)
            self.assertEqual({run["budget_tokens"] for run in runs}, {128, 256, 512})
            self.assertTrue(all(run["total_steps"] == 2 for run in runs))
            self.assertTrue(all(run["train_loss"] == 0.75 for run in runs))
            self.assertEqual(runs[0]["points"][1]["step"], 2)

    def test_plot_loss_curves_writes_png_and_pdf(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs, seed = read_loss_runs(self._write_fixture(root), seed=17)
            output_prefix = root / "figures" / "loss"
            plot_loss_curves(runs, seed, output_prefix, dpi=72)
            self.assertTrue(output_prefix.with_suffix(".png").is_file())
            self.assertTrue(output_prefix.with_suffix(".pdf").is_file())


if __name__ == "__main__":
    unittest.main()
