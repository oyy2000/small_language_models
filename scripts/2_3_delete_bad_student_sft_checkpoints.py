#!/usr/bin/env python3
"""Delete known incomplete student SFT grid checkpoints."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_RUN_IDS = ("018", "019", "020", "021")
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/student_sft_grid",
        help="Student SFT grid checkpoint root.",
    )
    parser.add_argument(
        "--run-ids",
        default=",".join(DEFAULT_RUN_IDS),
        help="Comma-separated grid run ids to delete.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete directories. Without this flag, only print what would be removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    run_ids = [item.strip() for item in args.run_ids.split(",") if item.strip()]

    targets = []
    errors = []
    for run_id in run_ids:
        matches = sorted(checkpoint_root.glob(f"qwen-student-sft-grid_{run_id}_*"))
        matches = [path for path in matches if path.is_dir()]
        if not matches:
            print(f"skip_missing run_id={run_id}")
            continue
        if len(matches) > 1:
            errors.append(f"multiple directories matched run_id={run_id}: {matches}")
            continue

        checkpoint_dir = matches[0]
        missing = [name for name in REQUIRED_ADAPTER_FILES if not (checkpoint_dir / name).is_file()]
        if not missing:
            errors.append(f"refusing to delete complete checkpoint: {checkpoint_dir}")
            continue
        targets.append((checkpoint_dir, missing))

    if errors:
        for error in errors:
            print(f"error: {error}")
        raise SystemExit(1)

    if not targets:
        print("no incomplete checkpoint directories found")
        return

    for checkpoint_dir, missing in targets:
        missing_text = ",".join(missing)
        if args.apply:
            print(f"delete {checkpoint_dir} missing={missing_text}")
            shutil.rmtree(checkpoint_dir)
        else:
            print(f"dry_run delete {checkpoint_dir} missing={missing_text}")

    if not args.apply:
        print("dry_run complete; rerun with --apply to delete these directories")


if __name__ == "__main__":
    main()
