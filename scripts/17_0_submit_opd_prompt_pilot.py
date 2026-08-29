#!/usr/bin/env python3
"""Submit the phase-17 OPD pilot as an audited, dependency-ordered Slurm DAG."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED_NODES = {"c30": "a6000", "c31": "a6000", "c32": "a5000ada"}
STAGES = (
    ("reference", "scripts/slurm/17_1_generate_opd_references_c49.sh"),
    ("preflight", "scripts/slurm/17_3_run_opd_preflight_c49.sh"),
    ("training", "scripts/slurm/17_4_train_opd_policies_c49.sh"),
    ("evaluation", "scripts/slurm/17_5_eval_analyze_audit_opd_c49.sh"),
)


def parse_node_target(value: str) -> str:
    nodes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not nodes or len(nodes) != len(set(nodes)):
        raise argparse.ArgumentTypeError("Node target must contain unique approved nodes.")
    unknown = sorted(set(nodes) - set(APPROVED_NODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unapproved nodes: {','.join(unknown)}")
    partitions = {APPROVED_NODES[node] for node in nodes}
    if len(partitions) != 1:
        raise argparse.ArgumentTypeError(
            "A node target must use one partition; C30/C31 may be combined, but C32 is separate."
        )
    return ",".join(nodes)


def partition_for_target(target: str) -> str:
    return APPROVED_NODES[target.split(",", 1)[0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/capacity_length_opd_prompt_pilot_v1.json"
    )
    parser.add_argument("--reference-node", default="c31", type=parse_node_target)
    parser.add_argument("--preflight-node", default="c31", type=parse_node_target)
    parser.add_argument("--training-node", default="c31", type=parse_node_target)
    parser.add_argument("--evaluation-node", default="c31", type=parse_node_target)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def submit_stage(
    *,
    name: str,
    script: str,
    node: str,
    config: Path,
    log_dir: Path,
    dependency: Optional[str],
    dry_run: bool,
) -> str:
    command: List[str] = [
        "sbatch",
        "--parsable",
        "--oversubscribe",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=16",
        f"--partition={partition_for_target(node)}",
        f"--nodelist={node}",
        f"--job-name=opd17_{name}",
        f"--output={log_dir}/%x_%j.out",
        f"--export=ALL,CONFIG={config}",
    ]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(PROJECT_ROOT / script))
    if dry_run:
        print(" ".join(command))
        return f"DRY_{name.upper()}"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip().split(";")[0]


def main() -> None:
    args = parse_args()
    config = (PROJECT_ROOT / args.config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    protocol = json.loads(config.read_text(encoding="utf-8"))
    result_root = (PROJECT_ROOT / protocol["outputs"]["result_root"]).resolve()
    log_dir = result_root / "slurm"
    if not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/17_0_prepare_opd_storage.py"),
                "--config",
                str(config),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        log_dir.mkdir(parents=True, exist_ok=True)

    stage_nodes: Dict[str, str] = {
        "reference": args.reference_node,
        "preflight": args.preflight_node,
        "training": args.training_node,
        "evaluation": args.evaluation_node,
    }
    job_ids: Dict[str, str] = {}
    dependency: Optional[str] = None
    for name, script in STAGES:
        job_id = submit_stage(
            name=name,
            script=script,
            node=stage_nodes[name],
            config=config,
            log_dir=log_dir,
            dependency=dependency,
            dry_run=args.dry_run,
        )
        job_ids[name] = job_id
        dependency = job_id

    payload = {
        "config": str(config),
        "result_root": str(result_root),
        "stage_nodes": stage_nodes,
        "job_ids": job_ids,
        "status": "dry_run" if args.dry_run else "submitted",
    }
    if not args.dry_run:
        payload["submitted_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest_path = log_dir / f"submission_{job_ids['reference']}.json"
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite submission manifest: {manifest_path}")
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload["submission_manifest"] = str(manifest_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
