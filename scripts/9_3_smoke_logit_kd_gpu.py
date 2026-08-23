#!/usr/bin/env python3
"""Run a two-record GPU smoke test for exact KD and top-k serialization."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.logit_kd import (
    causal_completion_logits,
    file_sha256,
    hybrid_kd_loss,
    invalid_vocab_probability_mass,
    load_protocol,
    load_teacher_and_student,
    protocol_hash,
    resolve_project_path,
    runtime_metadata,
    tokenize_completion_record,
    validate_budget_dataset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--label", default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from safetensors.torch import load_file, save_file

    protocol = load_protocol(args.config)
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    if not args.label.replace("_", "").isalnum():
        raise ValueError("Smoke label must contain only letters, digits, and underscores.")
    output_path = result_root / "smoke" / f"gpu_smoke_{args.label}.json"
    marker_path = result_root / "smoke" / f"SMOKE_COMPLETE_{args.label}"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if output_path.is_file() and marker.get("smoke_sha256") == file_sha256(output_path):
            logging.info("gpu_smoke_already_complete output=%s", output_path)
            return
        raise ValueError(f"Existing GPU smoke marker is invalid: {marker_path}")
    teacher, student, tokenizer, valid_vocab_size, model_evidence = load_teacher_and_student(
        protocol,
        train_student=True,
    )
    max_length = int(protocol["training"]["max_length"])
    examples = []
    for budget_name in ("short_128", "long_512"):
        _path, rows = validate_budget_dataset(protocol, budget_name)
        encoded_rows = [tokenize_completion_record(tokenizer, row, max_length) for row in rows]
        examples.append((budget_name, max(encoded_rows, key=lambda item: item["completion_token_count"])))
    torch.cuda.reset_peak_memory_stats()
    records = []
    optimizer = torch.optim.AdamW([parameter for parameter in student.parameters() if parameter.requires_grad], lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    last_snapshot = None
    for budget_name, encoded in examples:
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device="cuda")
        targets = torch.tensor(encoded["target_ids"], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            teacher_logits = causal_completion_logits(teacher, input_ids, encoded["prompt_token_count"])
        student_logits = causal_completion_logits(student, input_ids, encoded["prompt_token_count"])
        loss, metrics = hybrid_kd_loss(
            student_logits,
            teacher_logits,
            targets,
            alpha=0.5,
            temperature=2.0,
            valid_vocab_size=valid_vocab_size,
        )
        (loss / len(examples)).backward()
        top_values, top_ids = torch.topk(student_logits[:, :valid_vocab_size].float(), k=64, dim=-1)
        last_snapshot = {
            "topk_token_ids": top_ids.to(torch.int32).cpu(),
            "topk_logits": top_values.to(torch.float16).cpu(),
            "target_token_ids": targets.to(torch.int32).cpu(),
        }
        records.append(
            {
                "budget_name": budget_name,
                "record_id": encoded["record_id"],
                "sequence_tokens": len(encoded["input_ids"]),
                "completion_tokens": encoded["completion_token_count"],
                "loss": float(metrics["loss"]),
                "ce": float(metrics["ce"]),
                "kd": float(metrics["kd"]),
                "teacher_invalid_vocab_mass": float(
                    invalid_vocab_probability_mass(teacher_logits, valid_vocab_size).mean()
                ),
                "student_invalid_vocab_mass": float(
                    invalid_vocab_probability_mass(student_logits.detach(), valid_vocab_size).mean()
                ),
            }
        )
    optimizer.step()
    if last_snapshot is None:
        raise AssertionError("GPU smoke produced no snapshot.")
    runtime_dir = Path(os.environ.get("LBD_RUNTIME_LOGIT_ROOT", tempfile.gettempdir())) / protocol["experiment_name"] / "smoke"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = runtime_dir / "snapshot.safetensors"
    save_file(last_snapshot, tensor_path)
    loaded = load_file(tensor_path, device="cpu")
    if loaded["topk_token_ids"].shape[1] != 64 or loaded["topk_logits"].shape != loaded["topk_token_ids"].shape:
        raise ValueError("Top-k safetensors round-trip failed.")
    payload = {
        "status": "passed",
        "protocol_hash": protocol_hash(protocol),
        "records": records,
        "valid_vocab_size": valid_vocab_size,
        "top_k": 64,
        "snapshot_roundtrip": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "model_evidence": model_evidence,
        "runtime": runtime_metadata(),
    }
    write_json(output_path, payload)
    write_json(
        marker_path,
        {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "smoke_sha256": file_sha256(output_path),
        },
    )
    tensor_path.unlink()
    logging.info("gpu_smoke_complete output=%s peak_mib=%.1f", output_path, payload["peak_allocated_mib"])


if __name__ == "__main__":
    main()
