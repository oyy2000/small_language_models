#!/usr/bin/env python3
"""Extract matched-trajectory top-k logits and exact teacher/student distances."""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from length_budget_distill.factorial import validated_adapter_evidence
from length_budget_distill.logit_kd import (
    causal_completion_logits,
    copy_with_retries,
    file_sha256,
    invalid_vocab_probability_mass,
    load_protocol,
    load_teacher_and_student,
    protocol_hash,
    read_json,
    resolve_project_path,
    runtime_metadata,
    tokenize_completion_record,
    validate_budget_dataset,
    validated_training_marker,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/capacity_length_logit_kd_seed17_v1.json")
    parser.add_argument("--budget", choices=["short_128", "medium_256", "long_512"], required=True)
    parser.add_argument("--method", choices=["teacher", "base", "sft", "kd"], required=True)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=-1, help="-1 extracts every shard in one model load.")
    parser.add_argument("--skip-complete", action="store_true")
    return parser.parse_args()


def _student_adapter(protocol: Dict[str, Any], budget: str, method: str) -> Path | None:
    if method in {"teacher", "base"}:
        return None
    if method == "sft":
        path = resolve_project_path(protocol["budgets"][budget]["baseline_adapter"])
        if validated_adapter_evidence(path) is None:
            raise ValueError(f"Parent SFT adapter is incomplete: {path}")
        return path
    path = resolve_project_path(protocol["outputs"]["checkpoint_root"]) / "formal" / f"{budget}__seed_17"
    if validated_training_marker(path) is None:
        raise ValueError(f"Formal KD adapter is incomplete: {path}")
    return path


def _snapshot_tensors(logits: Any, targets: Any, valid_vocab_size: int, top_k: int) -> Dict[str, Any]:
    import torch

    valid = logits[:, :valid_vocab_size].float()
    logsumexp = torch.logsumexp(valid, dim=-1)
    log_probs = valid - logsumexp[:, None]
    probabilities = log_probs.exp()
    entropy = -torch.sum(probabilities * log_probs, dim=-1)
    top_values, top_ids = torch.topk(valid, k=top_k, dim=-1, largest=True, sorted=True)
    token_index = torch.arange(targets.numel(), device=targets.device)
    target_logits = valid[token_index, targets]
    target_rank = torch.sum(valid > target_logits[:, None], dim=-1) + 1
    return {
        "topk_token_ids": top_ids.to(torch.int32).cpu(),
        "topk_logits": top_values.to(torch.float16).cpu(),
        "logsumexp": logsumexp.to(torch.float32).cpu(),
        "entropy": entropy.to(torch.float32).cpu(),
        "target_token_ids": targets.to(torch.int32).cpu(),
        "target_logits": target_logits.to(torch.float16).cpu(),
        "target_rank": target_rank.to(torch.int32).cpu(),
        "invalid_vocab_mass": invalid_vocab_probability_mass(logits, valid_vocab_size).to(torch.float32).cpu(),
        "log_probs": log_probs,
        "probabilities": probabilities,
    }


def _pair_metrics(teacher_snapshot: Dict[str, Any], student_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    teacher_log_probs = teacher_snapshot["log_probs"]
    student_log_probs = student_snapshot["log_probs"]
    teacher_probs = teacher_snapshot["probabilities"]
    student_probs = student_snapshot["probabilities"]
    kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    mixture = 0.5 * (teacher_probs + student_probs)
    mixture_log = torch.log(mixture.clamp_min(torch.finfo(mixture.dtype).tiny))
    js = 0.5 * torch.sum(teacher_probs * (teacher_log_probs - mixture_log), dim=-1)
    js += 0.5 * torch.sum(student_probs * (student_log_probs - mixture_log), dim=-1)
    teacher_top = teacher_snapshot["topk_token_ids"].to(student_snapshot["topk_token_ids"].device)
    student_top = student_snapshot["topk_token_ids"]
    overlap = (teacher_top[:, :, None] == student_top[:, None, :]).any(dim=-1).sum(dim=-1)
    return {
        "teacher_to_student_kl": kl.to(torch.float32).cpu(),
        "jensen_shannon": js.to(torch.float32).cpu(),
        "topk_overlap_count": overlap.to(torch.int16).cpu(),
    }


def _clean_snapshot(snapshot: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in snapshot.items()
        if key not in {"log_probs", "probabilities"}
    }


def _complete_marker(marker_path: Path) -> Dict[str, Any] | None:
    if not marker_path.is_file():
        return None
    marker = read_json(marker_path)
    tensor_path = Path(str(marker.get("tensor_path")))
    metadata_path = Path(str(marker.get("metadata_path")))
    if (
        marker.get("status") == "complete"
        and tensor_path.is_file()
        and metadata_path.is_file()
        and marker.get("tensor_sha256") == file_sha256(tensor_path)
        and marker.get("metadata_sha256") == file_sha256(metadata_path)
    ):
        return marker
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    protocol = load_protocol(args.config)
    num_shards = args.num_shards or int(protocol["outputs"]["logit_shards_per_snapshot"])
    if num_shards <= 0 or args.shard_index < -1 or args.shard_index >= num_shards:
        raise ValueError("Invalid logit shard topology.")
    shard_indices = list(range(num_shards)) if args.shard_index == -1 else [args.shard_index]
    result_root = resolve_project_path(protocol["outputs"]["result_root"])
    output_dir = result_root / "formal" / "logits" / args.budget / args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_shards = []
    for shard_index in shard_indices:
        marker_path = output_dir / f"shard_{shard_index:02d}_of_{num_shards:02d}.complete.json"
        marker = _complete_marker(marker_path)
        if marker is not None and args.skip_complete:
            logging.info("logit_shard_already_complete marker=%s", marker_path)
            continue
        if marker is not None:
            raise FileExistsError(f"Logit shard is already complete: {marker_path}")
        stem = f"shard_{shard_index:02d}_of_{num_shards:02d}"
        existing = [path for path in (output_dir / f"{stem}.safetensors", output_dir / f"{stem}.json") if path.exists()]
        if existing:
            raise FileExistsError(f"Incomplete logit shard artifacts exist: {existing}")
        pending_shards.append(shard_index)
    if not pending_shards:
        return

    data_path, rows = validate_budget_dataset(protocol, args.budget)
    adapter = _student_adapter(protocol, args.budget, args.method)
    teacher, student, tokenizer, valid_vocab_size, model_evidence = load_teacher_and_student(
        protocol,
        student_adapter=adapter,
        train_student=False,
    )
    if args.method == "teacher":
        del student
        student = None
        import torch

        torch.cuda.empty_cache()
    max_length = int(protocol["training"]["max_length"])
    top_k = int(protocol["kd"]["top_k"])
    runtime_base = Path(os.environ.get("LBD_RUNTIME_LOGIT_ROOT", tempfile.gettempdir()))
    for shard_index in pending_shards:
        assigned = [(index, row) for index, row in enumerate(rows) if index % num_shards == shard_index]
        tensors: Dict[str, List[Any]] = {}
        record_metadata = []
        offset = 0
        for local_index, (source_index, row) in enumerate(assigned, start=1):
            import torch

            encoded = tokenize_completion_record(tokenizer, row, max_length)
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device="cuda")
            targets = torch.tensor(encoded["target_ids"], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                teacher_logits = causal_completion_logits(teacher, input_ids, encoded["prompt_token_count"])
                teacher_snapshot = _snapshot_tensors(teacher_logits, targets, valid_vocab_size, top_k)
                if args.method == "teacher":
                    output_tensors = _clean_snapshot(teacher_snapshot)
                else:
                    if student is None:
                        raise AssertionError("Student model was not loaded.")
                    student_logits = causal_completion_logits(student, input_ids, encoded["prompt_token_count"])
                    student_snapshot = _snapshot_tensors(student_logits, targets, valid_vocab_size, top_k)
                    output_tensors = _clean_snapshot(student_snapshot)
                    output_tensors.update(_pair_metrics(teacher_snapshot, student_snapshot))
                    del student_logits, student_snapshot
            token_count = encoded["completion_token_count"]
            for key, value in output_tensors.items():
                tensors.setdefault(key, []).append(value)
            record_metadata.append(
                {
                    "source_index": source_index,
                    "record_id": encoded["record_id"],
                    "problem_id": encoded["problem_id"],
                    "prompt_token_count": encoded["prompt_token_count"],
                    "completion_token_count": token_count,
                    "offset_start": offset,
                    "offset_end": offset + token_count,
                }
            )
            offset += token_count
            del input_ids, targets, teacher_logits, teacher_snapshot, output_tensors
            if local_index % 25 == 0 or local_index == len(assigned):
                logging.info(
                    "logit_progress budget=%s method=%s shard=%d/%d records=%d/%d tokens=%d",
                    args.budget,
                    args.method,
                    shard_index,
                    num_shards,
                    local_index,
                    len(assigned),
                    offset,
                )
        if not tensors:
            raise ValueError(f"Logit shard has no assigned records: {shard_index}/{num_shards}")
        concatenated = {key: torch.cat(values, dim=0).contiguous() for key, values in tensors.items()}
        concatenated["record_offsets"] = torch.tensor(
            [0] + [int(item["offset_end"]) for item in record_metadata], dtype=torch.int64
        )
        runtime_dir = runtime_base / protocol["experiment_name"] / args.budget / args.method
        runtime_dir.mkdir(parents=True, exist_ok=True)
        stem = f"shard_{shard_index:02d}_of_{num_shards:02d}"
        runtime_tensor = runtime_dir / f"{stem}.safetensors"
        runtime_metadata_path = runtime_dir / f"{stem}.json"
        if runtime_tensor.exists() or runtime_metadata_path.exists():
            raise FileExistsError(f"Runtime logit shard already exists: {runtime_dir / stem}")
        from safetensors.torch import save_file

        save_file(concatenated, runtime_tensor)
        metadata = {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "budget_name": args.budget,
            "method": args.method,
            "student_adapter": str(adapter) if adapter is not None else None,
            "source_path": str(data_path),
            "source_sha256": file_sha256(data_path),
            "source_records": len(rows),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "records": len(record_metadata),
            "completion_tokens": offset,
            "top_k": top_k,
            "valid_vocab_size": valid_vocab_size,
            "model_evidence": model_evidence,
            "record_metadata": record_metadata,
            "tensor_shapes": {key: list(value.shape) for key, value in concatenated.items()},
            "source_code_sha256": {
                "src/length_budget_distill/logit_kd.py": file_sha256(
                    PROJECT_ROOT / "src/length_budget_distill/logit_kd.py"
                ),
                "scripts/11_1_extract_matched_logit_snapshots.py": file_sha256(Path(__file__).resolve()),
            },
            "runtime": runtime_metadata(),
        }
        write_json(runtime_metadata_path, metadata)
        output_tensor = output_dir / runtime_tensor.name
        output_metadata = output_dir / runtime_metadata_path.name
        copy_with_retries(runtime_tensor, output_tensor)
        copy_with_retries(runtime_metadata_path, output_metadata)
        marker = {
            "status": "complete",
            "protocol_hash": protocol_hash(protocol),
            "budget_name": args.budget,
            "method": args.method,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "records": len(record_metadata),
            "completion_tokens": offset,
            "tensor_path": str(output_tensor),
            "tensor_sha256": file_sha256(output_tensor),
            "metadata_path": str(output_metadata),
            "metadata_sha256": file_sha256(output_metadata),
        }
        write_json(output_dir / f"{stem}.complete.json", marker)
        runtime_tensor.unlink()
        runtime_metadata_path.unlink()
        logging.info("logit_shard_complete tensor=%s", output_tensor)


if __name__ == "__main__":
    main()
