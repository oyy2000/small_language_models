#!/bin/bash

#SBATCH -J 1_5_sml_teacher
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH -w c30
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate sft

cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/home/${USER}/projects/.cache/hf_datasets/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

CONFIG="${CONFIG:-configs/real_length_budget_template.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/real_length_budget}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
LOG_EVERY="${LOG_EVERY:-10}"

mkdir -p "/home/${USER}/projects/small_language_model/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}" "${OUTPUT_DIR}"

echo "Running on $(hostname)"
which python3
python3 -V
nvidia-smi
echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpu_ids=${GPU_IDS}"
echo "log_every=${LOG_EVERY}"

wait_for_all_gpus_free() {
    local threshold_mib="${1:-40000}"
    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | awk -v t="${threshold_mib}" '$1 >= t {c++} END {print c+0}')
        if [ "${free_count}" -ge 4 ]; then
            echo "All 4 GPUs have at least ${threshold_mib} MiB free."
            nvidia-smi
            break
        fi
        echo "Waiting for free GPUs: ${free_count}/4 above ${threshold_mib} MiB at $(date)"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

wait_for_all_gpus_free 40000

GPU_IDS="${GPU_IDS}" LOG_EVERY="${LOG_EVERY}" bash scripts/run_length_budget_4gpu.sh \
    "${CONFIG}" \
    "${OUTPUT_DIR}"

python3 - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
budgets = ("small", "medium", "large")

def split_jsonl(input_path, output_prefix, budget_getter):
    handles = {
        budget: (output_dir / f"{output_prefix}_{budget}.jsonl").open("w", encoding="utf-8")
        for budget in budgets
    }
    counts = {budget: 0 for budget in budgets}
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                budget = budget_getter(row)
                if budget not in handles:
                    raise ValueError(f"{input_path}:{line_number} has unexpected budget_name={budget!r}")
                handles[budget].write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[budget] += 1
    finally:
        for handle in handles.values():
            handle.close()
    for budget in budgets:
        print(f"{output_prefix}_{budget}={output_dir / f'{output_prefix}_{budget}.jsonl'} count={counts[budget]}")

split_jsonl(
    output_dir / "traces_merged.jsonl",
    "traces",
    lambda row: row.get("budget_name"),
)
split_jsonl(
    output_dir / "sft_merged.jsonl",
    "sft",
    lambda row: row.get("metadata", {}).get("budget_name"),
)
PY

echo "Done."
echo "merged_traces=${OUTPUT_DIR}/traces_merged.jsonl"
echo "merged_sft=${OUTPUT_DIR}/sft_merged.jsonl"
echo "sft_small=${OUTPUT_DIR}/sft_small.jsonl"
echo "sft_medium=${OUTPUT_DIR}/sft_medium.jsonl"
echo "sft_large=${OUTPUT_DIR}/sft_large.jsonl"
