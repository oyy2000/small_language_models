#!/bin/bash

#SBATCH -J 2_2_sml_sft
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

BASE_CONFIG="${BASE_CONFIG:-configs/student_sft_template.json}"
GRID_TEMPLATE="${GRID_TEMPLATE:-configs/student_sft_grid_template.json}"
WORK_DIR="${WORK_DIR:-results/student_sft_grid}"
GRID_NAME="${GRID_NAME:-qwen_student_sft_sml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/student_sft_grid}"
GPU_IDS="${GPU_IDS:-1,2,3}"
USE_SYSTEMD_RUN="${USE_SYSTEMD_RUN:-0}"
RUNTIME_GRID_CONFIG="${WORK_DIR}/configs/student_sft_grid_runtime.json"
DRY_RUN_LOG="${WORK_DIR}/dry_run_commands.txt"

mkdir -p "/home/${USER}/projects/small_language_model/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}" "${WORK_DIR}/configs" "${WORK_DIR}/logs"

echo "Running on $(hostname)"
which python3
python3 -V
nvidia-smi
echo "base_config=${BASE_CONFIG}"
echo "grid_template=${GRID_TEMPLATE}"
echo "work_dir=${WORK_DIR}"
echo "grid_name=${GRID_NAME}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "gpu_ids=${GPU_IDS}"
echo "use_systemd_run=${USE_SYSTEMD_RUN}"

python3 - "${GRID_TEMPLATE}" "${RUNTIME_GRID_CONFIG}" "${GRID_NAME}" "${CHECKPOINT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
grid_name = sys.argv[3]
checkpoint_root = sys.argv[4]

with template_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)

config["name"] = grid_name
config["checkpoint_root"] = checkpoint_root

datasets = config.get("grid", {}).get("data.train_path", [])
expected = {
    "results/real_length_budget/sft_small.jsonl",
    "results/real_length_budget/sft_medium.jsonl",
    "results/real_length_budget/sft_large.jsonl",
}
if set(datasets) != expected:
    raise SystemExit(f"{template_path} does not point at the expected S/M/L datasets: {datasets}")

missing = [path for path in datasets if not Path(path).is_file()]
if missing:
    raise SystemExit(f"Missing SFT datasets: {missing}")

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

print(f"wrote_runtime_grid_config={output_path}")
print(f"datasets={datasets}")
print(f"learning_rates={config.get('grid', {}).get('training.learning_rate', [])}")
print(f"lora_ranks={config.get('grid', {}).get('student.lora.r', [])}")
PY

python3 -u scripts/2_2_grid_search_student_sft.py \
    --base-config "${BASE_CONFIG}" \
    --grid-config "${RUNTIME_GRID_CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dry-run > "${DRY_RUN_LOG}"

echo "dry_run_log=${DRY_RUN_LOG}"
tail -20 "${DRY_RUN_LOG}"

wait_for_requested_gpus_free() {
    local threshold_mib="${1:-40000}"
    local raw_gpu_ids="${2:-${GPU_IDS}}"
    local requested_gpus=()
    local normalized_gpu_ids=()
    local raw_gpu_id
    local normalized_gpu_ids_csv

    IFS=',' read -r -a requested_gpus <<< "${raw_gpu_ids}"
    for raw_gpu_id in "${requested_gpus[@]}"; do
        local gpu_id="${raw_gpu_id//[[:space:]]/}"
        if [ -n "${gpu_id}" ]; then
            normalized_gpu_ids+=("${gpu_id}")
        fi
    done
    if [ "${#normalized_gpu_ids[@]}" -eq 0 ]; then
        echo "GPU_IDS=${raw_gpu_ids} did not contain any usable GPU ids." >&2
        exit 1
    fi
    normalized_gpu_ids_csv="$(IFS=','; echo "${normalized_gpu_ids[*]}")"

    while true; do
        local ready_count=0
        for gpu_id in "${normalized_gpu_ids[@]}"; do
            local free_mib
            free_mib=$(nvidia-smi \
                -i "${gpu_id}" \
                --query-gpu=memory.free \
                --format=csv,noheader,nounits)
            free_mib="${free_mib//[[:space:]]/}"
            if [ "${free_mib}" -ge "${threshold_mib}" ]; then
                ready_count=$((ready_count + 1))
            fi
        done

        if [ "${ready_count}" -eq "${#normalized_gpu_ids[@]}" ]; then
            echo "Requested GPUs (${normalized_gpu_ids_csv}) all have at least ${threshold_mib} MiB free."
            for gpu_id in "${normalized_gpu_ids[@]}"; do
                nvidia-smi \
                    -i "${gpu_id}" \
                    --query-gpu=index,memory.used,memory.free,utilization.gpu \
                    --format=csv,noheader
            done
            break
        fi

        echo "Waiting for requested GPUs: ${ready_count}/${#normalized_gpu_ids[@]} above ${threshold_mib} MiB at $(date); GPU_IDS=${normalized_gpu_ids_csv}"
        for gpu_id in "${normalized_gpu_ids[@]}"; do
            nvidia-smi \
                -i "${gpu_id}" \
                --query-gpu=index,memory.used,memory.free,utilization.gpu \
                --format=csv,noheader
        done
        sleep 300
    done
}

wait_for_requested_gpus_free 40000 "${GPU_IDS}"

run_grid_search() {
    if [ "${USE_SYSTEMD_RUN}" = "1" ]; then
        if [ -z "${XDG_RUNTIME_DIR:-}" ] || [ ! -S "${XDG_RUNTIME_DIR}/bus" ]; then
            echo "USE_SYSTEMD_RUN=1 but user systemd bus is unavailable; running directly instead."
        elif command -v systemd-run >/dev/null 2>&1; then
            systemd-run --user --scope -p MemoryMax=32G "$@"
            return
        else
            echo "USE_SYSTEMD_RUN=1 but systemd-run is not available; running directly instead."
        fi
    fi
    "$@"
}

run_grid_search python3 -u scripts/2_2_grid_search_student_sft.py \
    --base-config "${BASE_CONFIG}" \
    --grid-config "${RUNTIME_GRID_CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --gpu-ids "${GPU_IDS}" \
    --skip-existing
