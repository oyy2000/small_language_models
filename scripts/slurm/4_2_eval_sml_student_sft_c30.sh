#!/bin/bash

#SBATCH -J 4_2_eval_sml_sft
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

MANIFEST="${MANIFEST:-results/student_sft_grid/manifest.json}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/real_length_budget_template.json}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/student_sft_grid/eval}"
GPU_IDS="${GPU_IDS:-1,2,3}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_SYSTEMD_RUN="${USE_SYSTEMD_RUN:-0}"
DRY_RUN_LOG="${OUTPUT_DIR}/dry_run_commands.txt"

mkdir -p "/home/${USER}/projects/small_language_model/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}" "${OUTPUT_DIR}/logs"

echo "Running on $(hostname)"
which python3
python3 -V
nvidia-smi
echo "manifest=${MANIFEST}"
echo "eval_config=${EVAL_CONFIG}"
echo "model_name=${MODEL_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpu_ids=${GPU_IDS}"
echo "split=${SPLIT}"
echo "limit=${LIMIT}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "skip_existing=${SKIP_EXISTING}"
echo "use_systemd_run=${USE_SYSTEMD_RUN}"

python3 - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"Manifest does not exist: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)

runs = manifest.get("runs", [])
if not runs:
    raise SystemExit(f"Manifest has no runs: {manifest_path}")

missing = []
for entry in runs:
    output_dir = Path(entry["output_dir"])
    needed = [output_dir / "adapter_config.json", output_dir / "adapter_model.safetensors"]
    if not all(path.is_file() for path in needed):
        missing.append(str(output_dir))

if missing:
    print(f"complete_checkpoints={len(runs) - len(missing)}/{len(runs)}")
    for path in missing[:20]:
        print(f"missing_checkpoint={path}")
    raise SystemExit("Some checkpoints are incomplete; run or resume S/M/L SFT before evaluation.")

print(f"complete_checkpoints={len(runs)}/{len(runs)}")
print(f"first_run={runs[0]['run_name']}")
print(f"last_run={runs[-1]['run_name']}")
PY

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
        local gpu_id
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

run_eval_grid() {
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

eval_args=(
    python3 -u scripts/4_2_eval_grid.py
    --manifest "${MANIFEST}"
    --config "${EVAL_CONFIG}"
    --model-name "${MODEL_NAME}"
    --split "${SPLIT}"
    --limit "${LIMIT}"
    --output-dir "${OUTPUT_DIR}"
    --gpu-ids "${GPU_IDS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --torch-dtype "${TORCH_DTYPE}"
)

if [ "${SKIP_EXISTING}" = "1" ]; then
    eval_args+=(--skip-existing)
fi

"${eval_args[@]}" --dry-run > "${DRY_RUN_LOG}"
echo "dry_run_log=${DRY_RUN_LOG}"
tail -20 "${DRY_RUN_LOG}"

wait_for_requested_gpus_free 40000 "${GPU_IDS}"

run_eval_grid "${eval_args[@]}"
