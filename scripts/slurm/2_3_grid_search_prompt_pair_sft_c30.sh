#!/bin/bash

#SBATCH -J 2_3_prompt_pair_sft
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

mkdir -p "/home/${USER}/projects/small_language_model/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}"

GRID_TEMPLATE="${GRID_TEMPLATE:-configs/student_sft_prompt_pair_grid_template.json}"
BASE_CONFIG="${BASE_CONFIG:-configs/student_sft_template.json}"
WORK_DIR="${WORK_DIR:-results/student_sft_prompt_pair}"
GRID_NAME="${GRID_NAME:-qwen_student_sft_prompt_pair}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/student_sft_prompt_pair}"
LEGACY_MANIFEST="${LEGACY_MANIFEST:-results/student_sft_prompt_pair_r4/manifest.json}"
GPU_IDS="${GPU_IDS:-1,2,3}"
RUNTIME_GRID_CONFIG="${WORK_DIR}/configs/student_sft_prompt_pair_grid_runtime.json"
DRY_RUN_LOG="${WORK_DIR}/dry_run_commands.txt"

mkdir -p "${WORK_DIR}/configs" "${WORK_DIR}/logs"

echo "Running on $(hostname)"
which python3
python3 -V
nvidia-smi

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

ranks = config.get("grid", {}).get("student.lora.r", [])
if sorted(ranks) == [4]:
    raise SystemExit(f"{template_path} only contains rank 4: {ranks}")

epochs = config.get("grid", {}).get("training.num_train_epochs", [])
if not epochs:
    raise SystemExit(f"{template_path} does not define training.num_train_epochs")

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

print(f"wrote_runtime_grid_config={output_path}")
print(f"grid_name={grid_name}")
print(f"checkpoint_root={checkpoint_root}")
print(f"ranks={ranks}")
print(f"epochs={epochs}")
PY

python3 -u scripts/2_2_grid_search_student_sft.py \
    --base-config "${BASE_CONFIG}" \
    --grid-config "${RUNTIME_GRID_CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --continue-epochs-from-previous \
    --dry-run > "${DRY_RUN_LOG}"

python3 - "${WORK_DIR}/manifest.json" "${LEGACY_MANIFEST}" <<'PY'
import json
import os
import sys
from pathlib import Path

new_manifest_path = Path(sys.argv[1])
legacy_manifest_path = Path(sys.argv[2])

def read_manifest(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("runs", [])

def complete(output_dir):
    output_dir = Path(output_dir)
    return (output_dir / "adapter_config.json").is_file() and (output_dir / "adapter_model.safetensors").is_file()

def override_key(entry):
    return tuple(sorted(entry.get("overrides", {}).items()))

new_runs = read_manifest(new_manifest_path)
legacy_by_key = {override_key(entry): entry for entry in read_manifest(legacy_manifest_path)}

linked = 0
already_complete = 0
for entry in new_runs:
    overrides = entry.get("overrides", {})
    if overrides.get("training.num_train_epochs") != 1:
        continue

    output_dir = Path(entry["output_dir"])
    if complete(output_dir):
        already_complete += 1
        continue
    if output_dir.exists():
        continue

    legacy_entry = legacy_by_key.get(override_key(entry))
    if not legacy_entry or not complete(legacy_entry["output_dir"]):
        continue

    legacy_dir = Path(legacy_entry["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(legacy_dir, output_dir.parent)
    os.symlink(target, output_dir)
    linked += 1
    print(f"linked_epoch1_reuse new={output_dir} old={legacy_dir}")

print(f"epoch1_reuse_links={linked}")
print(f"epoch1_already_complete={already_complete}")
PY

count_gpu_ids() {
    local ids="${1}"
    local count=0
    local gpu_id

    IFS=',' read -ra gpu_ids_array <<< "${ids}"
    for gpu_id in "${gpu_ids_array[@]}"; do
        if [ -n "${gpu_id}" ]; then
            count=$((count + 1))
        fi
    done

    echo "${count}"
}

wait_for_selected_gpus_free() {
    local threshold_mib="${1:-40000}"
    local required_count
    required_count=$(count_gpu_ids "${GPU_IDS}")
    if [ "${required_count}" -le 0 ]; then
        echo "GPU_IDS must contain at least one GPU id."
        exit 1
    fi

    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=index,memory.free \
            --format=csv,noheader,nounits \
            | awk -F',' -v ids="${GPU_IDS}" -v t="${threshold_mib}" '
                BEGIN {
                    split(ids, wanted, ",")
                    for (i in wanted) {
                        gsub(/^[ \t]+|[ \t]+$/, "", wanted[i])
                        if (wanted[i] != "") {
                            target[wanted[i]] = 1
                        }
                    }
                }
                {
                    gsub(/^[ \t]+|[ \t]+$/, "", $1)
                    gsub(/^[ \t]+|[ \t]+$/, "", $2)
                    if (($1 in target) && ($2 + 0) >= t) {
                        c++
                    }
                }
                END {print c+0}
            ')
        if [ "${free_count}" -ge "${required_count}" ]; then
            echo "Selected GPUs (${GPU_IDS}) have at least ${threshold_mib} MiB free."
            nvidia-smi
            break
        fi
        echo "Waiting for selected GPUs (${GPU_IDS}): ${free_count}/${required_count} above ${threshold_mib} MiB at $(date)"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

wait_for_selected_gpus_free 40000

systemd-run --user --scope -p MemoryMax=32G python3 -u scripts/2_2_grid_search_student_sft.py \
    --base-config "${BASE_CONFIG}" \
    --grid-config "${RUNTIME_GRID_CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --gpu-ids "${GPU_IDS}" \
    --continue-epochs-from-previous \
    --skip-existing
