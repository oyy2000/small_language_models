#!/bin/bash

#SBATCH -J 1_4_prompt_pair
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
conda activate fact

cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/home/${USER}/projects/.cache/hf_datasets/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "/home/${USER}/projects/small_language_model/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Running on $(hostname)"
which python3
python3 -V
nvidia-smi

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

GPU_IDS=0,1,2,3 LOG_EVERY=1 bash scripts/1_4_run_prompt_strategy_pair.sh
