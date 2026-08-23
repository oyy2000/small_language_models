#!/bin/bash

#SBATCH -J 9_3_logit_kd_smoke
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
if [ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
  source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${SFT_ENV:-sft}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-logit-kd-smoke/${SLURM_JOB_ID:-manual}}"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_LOGIT_ROOT="${LOCAL_RUNTIME_ROOT}/logits"
mkdir -p "${TMPDIR}" "${LBD_RUNTIME_LOGIT_ROOT}"
nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB:-28000}" "${MIN_IDLE_GPUS:-1}" "${WAIT_SECONDS:-60}" \
  "${MAX_USED_MIB:-500}" "${MAX_UTILIZATION:-10}" "${STABLE_CHECKS:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
python3 -u scripts/9_3_smoke_logit_kd_gpu.py \
  --config "${CONFIG:-configs/capacity_length_logit_kd_seed17_v1.json}" \
  --label "${SMOKE_LABEL:-default}"
