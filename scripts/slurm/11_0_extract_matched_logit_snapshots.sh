#!/bin/bash

#SBATCH -J 11_0_matched_logits
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
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
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-matched-logits/${SLURM_JOB_ID:-manual}}"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_LOGIT_ROOT="${LOCAL_RUNTIME_ROOT}/logits"
mkdir -p "${TMPDIR}" "${LBD_RUNTIME_LOGIT_ROOT}"

nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB:-28000}" "${MIN_IDLE_GPUS:-1}" "${WAIT_SECONDS:-60}" \
  "${MAX_USED_MIB:-500}" "${MAX_UTILIZATION:-10}" "${STABLE_CHECKS:-2}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
python3 -u scripts/11_0_launch_matched_logit_snapshots.py \
  --config "${CONFIG:-configs/capacity_length_logit_kd_seed17_v1.json}" \
  --gpu-ids "${GPU_ID_CSV}" \
  --launcher-shards "${LAUNCHER_SHARDS:?LAUNCHER_SHARDS is required}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}" \
  --skip-complete
