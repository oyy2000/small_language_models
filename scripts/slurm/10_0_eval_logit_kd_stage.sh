#!/bin/bash

#SBATCH -J 10_0_logit_kd_eval
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
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-logit-kd-eval/${SLURM_JOB_ID:-manual}}"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_EVAL_ROOT="${LOCAL_RUNTIME_ROOT}/eval"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_EVAL_ROOT}"

nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB:-12000}" "${MIN_IDLE_GPUS:-1}" "${WAIT_SECONDS:-60}" \
  "${MAX_USED_MIB:-500}" "${MAX_UTILIZATION:-10}" "${STABLE_CHECKS:-2}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
python3 -u scripts/10_0_launch_logit_kd_evaluations.py \
  --config "${CONFIG:-configs/capacity_length_logit_kd_seed17_v1.json}" \
  --stage "${STAGE:?STAGE is required}" \
  --gpu-ids "${GPU_ID_CSV}" \
  --launcher-shards "${LAUNCHER_SHARDS:?LAUNCHER_SHARDS is required}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}" \
  --skip-complete
