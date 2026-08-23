#!/bin/bash

#SBATCH -J 11_5_math_mix_eval
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${EVAL_ENV:-fact}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-math-mix-eval/${SLURM_JOB_ID:-manual}}"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

CONFIG="${CONFIG:-configs/capacity_length_math_mix_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_math_mix_pilot_v1}"
LAUNCHER_SHARDS="${LAUNCHER_SHARDS:-4}"
LAUNCHER_SHARD_INDEX="${LAUNCHER_SHARD_INDEX:?LAUNCHER_SHARD_INDEX is required}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

GPU_IDS=()
nvidia-smi
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

python3 -u scripts/11_5_launch_math_mix_evaluations.py \
  --config "${CONFIG}" \
  --eval-suite-manifest "${OUTPUT_ROOT}/pilot/eval_suite/eval_suite_manifest.json" \
  --reference-training-manifest-glob \
    "results/capacity_length_factorial_seed17_v1/formal/training/training_manifest_shard_*.json" \
  --pilot-training-manifest-glob "${OUTPUT_ROOT}/pilot/training/training_manifest_shard_*.json" \
  --output-dir "${OUTPUT_ROOT}/pilot/eval" \
  --gpu-ids "${GPU_ID_CSV}" \
  --launcher-shards "${LAUNCHER_SHARDS}" \
  --launcher-shard-index "${LAUNCHER_SHARD_INDEX}" \
  --skip-complete

echo "math_mix_eval_shard_complete shard=${LAUNCHER_SHARD_INDEX}/${LAUNCHER_SHARDS}"
