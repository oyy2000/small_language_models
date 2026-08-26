#!/bin/bash

#SBATCH -J 13_5_multiteacher_kd
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-sft}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-multiteacher-kd/${SLURM_JOB_ID:-manual}}"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LOCAL_RUNTIME_ROOT}/checkpoints"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_CHECKPOINT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
PROTOCOL="${PROTOCOL:-${OUTPUT_ROOT}/pilot/frozen_kd_protocol.json}"
TEACHER_NAMES="${TEACHER_NAMES:?TEACHER_NAMES is required}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
python3 -u scripts/13_5_launch_multiteacher_logit_kd.py \
  --protocol "${PROTOCOL}" \
  --teacher-names "${TEACHER_NAMES}" \
  --gpu-ids "${GPU_ID_CSV}" \
  --skip-complete
