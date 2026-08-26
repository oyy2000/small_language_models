#!/bin/bash

#SBATCH -J 14_4_paired_rewrite_train
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
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-paired-rewrite/${SLURM_JOB_ID:-manual}}"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
export LBD_RUNTIME_CHECKPOINT_ROOT="${LOCAL_RUNTIME_ROOT}/checkpoints"

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_paired_rewrite_7b_pilot_v1}"
TRAINING_STAGE="${TRAINING_STAGE:-grid}"
SELECTION_JSON="${SELECTION_JSON:-${OUTPUT_ROOT}/pilot/analysis/selection/recipe_selection.json}"
WORK_DIR="${OUTPUT_ROOT}/pilot/training/${TRAINING_STAGE}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

mkdir_with_retry "${WORK_DIR}"
mkdir_with_retry "${CHECKPOINT_ROOT}"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}" "${LBD_RUNTIME_CHECKPOINT_ROOT}"
prepare_args=(
  --config "${CONFIG}"
  --dataset-manifest "${OUTPUT_ROOT}/pilot/sft_data/dataset_manifest.json"
  --output-dir "${WORK_DIR}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --stage "${TRAINING_STAGE}"
)
if [ "${TRAINING_STAGE}" = "final" ]; then
  prepare_args+=(--selection-json "${SELECTION_JSON}")
fi
prepared_manifest="${WORK_DIR}/${TRAINING_STAGE}_training_manifest.json"
if [ -f "${prepared_manifest}" ]; then
  echo "Reusing prepared training manifest: ${prepared_manifest}"
else
  python3 -u scripts/14_3_prepare_paired_rewrite_training.py "${prepare_args[@]}"
fi

nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
python3 -u scripts/14_4_launch_paired_rewrite_training.py \
  --prepared-manifest "${prepared_manifest}" \
  --output-manifest "${WORK_DIR}/training_launch_manifest.json" \
  --log-dir "${WORK_DIR}/logs" \
  --gpu-ids "${GPU_ID_CSV}" \
  --skip-complete
