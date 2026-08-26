#!/bin/bash

#SBATCH -J 14_5_paired_rewrite_eval
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${EVAL_ENV:-sft}"
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-paired-rewrite-eval/${SLURM_JOB_ID:-manual}}"
export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"

CONFIG="${CONFIG:-configs/capacity_length_paired_rewrite_7b_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_paired_rewrite_7b_pilot_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/capacity_length_paired_rewrite_7b_pilot_v1}"
EVAL_STAGE="${EVAL_STAGE:-selection}"
SELECTION_JSON="${SELECTION_JSON:-${OUTPUT_ROOT}/pilot/analysis/selection/recipe_selection.json}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MIN_IDLE_GPUS="${MIN_IDLE_GPUS:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"

mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
nvidia-smi
GPU_IDS=()
select_stably_idle_gpus \
  "${MIN_FREE_MIB}" "${MIN_IDLE_GPUS}" "${WAIT_SECONDS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}" "${STABLE_CHECKS}"
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")

if [ "${EVAL_STAGE}" = "selection" ]; then
  manifest_glob="${CHECKPOINT_ROOT}/grid/*/training_manifest.json"
  output_dir="${OUTPUT_ROOT}/pilot/eval/selection"
  extra_args=()
elif [ "${EVAL_STAGE}" = "confirmatory" ]; then
  manifest_glob="${CHECKPOINT_ROOT}/*/*/training_manifest.json"
  output_dir="${OUTPUT_ROOT}/pilot/eval/confirmatory"
  extra_args=(--selection-json "${SELECTION_JSON}")
else
  echo "Unsupported EVAL_STAGE=${EVAL_STAGE}" >&2
  exit 2
fi

python3 -u scripts/14_5_launch_paired_rewrite_evaluations.py \
  --config "${CONFIG}" \
  --stage "${EVAL_STAGE}" \
  --training-manifest-glob "${manifest_glob}" \
  --output-dir "${output_dir}" \
  --gpu-ids "${GPU_ID_CSV}" \
  --skip-complete \
  "${extra_args[@]}"
