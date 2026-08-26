#!/bin/bash
# Evaluate, analyze, and audit ranked-length adapters inside the C49 allocation.

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
conda activate /mnt/beegfs/youyang7/.conda/envs/sft
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

CONFIG="${CONFIG:-configs/capacity_length_ranked_sampling_7b_eval_seed17_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_ranked_sampling_7b_v1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
RUN_TAG="${RUN_TAG:-c49_eval_${SLURM_JOB_ID}_${SLURM_STEP_ID:-step}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-length-eval/${RUN_TAG}}"

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
mkdir_with_retry "${OUTPUT_ROOT}/formal/eval"
mkdir_with_retry "${FIGURE_ROOT}/formal"

if [ ! -f "${OUTPUT_ROOT}/formal/training/audit/TRAINING_COMPLETE" ]; then
  echo "Ranked-length training is not complete: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [ -e "${OUTPUT_ROOT}/FORMAL_COMPLETE" ]; then
  echo "Ranked-length experiment is already complete: ${OUTPUT_ROOT}/FORMAL_COMPLETE" >&2
  exit 2
fi

echo "host=$(hostname) run_tag=${RUN_TAG} min_free_mib=${MIN_FREE_MIB}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_stably_memory_fit_gpus "${MIN_FREE_MIB}" 3 "${WAIT_SECONDS}" "${STABLE_CHECKS}"
GPU_IDS=("${GPU_IDS[@]:0:3}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
echo "evaluation_gpu_ids=${GPU_ID_CSV} runtime_root=${LOCAL_RUNTIME_ROOT}"

python3 -u scripts/16_6_eval_ranked_length_students.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/formal/eval" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel 3 \
  --skip-complete

python3 -u scripts/16_7_analyze_ranked_length_evaluation.py \
  --config "${CONFIG}" \
  --eval-manifest "${OUTPUT_ROOT}/formal/eval/eval_manifest_formal_shard_00_of_01.json" \
  --output-dir "${OUTPUT_ROOT}/formal/analysis" \
  --figure-dir "${FIGURE_ROOT}/formal"

python3 -u scripts/16_8_audit_ranked_length_experiment.py --config "${CONFIG}"
echo "ranked_length_eval_analysis_audit_complete output=${OUTPUT_ROOT}"
