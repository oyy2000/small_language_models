#!/bin/bash

#SBATCH -J 18_5_ranked_multiseed_eval
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
source /home/youyang7/projects/small_language_model/scripts/slurm/_gpu_idle_gate.sh
conda activate "${SFT_ENV:-/mnt/beegfs/youyang7/.conda/envs/sft}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-4}"
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"

TEMPLATE_CONFIG="${TEMPLATE_CONFIG:-configs/capacity_length_ranked_sampling_7b_eval_multiseed_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_ranked_sampling_7b_multiseed_v1}"
FIGURE_ROOT="${FIGURE_ROOT:-figures/capacity_length_ranked_sampling_7b_multiseed_v1}"
FROZEN_CONFIG="${FROZEN_CONFIG:-${OUTPUT_ROOT}/formal/protocol/frozen_eval_config.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
MIN_GPUS="${MIN_GPUS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
MAX_USED_MIB="${MAX_USED_MIB:-500}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"
RUN_TAG="${RUN_TAG:-ranked_multiseed_eval_${SLURM_JOB_ID:-manual}}"
LOCAL_RUNTIME_ROOT="${LOCAL_RUNTIME_ROOT:-/var/tmp/${USER}-ranked-multiseed-eval/${RUN_TAG}}"

export HF_DATASETS_CACHE="${LOCAL_RUNTIME_ROOT}/hf_datasets"
export TMPDIR="${LOCAL_RUNTIME_ROOT}/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
mkdir_with_retry "${OUTPUT_ROOT}/formal/protocol"
mkdir_with_retry "${OUTPUT_ROOT}/formal/eval"
mkdir_with_retry "${FIGURE_ROOT}/formal"

if [ ! -f "${OUTPUT_ROOT}/formal/training/seed42_73/audit/TRAINING_COMPLETE" ]; then
  echo "Seed-42/73 training is not complete: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [ -e "${OUTPUT_ROOT}/MULTISEED_COMPLETE" ]; then
  echo "Multi-seed experiment is already sealed: ${OUTPUT_ROOT}/MULTISEED_COMPLETE" >&2
  exit 2
fi

if [ ! -f "${FROZEN_CONFIG}" ]; then
  python3 -u scripts/18_4_freeze_ranked_multiseed_eval.py \
    --template-config "${TEMPLATE_CONFIG}" \
    --output-config "${FROZEN_CONFIG}"
else
  echo "reusing_frozen_eval_config=${FROZEN_CONFIG}"
fi

echo "host=$(hostname) run_tag=${RUN_TAG} min_free_mib=${MIN_FREE_MIB} min_gpus=${MIN_GPUS}"
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
GPU_IDS=()
select_gpus_for_approved_node \
  "${MIN_FREE_MIB}" "${MIN_GPUS}" "${WAIT_SECONDS}" "${STABLE_CHECKS}" \
  "${MAX_USED_MIB}" "${MAX_UTILIZATION}"
GPU_IDS=("${GPU_IDS[@]:0:${MIN_GPUS}}")
GPU_ID_CSV=$(IFS=,; echo "${GPU_IDS[*]}")
echo "evaluation_gpu_ids=${GPU_ID_CSV} runtime_root=${LOCAL_RUNTIME_ROOT}"

python3 -u scripts/18_5_eval_ranked_multiseed_students.py \
  --config "${FROZEN_CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/formal/eval" \
  --gpu-ids "${GPU_ID_CSV}" \
  --max-parallel "${MIN_GPUS}" \
  --skip-complete

conda activate "${ANALYSIS_ENV:-/mnt/beegfs/youyang7/.conda/envs/fact}"
python3 -u scripts/18_6_analyze_ranked_multiseed.py \
  --config "${FROZEN_CONFIG}" \
  --eval-manifest "${OUTPUT_ROOT}/formal/eval/eval_manifest_formal_shard_00_of_01.json" \
  --output-dir "${OUTPUT_ROOT}/formal/analysis" \
  --figure-dir "${FIGURE_ROOT}/formal"

python3 -u scripts/18_7_audit_ranked_multiseed_experiment.py --config "${FROZEN_CONFIG}"
echo "ranked_multiseed_eval_analysis_audit_complete output=${OUTPUT_ROOT}"
