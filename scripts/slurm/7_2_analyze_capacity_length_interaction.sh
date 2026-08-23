#!/bin/bash

#SBATCH -J 7_2_capacity_length_analysis
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${SFT_ENV:-sft}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
STATSMODELS_WHEEL="/mnt/beegfs/youyang7/projects/small_language_model/wheels/statsmodels-0.14.6-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
STATSMODELS_SHA256="06eec42d682fdb09fe5d70a05930857efb141754ec5a5056a03304c1b5e32fd9"
ANALYSIS_DEPS_DIR="/var/tmp/youyang7-small-language-model-statsmodels-0.14.6"
if [ ! -f "${ANALYSIS_DEPS_DIR}/STATS_MODELS_READY" ]; then
  if [ -e "${ANALYSIS_DEPS_DIR}" ]; then
    mv "${ANALYSIS_DEPS_DIR}" "${ANALYSIS_DEPS_DIR}.partial-${SLURM_JOB_ID:-manual}"
  fi
  echo "${STATSMODELS_SHA256}  ${STATSMODELS_WHEEL}" | sha256sum -c -
  mkdir -p "${ANALYSIS_DEPS_DIR}"
  unzip -q "${STATSMODELS_WHEEL}" -d "${ANALYSIS_DEPS_DIR}"
  test -f "${ANALYSIS_DEPS_DIR}/statsmodels/api.py"
  touch "${ANALYSIS_DEPS_DIR}/STATS_MODELS_READY"
fi
export PYTHONPATH="${ANALYSIS_DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python3 -c 'import statsmodels, statsmodels.api, statsmodels.formula.api; print("statsmodels=" + statsmodels.__version__)'
CONFIG="${CONFIG:-configs/capacity_length_factorial_v1.json}"
STAGE="${STAGE:-formal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_factorial_v1}"

python3 -u scripts/7_2_analyze_capacity_length_interaction.py \
  --config "${CONFIG}" \
  --eval-manifest-glob "${OUTPUT_ROOT}/${STAGE}/eval/eval_manifest_${STAGE}_shard_*.json" \
  --dataset-manifest "${OUTPUT_ROOT}/${STAGE}/sft_data/dataset_manifest.json" \
  --selected-traces "${OUTPUT_ROOT}/${STAGE}/selected/selected_traces.jsonl" \
  --selection-audit "${OUTPUT_ROOT}/${STAGE}/selected/selection_audit.json" \
  --output-dir "${OUTPUT_ROOT}/${STAGE}/analysis" \
  --stage "${STAGE}"
