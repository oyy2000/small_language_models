#!/bin/bash

#SBATCH -J 13_2_multiteacher_build
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${BUILD_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/mnt/beegfs/youyang7/.cache/huggingface}"
CONFIG="${CONFIG:-configs/capacity_length_multibench_multiteacher_kd_pilot_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"

for generator_name in qwen2p5_1p5b qwen2p5_3b qwen2p5_14b; do
  python3 -u scripts/5_2_merge_select_capacity_length_traces.py \
    --config "${CONFIG}" \
    --generator-name "${generator_name}" \
    --input-glob "${OUTPUT_ROOT}/pilot/raw/${generator_name}/shard_*.jsonl" \
    --output-dir "${OUTPUT_ROOT}/pilot/selected/${generator_name}" \
    --stage pilot \
    --expected-problems 1000
done

python3 -u scripts/13_2_build_multiteacher_multibench_kd_data.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}/pilot/sft_data"

echo "multiteacher_multibench_build_complete output=${OUTPUT_ROOT}/pilot/sft_data"
