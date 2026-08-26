#!/bin/bash

#SBATCH -J 13_6_model_registry
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate "${PREP_ENV:-fact}"
cd /home/youyang7/projects/small_language_model

OUTPUT_ROOT="${OUTPUT_ROOT:-results/capacity_length_multibench_multiteacher_kd_pilot_v1}"
python3 -u scripts/13_6_prepare_multiteacher_model_registry.py \
  --protocol "${OUTPUT_ROOT}/pilot/frozen_kd_protocol.json" \
  --sft-manifest-glob "${OUTPUT_ROOT}/pilot/training/training_manifest_shard_*_of_*.json" \
  --output-json "${OUTPUT_ROOT}/pilot/model_registry.json"
