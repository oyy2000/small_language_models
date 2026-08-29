#!/bin/bash

#SBATCH -J 19_0_submit_ranked_matrix
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --oversubscribe
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

# Submit Phase C only after the Phase-A eval/analysis/audit job succeeds.

set -euo pipefail
cd /home/youyang7/projects/small_language_model

if [ ! -f results/capacity_length_ranked_sampling_7b_multiseed_v1/MULTISEED_COMPLETE ]; then
  echo "Phase-A dependency completed without MULTISEED_COMPLETE." >&2
  exit 2
fi

bash scripts/19_0_submit_ranked_multiteacher_matrix.sh
