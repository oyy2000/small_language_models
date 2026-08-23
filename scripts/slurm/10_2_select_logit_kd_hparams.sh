#!/bin/bash

#SBATCH -J 10_2_logit_kd_select
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --oversubscribe
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/projects/small_language_model/logs/%x-%j.out
#SBATCH --error=/home/%u/projects/small_language_model/logs/%x-%j.err

set -euo pipefail
if [ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
  source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate "${SFT_ENV:-sft}"
cd /home/youyang7/projects/small_language_model
python3 -u scripts/10_2_select_logit_kd_hparams.py \
  --config "${CONFIG:-configs/capacity_length_logit_kd_seed17_v1.json}"
