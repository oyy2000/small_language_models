#!/bin/bash
# Submit the registered 7B teacher-prompt versus student-prompt KD ablation.

set -euo pipefail
cd "$(dirname "$0")/.."

export CONFIG="${CONFIG:-configs/capacity_length_logit_kd_teacher_prompt_equal_token_seed17_v1.json}"
export KD_NODES="${KD_NODES:-c32,c31}"
export KD_PARTITIONS="${KD_PARTITIONS:-a5000ada,a6000}"
exec bash scripts/9_0_submit_logit_kd_experiment.sh
