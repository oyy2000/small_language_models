# Multi-Benchmark, Multi-Teacher KD Pilot Protocol

## Status and objective

This is an exploratory, single-seed extension of the completed seed-17
capacity-by-length SFT experiment, the completed 7B MATH-mix pilot, and the
equal-token 7B logit-KD experiment. It tests matched-teacher online logit
distillation across multiple training benchmarks, teacher capacities, and
trajectory lengths. It does not modify or supersede any frozen parent artifact.

Unlike the original black-box sequence-level response-distillation pipeline,
the KD arm in this pilot uses exact teacher-to-student forward KL on the shared
valid vocabulary plus hard-target cross-entropy. Every condition also has a
matched hard-target SFT baseline trained on the same mixed JSONL input.

## Reused registered settings

- Student: Qwen2.5-1.5B-Instruct, rank-4 LoRA, alpha 16, dropout 0.05,
  all-linear target modules.
- Training: seed 17, one epoch, learning rate `2e-5`, batch size 1,
  gradient accumulation 4, warmup ratio 0.03, max length 2,048, and
  completion-only loss.
- Teachers: Qwen2.5-1.5B/3B/7B/14B-Instruct at the snapshots used by the
  registered GSM8K factorial. The 1.5B teacher is a self-distillation control.
- Trajectory lengths: 128, 256, and 512 solution tokens.
- Generation: three candidates, temperature 0.7, top-p 0.95, and deterministic
  identity-derived seeds.
- GSM8K contribution: the completed seed-17 equal-token datasets, approximately
  31,443 verified solution tokens in every teacher-length condition.
- MATH source: the same deterministic 1,000-problem MATH-train cohort used by
  `capacity_length_math_mix_pilot_v1`.
- Evaluation: the frozen GSM8K-200, MATH-500-100, and AIME-2025-30 suite from
  the completed MATH-mix pilot.

The 7B MATH trajectories are reused by hash. Only the missing 1.5B, 3B, and
14B trajectories are generated under the new experiment root.

## Support and supervision controls

Correct, length-compliant candidates are selected independently for each
teacher-length condition. Common problem support is intersected across the
three lengths within each teacher. This preserves a paired length comparison
within a teacher while avoiding a potentially severe four-teacher global
intersection. Teacher-capacity comparisons therefore do not have identical
problem support and remain descriptive.

For MATH, the smallest verified solution-token total among all 12 conditions
defines one global target. Whole trajectories are deterministically subsampled
without exceeding that target. The resulting MATH data are combined with the
already globally balanced GSM8K equal-token data. Source counts, source token
totals, hashes, and teacher identities are recorded for every condition.

## KD hyperparameters and comparison

The pilot does not search KD hyperparameters on MATH-500 or AIME. Alpha and
temperature are inherited from the completed
`capacity_length_logit_kd_equal_token_seed17_v1` validation selection, which
uses GSM8K train `[2000:2500]`. The inherited selection and completion audit are
hash-bound into `pilot/frozen_kd_protocol.json` before training begins.

The planned model registry contains 25 models:

- one base Qwen2.5-1.5B-Instruct model;
- 12 matched hard-target SFT adapters;
- 12 matched-teacher logit-KD adapters.

All 25 models are evaluated on all three frozen datasets, producing 75
prediction artifacts. The analysis reports Wilson intervals, paired bootstrap
intervals for KD-minus-SFT accuracy, exact McNemar tests, and Holm correction
over the 12 teacher-length comparisons within each dataset.

## Artifact contract

Large artifacts use the stable BeeGFS-backed path
`results/capacity_length_multibench_multiteacher_kd_pilot_v1`. Adapters are
stored under `checkpoints/capacity_length_multibench_multiteacher_kd_pilot_v1`,
and publication figures under
`figures/capacity_length_multibench_multiteacher_kd_pilot_v1`.

Completion requires:

- complete raw shard matrices and passed per-teacher selection audits for the
  three newly generated teachers;
- 12 hash-verified mixed equal-token datasets;
- a frozen KD protocol bound to the inherited selection and completion audit;
- 12 hash-verified SFT adapters and 12 hash-verified logit-KD adapters;
- 25 model manifests and 75 prediction/summary pairs;
- CSV, JSON, report, PNG, and PDF analysis artifacts;
- a passed independent completion audit and root `PILOT_COMPLETE` marker.

Queued jobs, partial shards, and unverified adapters are not completed results.

## Main commands

Preview and submit missing-teacher trajectory generation plus mixed-data build:

```bash
DRY_RUN=1 UPSTREAM_KD_AUDIT_JOB=<equal-token-kd-audit-job> \
  bash scripts/13_0_submit_multiteacher_trace_build.sh

UPSTREAM_KD_AUDIT_JOB=<equal-token-kd-audit-job> \
  bash scripts/13_0_submit_multiteacher_trace_build.sh
```

Submit matched training and evaluation after the build job is known:

```bash
BUILD_JOB_ID=<build-job> \
  bash scripts/13_0_submit_multiteacher_kd_train_eval.sh
```

Submit analysis and the completion audit after both evaluation launchers:

```bash
EVAL_JOB_IDS=<eval-job-0>:<eval-job-1> \
  bash scripts/13_0_submit_multiteacher_analysis_audit.sh
```

Every GPU launcher inspects physical occupancy and waits for stable idle GPUs.
It never treats an exclusive allocation or null Slurm GRES field as proof of
GPU isolation.

## Interpretation boundary

- The experiment uses one training seed and does not estimate training-seed
  variability.
- Common support differs across teachers.
- GSM8K official test results were previously observed, so GSM8K-200 is an
  adaptive diagnostic.
- MATH-500-100 and AIME-2025-30 remain small exploratory cohorts; AIME may have
  a floor effect.
- The inherited KD hyperparameters were selected for the 7B GSM8K condition,
  so transfer to other teachers and benchmarks is itself part of the test.
