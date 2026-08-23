# Capacity-Length Logit-Distillation Protocol

## Objective

This experiment compares the registered seed-17 sequence-level SFT adapters with exact online
logit distillation from `Qwen/Qwen2.5-7B-Instruct` into the fixed
`Qwen/Qwen2.5-1.5B-Instruct` LoRA student. It covers the existing short-128, medium-256, and
long-512 verified trajectories on GSM8K only.

The parent seed-17 artifacts remain immutable. New outputs use
`capacity_length_logit_kd_seed17_v1` result and checkpoint roots.

## Registered Method

- Training data: the 881-problem common intersection for each selected 7B length condition.
- Loss: completion-only hard-label cross-entropy plus temperature-scaled forward
  `teacher -> student` KL.
- Shared hyperparameter grid: KL weight `{0.25, 0.5, 0.75}` and temperature `{1, 2, 4}`.
- Selection split: GSM8K `train[2000:2500]`; one parameter pair is shared across all budgets.
- Formal comparison split: GSM8K `test[50:1319]`, greedy decoding, 512 generated-token cap.
- Student: seed 17, one epoch, LoRA rank 4/alpha 16, learning rate `2e-5`, effective batch size 4.
- Vocabulary alignment: both models are normalized over the 151,665 valid shared tokenizer IDs;
  model-head padding mass is recorded separately.

The official test cohort has already been observed by the parent experiment. This experiment is a
locked comparative rerun, not a fresh untouched confirmatory test. Formal results are never used to
retune this protocol; a later change requires a separately registered version.

## Logit Evidence

For teacher, base student, SFT student, and KD student, top-64 raw logits are saved at every
completion position under teacher forcing on the same verified 7B trajectory. Each snapshot also
contains log-sum-exp, entropy, target-token logit/rank, and invalid-vocabulary mass. Student
snapshots include exact full-valid-vocabulary teacher-to-student KL and Jensen-Shannon distance.

Snapshots are split into eight safetensors shards per budget and method. JSON metadata binds record
IDs and offsets to the source JSONL. Per-shard markers register data, config, model, tokenizer,
source-code, tensor, and metadata hashes.

## Execution and Completion

The complete dependency graph is exposed by:

```bash
DRY_RUN=1 bash scripts/9_0_submit_logit_kd_experiment.sh
bash scripts/9_0_submit_logit_kd_experiment.sh
```

The pipeline performs parent preflight, a lower-memory RTX 5000 Ada smoke test, validation sweep,
selection, formal training, formal evaluation, matched-logit extraction, analysis, and completion
audit. GPU wrappers use the shared stable-idle-GPU gate and never terminate unrelated processes.

The experiment is complete only when the root `FORMAL_COMPLETE` marker is bound to a passing
completion audit. A favorable result is not a completion requirement. Improvement is classified
only when accuracy rises and budget compliance does not decline in all three conditions; paired
confidence intervals and Holm-adjusted McNemar tests are reported separately.
