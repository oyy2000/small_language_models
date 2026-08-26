# 7B Teacher-Prompt Logit-KD Protocol

## Objective and evidence level

This experiment isolates the prompt context used for online logit distillation.
The Qwen2.5-7B-Instruct teacher receives the registered 128-, 256-, or
512-token `teacher_prompt`, while the Qwen2.5-1.5B-Instruct student receives the
ordinary budget-blind `prompt`. Both models are teacher-forced over the same
verified completion, and exact forward KL is aligned only over the identical
completion target tokens.

The experiment is a single-seed adaptive GSM8K diagnostic. The locked
`test[50:1319]` cohort has already been observed and is not new confirmatory
evidence. Existing same-prompt KD, factorial SFT, multi-teacher, and paired-
rewrite artifacts remain immutable.

## Registered controls

- Three independent LoRA adapters represent short-128, medium-256, and
  long-512; the student prompt does not expose the budget.
- Training reuses the completed 7B equal-token subsets: 881/381/179 records and
  31,443/31,442/31,407 verified solution tokens.
- Student, LoRA, seed 17, one epoch, optimizer, decoding, and data hashes match
  the completed equal-token KD experiment.
- The validation grid is alpha 0.25/0.5/0.75 by temperature 1/2/4, shared across
  all three budgets and selected only on GSM8K train `[2000:2500]`.
- A candidate is feasible only if every budget's compliance is no more than
  0.02 below its matched SFT baseline. If no candidate is feasible,
  `VALIDATION_BLOCKED` is written and the formal dependency chain stops.

## Artifact and analysis contract

Preflight verifies all registered hashes, the correct budget marker in every
teacher context, the absence of a budget marker in every student context, zero
teacher/student completion-target mismatches, and both sequence-length bounds.
Training and matched-logit manifests record both context fields and prompt
lengths.

Primary analysis compares teacher-prompt KD with matched SFT. Secondary analysis
compares it with the completed same-prompt KD predictions on identical problem
support. Each comparison reports accuracy, strict budget compliance, generated
length, paired bootstrap intervals, exact McNemar tests, and Holm correction
over the three budgets. PNG and PDF figures show accuracy, compliance, the
length-accuracy frontier, and matched teacher-student KL.

Technical completion requires all 27 validation adapters/evaluations, one
feasible frozen selection, three diagnostic adapters with 1,269 predictions
each, complete matched-logit shards, analysis artifacts, a passing independent
audit, and the root `FORMAL_COMPLETE` marker. A scientifically negative result
can still be technically complete.

## Execution

Preview the complete dependency graph:

```bash
DRY_RUN=1 KD_NODES=c32,c31 KD_PARTITIONS=a5000ada,a6000 \
  bash scripts/15_0_submit_teacher_prompt_logit_kd.sh
```

After inspecting physical GPU occupancy, submit with the same command without
`DRY_RUN=1`. Every GPU stage uses the shared stable-idle-GPU gate, one process
per selected GPU, node-local runtime directories, and hash-verified publication
to BeeGFS.
