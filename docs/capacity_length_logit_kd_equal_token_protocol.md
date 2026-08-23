# Capacity-Length Equal-Token Logit-Distillation Protocol

## Objective

This experiment is the equal-supervision-token robustness counterpart to
`capacity_length_logit_kd_seed17_v1`. It trains exact online logit-KD students on the registered
seed-17 Qwen2.5-7B trajectories for short-128, medium-256, and long-512 conditions while holding
the total verified solution-token count approximately constant.

The parent factorial artifacts and the completed equal-example logit-KD artifacts remain
immutable. New outputs use the independent
`capacity_length_logit_kd_equal_token_seed17_v1` result and checkpoint roots.

## Registered Supervision

- Short-128: 881 records and 31,443 verified solution tokens.
- Medium-256: 381 records and 31,442 verified solution tokens.
- Long-512: 179 records and 31,407 verified solution tokens.
- Maximum registered cross-budget difference: 36 solution tokens.
- Subset construction seed: 17.

The token count is `sum(metadata.solution_token_count)` over each immutable JSONL input. Records
are not split, so the three totals are near-equal rather than exactly identical. The number of
examples and optimizer steps remains different across budgets; this protocol is an equal-total-
solution-token comparison, not an equal-step comparison.

## Method and Selection

All model revisions, LoRA settings, validation and formal cohorts, KL-weight and temperature
grid, selection rule, decoding settings, matched-logit evidence, and completion gates are inherited
from `docs/capacity_length_logit_kd_protocol.md`. Hyperparameters are selected again on GSM8K
`train[2000:2500]` using only the equal-token KD adapters, with one shared pair across all three
budgets. Formal evaluation remains the locked GSM8K `test[50:1319]` comparative cohort and does
not estimate training-seed variability.

## Execution

```bash
CONFIG=configs/capacity_length_logit_kd_equal_token_seed17_v1.json \
  DRY_RUN=1 bash scripts/9_0_submit_logit_kd_experiment.sh

CONFIG=configs/capacity_length_logit_kd_equal_token_seed17_v1.json \
  bash scripts/9_0_submit_logit_kd_experiment.sh
```

When only a subset of the registered nodes has stably idle physical GPUs, the same dependency
graph can be restricted without changing experiment semantics. For example:

```bash
CONFIG=configs/capacity_length_logit_kd_equal_token_seed17_v1.json \
  KD_NODES=c32 KD_PARTITIONS=a5000ada \
  bash scripts/9_0_submit_logit_kd_experiment.sh
```

The experiment is complete only when its root `FORMAL_COMPLETE` marker is bound to a passing
completion audit. Partial adapters, queued jobs, and completed validation sweeps are not formal
results.
