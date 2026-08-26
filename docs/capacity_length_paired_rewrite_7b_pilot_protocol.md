# Capacity-Length Paired Rewrite 7B Pilot

## Status and scope

This is an exploratory, single-seed GSM8K pilot. It is independent of the historical capacity-length factorial artifacts and the phase-13 logit-KD DAG. It must not be reported as a formal multi-seed result.

The pilot tests whether shortening a verified standard 7B rationale by paired, structure-preserving rewriting transfers better than generating an independently short answer from a length-constrained prompt.

## Paired data construction

The immutable source is the 881-example `qwen2p5_7b__long_512` equal-example SFT file from the revised seed-17 protocol. The historical 7B-short file is included only as a negative comparison. Both input paths and SHA256 hashes are fixed in `configs/capacity_length_paired_rewrite_7b_pilot_v1.json`.

For each verified source rationale, the same Qwen2.5-7B-Instruct model receives the problem, gold final answer, complete source solution, measured source length, target length, and arithmetic intermediate values extracted from the source. It produces three candidates at each target ratio:

- `rewrite_80`: at most 80 percent of the source completion tokens;
- `rewrite_65`: at most 65 percent of the source completion tokens.

Each target is an interval, not only an upper bound. A candidate must contain at least 75 percent of its ratio-specific target length. Thus rewrite-80 accepts approximately 60--80 percent of the source length, and rewrite-65 accepts approximately 49--65 percent. This explicitly rejects severe undershooting.

A candidate is eligible only when it:

- ends with an explicit answer line and matches the GSM8K gold answer;
- contains no detected invalid fully numeric equality;
- preserves every required intermediate value extracted by the conservative checker;
- is no longer than the source and is within the requested target.

Selection chooses the eligible candidate closest to the target, not the shortest candidate. `rewrite_65` may fall back to an eligible `rewrite_80` candidate and then to the original standard rationale. `rewrite_80` may fall back to the original. Every decision and fallback is recorded per example. Symbolic equations are skipped by the arithmetic checker instead of being transformed into numeric expressions.

The four SFT datasets use exactly the same 881 problem IDs:

- original standard 7B-long;
- paired rewrite at 80 percent;
- adaptive paired rewrite at 65 percent;
- historical independently generated 7B-short.

## Training correction

All core conditions use the fixed Qwen2.5-1.5B-Instruct student, the existing completion-only LoRA objective, identical example count, identical batch size, and identical optimizer-step count. This removes the equal-token confound in which shorter targets produce a different number of examples or steps.

The core recipe grid is shared across the standard, rewrite-80, and rewrite-65 conditions:

- learning rate: `5e-6`, `1e-5`, or `2e-5`;
- snapshots: 0.5, 1.0, and 2.0 epochs;
- per-device batch size: 16;
- gradient accumulation: 1;
- gradient clipping: 1.0;
- LoRA rank: 4.

Recipe selection uses the fixed GSM8K development slice `train[2000:3000]`. It maximizes macro accuracy across all three core target conditions. Recipes within 0.5 percentage points of the best macro accuracy are resolved by lower observed gradient-clip rate, fewer epochs, and lower learning rate, in that order. Each published snapshot contains the adapter hashes, exact training-data hash, run-config hash, and training log history.

The selection analysis plots both completion-only loss and gradient norm against epoch for every condition and learning rate. Loss is treated as an optimization diagnostic rather than the decision metric, because target length changes the entropy and redundancy of the supervised sequence.

After selection, three additional schedules are trained:

- direct-short from the base student with the selected base recipe;
- rewrite-65 followed by an additional 0.5 epoch of rewrite-65 at half learning rate;
- standard followed by 0.5 epoch of rewrite-65 at half learning rate.

The second schedule is the matched control for the progressive schedule's extra training stage.

## Evaluation and decision rule

Selection evaluation uses greedy decoding with a 512-token ceiling. Confirmatory evaluation uses the disjoint fixed slice `train[3000:7473]` and greedy budgets of 64, 96, 128, 192, 256, and 512 tokens. A four-sample stochastic evaluation is additionally run at 512 tokens.

Each evaluation stores per-example predictions, extracted answers, correctness, exact generated token counts, EOS completion, and actual length termination. The primary comparison is not raw training loss. It is the accuracy-length frontier under controlled inference budgets.

The paired rewrite-65 condition advances only if, relative to the selected standard condition at the 512-token evaluation ceiling:

- accuracy drops by no more than 2.0 percentage points; and
- mean generated length is at most 80 percent of the standard condition.

The same gate is reported separately for the progressive schedule. Passing the pilot gate motivates a formal multi-seed protocol; it does not itself establish a paper claim.

## Execution

The pipeline is intentionally staged so that the selected training recipe is frozen before final training:

```bash
# 1. Generate and audit paired training data.
bash scripts/14_0_submit_paired_rewrite_data.sh

# 2. After obtaining the build job ID, submit the grid and selection stages.
UPSTREAM_BUILD_JOB=<job_id> bash scripts/14_0_submit_paired_rewrite_grid.sh

# 3. After recipe selection, submit final schedules and confirmatory evaluation.
bash scripts/14_0_submit_paired_rewrite_final.sh
```

Use `DRY_RUN=1` with any submission script to inspect commands without creating jobs. GPU entrypoints inspect `nvidia-smi` and use the shared stable-idle-GPU gate before launching one independent process per selected GPU. Runtime caches and intermediate Trainer checkpoints stay under node-local `/var/tmp`; only verified LoRA snapshots and evidence artifacts are published to the project paths on BeeGFS.
