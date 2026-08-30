# Phase 17 OPD Prompt Pilot: C49 Preflight Report

Date: 2026-08-28  
Experiment: `capacity_length_opd_prompt_pilot_v1`  
Evidence level: exploratory single-seed GSM8K preflight  
Execution node: C49, allocation `277424`

## Executive summary

The registered reference-generation and preflight stages completed successfully at the
artifact level. The bounded-concise prompt did not satisfy the registered behavioral
gate: only 12.0% of its 400 rollouts fell inside the requested relative length band,
compared with the 70.0% minimum. The prompt usually made the student longer rather than
shorter. Consequently, the original v1 supervisor correctly stopped before PPO training
and did not create `PREFLIGHT_COMPLETE`, a training launcher manifest, adapters, evaluation
outputs, or `PILOT_COMPLETE`.

This is a completed failed-preflight result, not a completed PPO performance experiment.
The teacher signal was finite and diagnostically useful, but that does not override the
failed prompt-adherence gate.

## Registered method

The fixed student is Qwen2.5-1.5B-Instruct and the fixed teacher is
Qwen2.5-7B-Instruct. The two student rollout contexts are:

- `standard_prompt`
- `bounded_concise_prompt`, requesting 70--90% of the frozen base-student greedy
  reference length, clamped to 96--256 tokens

The student samples exact token trajectories. The teacher scores those same sampled token
IDs under the common standard prompt. The objective is sampled-token reverse-KL with
PPO-style clipping. Correctness and length are diagnostics only; neither enters the loss.
There is no scalar reward, value head, hard cross-entropy loss, correctness reward, or
length reward.

The immutable experiment settings are in
`configs/capacity_length_opd_prompt_pilot_v1.json`.

## Execution and artifact status

### Reference generation

- Three independent C49 shards completed.
- Shard cardinalities were 1,167, 1,167, and 1,166.
- The merged training reference set contains 3,500 records.
- `results/capacity_length_opd_prompt_pilot_v1/pilot/references/REFERENCES_COMPLETE`
  exists.
- The merged data and manifest are:
  - `results/capacity_length_opd_prompt_pilot_v1/pilot/references/training_references.jsonl`
  - `results/capacity_length_opd_prompt_pilot_v1/pilot/references/reference_manifest.json`

### Preflight

- Two shards completed with 50 prompts and 400 rollouts each.
- The merge contains 100 unique problem IDs and 800 unique
  `(problem_id, arm, candidate_index)` rollout identities.
- The merged rollout SHA-256 matches the registered preflight manifest.
- The teacher signal is finite.
- The merge status is `failed` because prompt adherence did not reach the registered
  threshold.
- The authoritative artifacts are:
  - `results/capacity_length_opd_prompt_pilot_v1/pilot/preflight/preflight_rollouts.jsonl.gz`
  - `results/capacity_length_opd_prompt_pilot_v1/pilot/preflight/preflight_summary.json`
  - `results/capacity_length_opd_prompt_pilot_v1/pilot/preflight/preflight_manifest.json`

### Training and evaluation

The original v1 PPO training did not start. The following required completion artifacts
are intentionally absent:

- `results/capacity_length_opd_prompt_pilot_v1/pilot/preflight/PREFLIGHT_COMPLETE`
- `results/capacity_length_opd_prompt_pilot_v1/pilot/training/training_launcher_manifest.json`
- `results/capacity_length_opd_prompt_pilot_v1/PILOT_COMPLETE`

No v1 adapter or post-training performance claim is available.

## Preflight results

| Diagnostic | Standard prompt | Bounded-concise prompt | Concise minus standard |
|---|---:|---:|---:|
| Rollouts | 400 | 400 | 0 |
| Mean output tokens | 227.70 | 271.70 | +44.01 |
| Median output tokens | 201.5 | 253.0 | +51.5 |
| Diagnostic accuracy | 55.50% | 52.25% | -3.25 pp |
| Answer-extraction failure | 3.75% | 9.50% | +5.75 pp |
| Hit 512-token cap | 3.50% | 5.50% | +2.00 pp |
| Mean sampled-token teacher advantage | -0.5868 | -0.4522 | +0.1346 |

For the bounded-concise arm:

- 48/400 rollouts were in band: 12.0%.
- 306/400 rollouts exceeded the upper bound: 76.5%.
- 46/400 rollouts fell below the lower bound: 11.5%.
- In paired `(problem_id, candidate_index)` comparisons, the concise arm was longer in
  66.25% of pairs, shorter in 32.25%, and equal in 1.50%.
- The mean paired token difference was +44.01 tokens and the median paired difference was
  +34.5 tokens.

The global teacher-reward AUC for diagnostic correctness was 0.7728. Mean teacher
advantage was -0.3563 for correct rollouts and -0.7102 for incorrect rollouts. This
supports the presence of a useful teacher signal, but the signal result is separate from
prompt adherence and does not establish that the concise prompt is trainable or better.

## Interpretation

The bounded-concise instruction backfired for the frozen base student. It increased mean
length, increased truncation and extraction failures, and reduced diagnostic accuracy.
The dominant failure mode was exceeding the upper bound, not producing answers that were
too short. Under the registered decision rule, stopping before PPO was therefore the
correct outcome.

The standard arm is the more reliable observed rollout baseline at this stage. This does
not prove that standard-prompt PPO is superior after learning because no PPO optimization
or common-prompt post-training evaluation has yet occurred.

## C49 execution notes

The model snapshots were staged to node-local `/var/tmp` before loading, avoiding direct
BeeGFS Safetensors mmap. Reference generation used three GPUs. Preflight used GPU1 and
GPU2 after repeated physical-memory checks. Each dense 7B-teacher plus 1.5B-student
preflight worker used about 18.25 GiB, with approximately 4.2 GiB free at observed scoring
peaks after accounting for existing processes. No OOM, SIGBUS, or other-user process
interference occurred.

The duplicate C30 DAG (`277457`--`277460`) was cancelled after C49 execution was verified.
The user-owned C49 allocation and `gg` keepalive processes were left open.

## User-authorized continuation

After reviewing the failed gate, the user explicitly requested that the experiment
continue. Any subsequent optimization must be labeled as a gate-waived exploratory
continuation and must use separate result and checkpoint roots. It must retain immutable
links and hashes to this failed preflight, must not create or imply the original v1
`PREFLIGHT_COMPLETE`, and must not be reported as confirmatory evidence or as a passed
registered pilot.

The continuation is registered in
`configs/capacity_length_opd_prompt_gate_waived_continuation_v1.json`. Its outputs are
isolated under `results/capacity_length_opd_prompt_gate_waived_continuation_v1` and
`checkpoints/capacity_length_opd_prompt_gate_waived_continuation_v1`. The continuation
config is hash-bound to the original protocol, the 3,500-record reference manifest, and
the failed 800-rollout preflight manifest and summary.

At 23:14 EDT on 2026-08-28, a C49 supervisor was started inside allocation `277424` using
`srun --jobid=277424 --overlap`. It first runs a one-prompt-batch smoke for both prompt
arms and starts the full two-arm continuation only if both smoke adapters pass their
artifact validation. At launch, all three physical GPUs had less than the registered
22,000 MiB free-memory threshold because another user-owned project run and unrelated
processes were active. The supervisor waited at the repeated physical GPU admission gate
without terminating or interfering with any process.

The two-arm smoke subsequently ran on GPU0 and GPU1. Both arms completed 16 prompts and 64
rollouts. The first publication attempt found an implementation error after optimization:
identical hash-bound reference fields were treated as prohibited evidence overrides. A
second attempt exposed the same stable-path issue for `/home` project paths backed by a
BeeGFS symbolic link. Both failed attempts were after the batch checkpoint and neither
lost or regenerated the rollouts. The publication validator was corrected to accept
identical fields, reject conflicting fields, and preserve stable project paths; 18 OPD
tests then passed.

The recovered smoke completed at 23:29 EDT. Its validated diagnostics were:

| Diagnostic | Standard prompt | Bounded-concise prompt |
|---|---:|---:|
| Prompts | 16 | 16 |
| Rollouts | 64 | 64 |
| Sampled tokens | 17,728 | 19,830 |
| Mean output tokens | 276.11 | 308.94 |
| Diagnostic accuracy | 48.44% | 40.62% |
| In-band rate | 12.50% | 10.94% |

This smoke validates the training, resume, publication, and provenance path; it does not
reverse the failed preflight or constitute a performance result. At 00:44 EDT on
2026-08-29, the user explicitly requested immediate execution without further waiting.
The waiting OPD step `277424.44` was cancelled without changing allocation `277424`, its
`gg` keepalive, or the concurrent ranked-evaluation step. The full continuation was then
launched on GPU1 and GPU2 with a user-authorized 21,000 MiB one-check admission override.
Its launcher log is
`results/capacity_length_opd_prompt_gate_waived_continuation_v1/slurm/c49_full_training_21000mib.log`.
Because allocation `277424` ends at 22:06 EDT on 2026-08-29, Slurm job `277628` is queued
with `afterany:277424` on C49 for a hash-preserving resume. It will skip the already
validated smoke and either resume symmetric per-batch full-training checkpoints or exit
for audit if the two arms have an unsafe asymmetric incomplete state.
