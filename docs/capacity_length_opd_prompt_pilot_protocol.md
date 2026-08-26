# Capacity-length pure OPD prompt pilot

## Status and scope

This document registers the implementation of `capacity_length_opd_prompt_pilot_v1`.
The implementation is complete at the code and configuration level, but no GPU run or
performance result is implied by this document. A result is complete only after the
independent phase-17 audit creates `results/capacity_length_opd_prompt_pilot_v1/pilot/PILOT_COMPLETE`.

The experiment is an exploratory, single-training-seed GSM8K pilot. It must not be
reported as a formal multi-seed result or as evidence on benchmarks other than GSM8K.

## Research question

The experiment asks whether a small model learns a more useful response style when its
own rollout prompt requests concise but complete reasoning, while a larger teacher scores
the sampled response under a common standard task prompt. It compares two independently
trained policies:

- `standard_prompt`: the student samples under the standard GSM8K reasoning prompt.
- `bounded_concise_prompt`: the student samples under a paired relative-length prompt.

Both policies are evaluated greedily under the same standard prompt. Consequently, a
shorter bounded-concise policy at evaluation reflects behavior internalized during
training, not an evaluation-time instruction advantage.

## Method boundary

This is pure dense on-policy distillation with PPO-style clipping. It is not scalar
LLM-as-a-judge PPO, and it is not the repository's hard-CE plus forward-KL logit-KD
pipeline.

For a sampled completion token (y_t), the registered signal is

\[
a_t = \log p_T(y_t \mid y_{<t}, x_{standard})
      - \log p_{old}(y_t \mid y_{<t}, x_{student-arm}).
\]

The updated student uses the importance ratio

\[
r_t(\theta) = \exp(\log p_\theta(y_t)-\log p_{old}(y_t))
\]

and minimizes the negative clipped surrogate

\[
-\operatorname{mean}_t\left[
\min\left(r_t a_t,\operatorname{clip}(r_t,0.8,1.2)a_t\right)
\right].
\]

The teacher always scores the exact student token IDs; no teacher text is sampled. The
implementation disables LoRA dropout for both rollout-time and update-time probability
evaluation so that the importance ratio has a consistent policy reference. During the
update, the parent model remains in training mode to retain gradient checkpointing while
all dropout modules are set to evaluation mode. Generation
and log-probability normalization are restricted to the 151,665 hash-verified shared
tokenizer entries, excluding padded output-head rows.

The following signals are recorded but are never used in the loss:

- gold-answer correctness;
- output length and concise-band compliance;
- answer extraction success;
- EOS and maximum-token truncation status.

There is no scalar teacher reward, correctness reward, length reward, hard-CE auxiliary
loss, or value head. These exclusions are enforced by config validation, persisted in
training artifacts, tested, and checked again by the independent audit.

## Frozen protocol

The registered config is
[`configs/capacity_length_opd_prompt_pilot_v1.json`](../configs/capacity_length_opd_prompt_pilot_v1.json).

- Teacher: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Initial student: raw `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Trainable parameters: LoRA, rank 4, alpha 16, dropout 0.05, all linear modules.
- Training seed: 17.
- Rollouts: four per prompt, temperature 1.0, top-p 1.0, at most 512 new tokens.
- One on-policy epoch over 3,500 training prompts.
- Fresh rollout batch: 16 prompts and 64 trajectories.
- Update mini-batch: eight trajectories.
- Learning rate: `1e-6`; clip ratio: `0.2`; maximum gradient norm: `1.0`.
- Loss aggregation: completion-token mean.
- Maximum prompt-plus-completion training sequence: 2,048 tokens, enforced without truncation.

The expected full-pilot cardinality is 14,000 rollouts per arm, 28,000 total. The last
prompt batch contains 12 prompts, so the registered scheduler has 1,750 optimizer steps
per arm rather than treating the last batch as full.

## Paired relative-length prompt

For every training problem, the frozen raw 1.5B model first produces a greedy response
under the standard prompt. If that response has (L) non-EOS completion tokens, the
bounded-concise target is

- lower bound: `floor(0.70 * L)`;
- upper bound: `ceil(0.90 * L)`;
- both bounds constrained to the inclusive range 96--256 tokens;
- the upper bound is never below the lower bound.

The instruction asks for the shortest complete and independently checkable reasoning,
forbids answer-only responses and repetition, and explicitly gives correctness and
complete reasoning priority over exact band compliance.

Reference generation is sharded across three GPUs, merged in source-index order, and
bound by record, source, config, and file hashes. Smoke references are written under a
separate `smoke/` root and cannot satisfy the pilot merge validator.

## Data separation

| Role | GSM8K support | Use |
|---|---|---|
| Calibration pool | `train[2000:2500]` | Preflight only; the first 100 prompts are frozen for the gate |
| OPD training | `train[3000:6500]` | Unlabeled prompts for rollout and dense teacher scoring |
| Primary evaluation | `train[6500:7473]` | Disjoint registered pilot decision cohort |
| Secondary evaluation | `test[50:1319]` | Adaptive descriptive comparison only |

The two training arms use the same problem support and order. Gold labels are retained
only to compute diagnostics after completions have been sampled.

## Preflight gate

Before training, phase 17.3 uses two GPUs with disjoint 50-prompt shards, generating four
rollouts for each arm on each of 100 frozen calibration prompts. This produces 800 scored
rollouts. The merge phase checks exact support and multiplicity before applying the gate.
The phase records teacher-signal separation, answer-only proxies, answer extraction, and
top-64 student/teacher overlap.

Training is permitted only when at least 70% of bounded-concise rollouts fall inside their
paired length bands and every sampled-token signal is finite. A failed gate does not
authorize adding a length or correctness reward; it blocks the registered pilot and
requires an explicitly revised protocol.

## Evaluation and registered decision

The raw base student and both final OPD adapters are evaluated greedily with a common
standard prompt and a 512-token output cap. The primary paired contrast is
`bounded_concise_prompt - standard_prompt`.

The bounded-concise arm advances only if all conditions hold on the primary cohort:

- lower bound of the paired 95% bootstrap accuracy difference is at least -1 percentage point;
- mean output-token ratio is at most 0.90;
- answer-extraction failure increases by at most 1 percentage point;
- truncation increases by at most 1 percentage point.

Accuracy, output length, extraction failure, and truncation remain separate reported
quantities. The analysis also reports final accuracy against teacher-scored completion
tokens and training-time dense-signal dynamics. Teacher-scored completion tokens are a
cost proxy and do not include prompt tokens or represent total FLOPs.

## Entrypoints and artifacts

Reusable method and analysis logic:

- `src/length_budget_distill/opd.py`
- `src/length_budget_distill/opd_analysis.py`

Phase-first entrypoints:

- `scripts/17_0_prepare_opd_storage.py`
- `scripts/17_0_run_opd_prompt_pilot_c49.sh`
- `scripts/17_1_generate_opd_reference_lengths.py`
- `scripts/17_2_merge_opd_reference_lengths.py`
- `scripts/17_3_preflight_opd_signal.py`
- `scripts/17_3_merge_opd_preflight.py`
- `scripts/17_4_train_opd_policy.py`
- `scripts/17_4_launch_opd_training.py`
- `scripts/17_5_eval_opd_model.py`
- `scripts/17_5_launch_opd_evaluation.py`
- `scripts/17_6_analyze_opd_prompt_pilot.py`
- `scripts/17_7_audit_opd_prompt_pilot.py`

Pilot artifacts are isolated under:

- results: `results/capacity_length_opd_prompt_pilot_v1/pilot/`;
- adapters: `checkpoints/capacity_length_opd_prompt_pilot_v1/pilot/`;
- figures: `figures/capacity_length_opd_prompt_pilot_v1/pilot/`.

Before any GPU stage, phase 17.0 requires the experiment-specific result path to be a
stable symlink to
`/mnt/beegfs/youyang7/projects/small_language_model/results/capacity_length_opd_prompt_pilot_v1`.
It creates that empty target and link only when both are collision-free. It refuses to
replace a directory, retarget a link, or adopt a non-empty BeeGFS directory. The existing
project-wide `checkpoints/` link is also required to resolve under BeeGFS.

Rollout shards are gzip JSONL files. Each shard retains exact completion token IDs,
student and teacher sampled-token log probabilities, dense advantages, prompt contexts,
and diagnostic-only outcome fields. Adapter completion markers bind the protocol,
reference and rollout manifests, implementation sources, training metrics, and final LoRA
file hashes. Training resume state remains node-local under `/var/tmp` and is removed only
after a hash-verified adapter is published.

## C49 execution

The orchestrator uses the user's existing C49 allocation and does not cancel or close it:

```bash
DRY_RUN=1 ALLOCATION_JOB_ID=<active-job-id> \
  bash scripts/17_0_run_opd_prompt_pilot_c49.sh

ALLOCATION_JOB_ID=<active-job-id> \
  bash scripts/17_0_run_opd_prompt_pilot_c49.sh
```

Every stage enters the allocation with `srun --jobid=<active-job-id> --overlap`. Before
each GPU launch, the runner prints `nvidia-smi` process evidence and selects GPUs by
remaining memory across two checks. On C49 this deliberately treats the user's `gg`
processes as keepalives and subtracts their memory through `memory.free`; it does not use
their utilization as an exclusion criterion.

Reference generation uses three independent GPU shards, the two OPD policies train on
two separate GPUs, and the three evaluation models run on three separate GPUs. If
training was preempted after valid node-local resume state was saved, set `RESUME=1` for
the training stage. Existing completed artifacts are validated and skipped; incomplete or
hash-mismatched artifacts are never overwritten automatically.

The final marker is created only after the audit verifies reference support, preflight
gate, rollout multiplicity, exact sampled-token signal construction, optimizer counts,
adapter hashes, common-prompt evaluation support, paired statistics, reports, and all PNG
and PDF figure hashes.

## Method references

- Agarwal et al., [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649).
- Thinking Machines Lab, [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).
- MiniMax, [Rethinking On-Policy Distillation: A Policy-Gradient Perspective](https://arxiv.org/abs/2604.13016).
