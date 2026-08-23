# Capacity-Length MATH-Mix Pilot Protocol

## Status and purpose

This is an exploratory, single-seed extension of the completed
`capacity_length_factorial_seed17_v1` experiment. It tests whether adding
competition-math supervision reduces narrow GSM8K specialization for the 7B
teacher length conditions. It does not modify or supersede the frozen GSM8K
formal artifacts and does not estimate training-seed variability.

MATH-500 is reserved for evaluation. Training uses only the official MATH
train split from `EleutherAI/hendrycks_math` at revision
`21a5633873b6a120296cce3e2df9d5550074f4a3`.

## Frozen pilot design

- Source pool: 1,000 MATH train problems, sampled deterministically across the
  joint subject-by-level strata with seed `20260823`.
- Teacher: Qwen2.5-7B-Instruct snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Length conditions: 128, 256, and 512 solution tokens.
- Generation: three candidates per problem-length request, temperature 0.7,
  top-p 0.95, and identity-derived seeds. The complete matrix contains exactly
  9,000 raw candidates.
- Verification: `math-verify[antlr4_13_2]==0.9.0`; selection takes the shortest
  correct, length-compliant candidate.
- Common support: a problem enters SFT only if all three length conditions have
  an eligible selected trace. Training is blocked if fewer than 300 problems
  remain.
- Student: Qwen2.5-1.5B-Instruct with the existing rank-4 LoRA, learning rate
  `2e-5`, batch size 4, one epoch, completion-only loss, and seed 17.

The pilot trains six new adapters. Equal-example combines all 881 historical
GSM8K common-intersection records with every MATH common-intersection record of
the corresponding length. Equal-token reuses the historical GSM8K seed-17
equal-token subset and independently balances the MATH contribution to the
smallest MATH condition-level token total. Balancing within each source prevents
the GSM8K/MATH mixture from drifting across lengths.

## Frozen evaluation suite

All 13 models are evaluated: the base student, six historical 7B GSM8K-only
adapters, and six new MATH-mixed adapters.

- GSM8K: 200 deterministic questions from the historical formal window
  `test[50:1319]`, greedy decoding, 512 new-token limit. This is an adaptive
  diagnostic because the official test has already been observed.
- MATH-500: 100 deterministic subject-by-level stratified questions from the
  test split, greedy decoding, 1,024 new-token limit.
- AIME 2025: all 30 questions, greedy decoding, 1,024 new-token limit.

The report includes accuracy with Wilson intervals, extraction failures,
maximum-token hits, output length, paired bootstrap intervals, exact McNemar
tests, and Holm adjustment over the six GSM-only versus MATH-mixed adapter pairs
within each dataset. AIME results remain descriptive because the cohort is only
30 problems and may exhibit a floor effect.

## Artifact contract

Large results and adapters are stored on BeeGFS through the stable project paths
`results/capacity_length_math_mix_pilot_v1` and
`checkpoints/capacity_length_math_mix_pilot_v1`. Publication figures are stored
under `figures/capacity_length_math_mix_pilot_v1`.

Completion requires a 1,000-row source manifest, all raw shard hashes, a passed
selection audit with at least 300 common problems, six complete SFT datasets,
six hash-verified adapters, 39 model-by-dataset prediction artifacts, analysis
PNG/PDF/CSV/report artifacts, and a passed independent completion audit. The
completion marker labels the result `exploratory_single_seed_pilot`.

## Main commands

Dry-run the complete Slurm DAG:

```bash
DRY_RUN=1 bash scripts/11_0_submit_math_mix_pilot.sh
```

Submit the pilot:

```bash
bash scripts/11_0_submit_math_mix_pilot.sh
```

The launchers inspect physical GPU occupancy through the shared stable-idle-GPU
gate and use one process per selected GPU with disjoint shards or model runs.
