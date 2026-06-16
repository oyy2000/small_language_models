# TODO

## Immediate

- Review the prefilled Qwen teacher model for the first real run.
- Review the prefilled GSM8K training slice and choose an evaluation set.
- Review the prefilled Qwen student model and LoRA settings.
- Decide the first real length budgets, for example short, medium, and long token caps.
- Decide whether to keep only verified-correct traces or also save incorrect traces for analysis.

## Minimal Real Pilot

- Generate short, medium, and long teacher traces on a small training slice.
- Verify final answers with a task-specific verifier.
- Train the same student under equal-example and equal-token settings.
- Evaluate raw accuracy and accuracy gain per supervised solution token.

## Analysis

- Plot average solution tokens per budget.
- Plot verified-correct rate per budget.
- Plot downstream accuracy gain per million SFT tokens.
- Add paired confidence intervals after multiple seeds are available.
