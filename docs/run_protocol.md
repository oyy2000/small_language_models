# Run Protocol

This project does not require a separate config gate or command-line gate before
real runs.

## Qwen Runs

Runs use the configured teacher, dataset, tokenizer, and student directly.
Before launching generation or SFT, review the relevant config file:

- `configs/real_length_budget_template.json`
- `configs/student_sft_template.json`

Generation or training may download Hugging Face datasets or model weights,
load GPU memory, and write traces, summaries, or checkpoints under `results/`
and `checkpoints/`.
